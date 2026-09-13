"""Gene-grouped CV folds and nested XGBoost cross-validation.

Builds on ``features`` (matrices, encoding, fit helpers). Assigns each gene
to a fold, then runs nested CV: outer folds estimate generalization; inner CV on
the remaining folds selects hyperparameters.

Writes:
  data/processed/cv_folds.parquet   (column ``fold`` ∈ {0 .. n_folds-1})
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

from features import (
    DEFAULT_SEED,
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    TARGET_COLUMN,
    _keep_columns,
    classification_metrics_at_threshold,
    data_processed,
    fit_predict,
    fit_predict_fixed_trees,
    gene_stats,
    pick_threshold,
    prepare_modeling_frame,
    resolve_position_matched_path,
    split_summary,
)

CV_FOLDS_OUT = data_processed / "cv_folds.parquet"

DEFAULT_N_FOLDS = 5
DEFAULT_N_TUNING_TRIALS = 12

PARAM_DISTRIBUTIONS = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_lambda": [0.5, 1.0, 5.0],
}


def assign_cv_folds(
    df: pd.DataFrame,
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
    gene_col: str = GROUP_COLUMN,
    label_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Assign every gene to a fold ``0 .. n_folds-1`` (gene-grouped GroupKFold style)."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    out = df.copy()
    stats = gene_stats(out, gene_col, label_col, seed)

    n_total = len(out)
    n_path_total = int((out[label_col] == "pathogenic").sum())
    target = n_total / n_folds
    path_target = n_path_total / n_folds
    counts = np.zeros(n_folds, dtype=float)
    path_counts = np.zeros(n_folds, dtype=float)
    assignment: dict[str, int] = {}

    for row in stats.itertuples(index=False):
        deficits = target - counts
        path_deficits = path_target - path_counts
        fold = int(np.lexsort((-path_deficits, -deficits))[0])
        assignment[row[0]] = fold
        counts[fold] += row.n
        path_counts[fold] += row.n_pathogenic

    out["fold"] = out[gene_col].map(assignment).astype("Int64")
    if out["fold"].isna().any():
        raise RuntimeError("Some rows did not receive a fold assignment.")
    return out


def iter_cv_folds(
    df: pd.DataFrame,
    *,
    n_folds: int | None = None,
    valid_fold_offset: int = 1,
) -> Iterator[dict]:
    """Yield train / valid / test frames for each gene-grouped CV fold.

    For fold ``k``, test is fold ``k`` and valid is the deterministic neighbour
    ``(k + valid_fold_offset) % n_folds``. Set ``valid_fold_offset=0`` to skip a
    dedicated valid fold.
    """
    if "fold" not in df.columns:
        raise ValueError("DataFrame is missing a 'fold' column; call assign_cv_folds first.")

    folds = sorted(int(f) for f in df["fold"].dropna().unique())
    if n_folds is None:
        n_folds = max(folds) + 1
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if valid_fold_offset < 0 or valid_fold_offset >= n_folds:
        raise ValueError(f"valid_fold_offset must be in [0, {n_folds}), got {valid_fold_offset}")

    for test_fold in range(n_folds):
        if valid_fold_offset == 0:
            valid_fold = None
            train_mask = df["fold"] != test_fold
            valid_df = df.iloc[0:0].copy()
        else:
            valid_fold = (test_fold + valid_fold_offset) % n_folds
            train_mask = ~df["fold"].isin([test_fold, valid_fold])
            valid_df = df[df["fold"] == valid_fold].reset_index(drop=True)

        yield {
            "fold": test_fold,
            "valid_fold": valid_fold,
            "train": df[train_mask].reset_index(drop=True),
            "valid": valid_df,
            "test": df[df["fold"] == test_fold].reset_index(drop=True),
        }


def run_cv_folds(
    *,
    input_path: Path | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Prepare, assign gene-grouped CV folds, write parquet, and return the frame."""
    input_path = input_path or resolve_position_matched_path()
    raw = pd.read_parquet(input_path)
    modeling = prepare_modeling_frame(raw)
    folded = assign_cv_folds(modeling, n_folds=n_folds, seed=seed)
    folded = folded[_keep_columns("fold")].copy()

    data_processed.mkdir(parents=True, exist_ok=True)
    folded.to_parquet(CV_FOLDS_OUT, index=False)
    return folded


def nested_tune(
    pool: pd.DataFrame,
    *,
    n_trials: int = DEFAULT_N_TUNING_TRIALS,
    seed: int = DEFAULT_SEED,
    param_distributions: dict | None = None,
    threshold_method: str = "youden",
) -> tuple[dict, float, int, float, pd.DataFrame]:
    """Inner gene-grouped CV on ``pool``.

    Returns best hyperparameters, mean inner AUC, refit tree budget
    (median of ``best_iteration + 1``), and a probability cutoff — nested
    thresholding via the **median of per-inner-fold** cutoffs (not pooled
    scores; fold score scales differ). Default cutoff rule is Youden's J;
    plain F1 often collapses to near-all-pathogenic on path-heavy folds.

    Each fit sets ``scale_pos_weight = n_benign / n_pathogenic`` from that fit's
    training labels unless the sampled params already include it.
    """
    inner_folds = sorted(int(f) for f in pool["fold"].unique())
    if len(inner_folds) < 2:
        raise ValueError("Need >= 2 folds in the outer-train pool for inner CV.")

    distributions = param_distributions or PARAM_DISTRIBUTIONS
    sampler = ParameterSampler(distributions, n_iter=n_trials, random_state=seed)
    trial_rows = []
    best_params: dict | None = None
    best_inner_auc = -np.inf
    best_n_estimators = 1
    best_threshold = 0.5

    for trial_idx, params in enumerate(sampler):
        params = dict(params)
        fold_aucs = []
        fold_n_trees = []
        fold_thresholds = []
        for i, inner_test_fold in enumerate(inner_folds):
            candidates = [f for f in inner_folds if f != inner_test_fold]
            valid_fold = candidates[i % len(candidates)]

            train_df = pool[~pool["fold"].isin([inner_test_fold, valid_fold])]
            valid_df = pool[pool["fold"] == valid_fold]
            test_df = pool[pool["fold"] == inner_test_fold]
            if train_df.empty:
                train_df = pool[pool["fold"] != inner_test_fold]
                valid_df = test_df

            model, proba_test, _, inner_test_auc = fit_predict(
                train_df, valid_df, test_df, params=params, seed=seed
            )
            fold_aucs.append(inner_test_auc)
            fold_n_trees.append(int(model.best_iteration) + 1)
            y_inner = (test_df[TARGET_COLUMN] == "pathogenic").astype(int).to_numpy()
            fold_thresholds.append(
                pick_threshold(y_inner, proba_test, method=threshold_method)
            )

        mean_auc = float(np.mean(fold_aucs))
        median_trees = int(np.median(fold_n_trees))
        threshold = float(np.median(fold_thresholds))
        trial_rows.append(
            {
                "trial": trial_idx,
                "inner_mean_roc_auc": mean_auc,
                "inner_std_roc_auc": float(np.std(fold_aucs)),
                "inner_median_n_estimators": median_trees,
                "inner_threshold": threshold,
                **params,
            }
        )
        if mean_auc > best_inner_auc:
            best_inner_auc = mean_auc
            best_params = params
            best_n_estimators = max(1, median_trees)
            best_threshold = threshold

    assert best_params is not None
    return (
        best_params,
        best_inner_auc,
        best_n_estimators,
        best_threshold,
        pd.DataFrame(trial_rows),
    )


def run_nested_cv(
    folded: pd.DataFrame | None = None,
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    n_trials: int = DEFAULT_N_TUNING_TRIALS,
    seed: int = DEFAULT_SEED,
    param_distributions: dict | None = None,
    threshold_method: str = "youden",
    verbose: bool = True,
) -> dict:
    """Nested gene-grouped CV: tune θ, n_estimators, and decision threshold inwardly.

    Fits use ``scale_pos_weight`` from training-label counts. Outer refit trains on
    the full pool with the inner-chosen tree budget, then applies the inner-chosen
    probability cutoff (default Youden's J; median of per-inner-fold thresholds)
    to the outer test fold.
    """
    if folded is None:
        folded = run_cv_folds(n_folds=n_folds, seed=seed)

    fold_rows = []
    oof_parts = []
    importances = []
    tuning_logs = []

    for outer_fold in range(n_folds):
        outer_test_df = folded[folded["fold"] == outer_fold].reset_index(drop=True)
        pool = folded[folded["fold"] != outer_fold].reset_index(drop=True)

        best_params, best_inner_auc, n_estimators, threshold, trials = nested_tune(
            pool,
            n_trials=n_trials,
            seed=seed + outer_fold,
            param_distributions=param_distributions,
            threshold_method=threshold_method,
        )
        trials = trials.copy()
        trials.insert(0, "outer_fold", outer_fold)
        tuning_logs.append(trials)

        model, proba_test, test_auc = fit_predict_fixed_trees(
            pool,
            outer_test_df,
            params=best_params,
            n_estimators=n_estimators,
            seed=seed,
        )
        y_test = (outer_test_df[TARGET_COLUMN] == "pathogenic").astype(int)
        cls = classification_metrics_at_threshold(y_test, proba_test, threshold)

        row = {
            "fold": outer_fold,
            "n_train": len(pool),
            "n_test": len(outer_test_df),
            "n_test_genes": outer_test_df[GROUP_COLUMN].nunique(),
            "best_inner_roc_auc": best_inner_auc,
            "n_estimators": n_estimators,
            "threshold": threshold,
            "roc_auc": test_auc,
            "pr_auc": float(average_precision_score(y_test, proba_test)),
            "accuracy": cls["accuracy"],
            "precision": cls["precision"],
            "recall": cls["recall"],
            "f1": cls["f1"],
            "tn": cls["tn"],
            "fp": cls["fp"],
            "fn": cls["fn"],
            "tp": cls["tp"],
            **{f"param_{key}": value for key, value in best_params.items()},
        }
        fold_rows.append(row)
        if verbose:
            print(
                f"outer {outer_fold}: thr={threshold:.3f}  "
                f"ROC={row['roc_auc']:.3f}  P={row['precision']:.3f}  "
                f"R={row['recall']:.3f}  Acc={row['accuracy']:.3f}  "
                f"n_estimators={n_estimators}"
            )

        oof_parts.append(
            pd.DataFrame(
                {
                    "VariationID": outer_test_df["VariationID"].values,
                    "gene": outer_test_df[GROUP_COLUMN].values,
                    "fold": outer_fold,
                    "y_true": y_test.values,
                    "y_proba": proba_test,
                    "threshold": threshold,
                    "y_pred": (proba_test >= threshold).astype(int),
                }
            )
        )
        importances.append(
            pd.Series(
                model.feature_importances_,
                index=FEATURE_COLUMNS,
                name=f"fold_{outer_fold}",
            )
        )

    fold_metrics = pd.DataFrame(fold_rows).set_index("fold")
    oof = pd.concat(oof_parts, ignore_index=True)
    tuning_log = pd.concat(tuning_logs, ignore_index=True)
    importance_df = pd.concat(importances, axis=1)
    importance_df["mean"] = importance_df.mean(axis=1)
    importance_df["std"] = importance_df.std(axis=1)
    importance_df = importance_df.sort_values("mean", ascending=False)

    confusion_total = fold_metrics[["tn", "fp", "fn", "tp"]].sum().astype(int)
    cls_summary = fold_metrics[
        ["threshold", "accuracy", "precision", "recall", "f1"]
    ].agg(["mean", "std", "min", "max"])

    return {
        "folded": folded,
        "fold_metrics": fold_metrics,
        "oof": oof,
        "tuning_log": tuning_log,
        "importance": importance_df,
        "confusion_total": confusion_total,
        "classification_summary": cls_summary,
        "threshold_method": threshold_method,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    folded = run_cv_folds(n_folds=args.n_folds, seed=args.seed)
    print("Read:", resolve_position_matched_path())
    print("Wrote:", CV_FOLDS_OUT)
    print(split_summary(folded, partition_col="fold").to_string(index=False))
    gene_folds = folded.groupby(GROUP_COLUMN)["fold"].nunique()
    assert (gene_folds == 1).all()
    print(f"CV gene overlap across {args.n_folds} folds: none")


if __name__ == "__main__":
    main()
