"""Gene-grouped train/valid/test split and XGBoost holdout fit/eval.

Reads data/processed/clinvar_uniprot_position_matched.parquet, keeps matched
labelled variants (one row per VariationID), assigns each gene wholly to
train/valid/test, and provides helpers to build matrices and fit XGBoost.

Writes:
  data/processed/train.parquet
  data/processed/valid.parquet
  data/processed/test.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data/processed"

POSITION_MATCHED_IN = data_processed / "clinvar_uniprot_position_matched.parquet"
TRAIN_OUT = data_processed / "train.parquet"
VALID_OUT = data_processed / "valid.parquet"
TEST_OUT = data_processed / "test.parquet"

DEFAULT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_SEED = 42

# Gene-proxy / high-cardinality identity-like columns are omitted: under
# gene-grouped evaluation they mostly memorize training genes rather than
# transferring biology (Chromosome/OriginSimple/Length; free-text domain notes).
NUMERIC_FEATURES = [
    "protein_position",
    "relative_protein_position",
    "distance_to_closest_feature",
]

BOOLEAN_FEATURES = [
    "has_protein_position",
    "in_domain",
    "in_region",
    "in_zinc_finger",
    "in_active_site",
    "in_binding_site",
    "in_disulfide",
    "in_mod_res",
    "in_functional_site",
    "in_any_feature",
]

CATEGORICAL_FEATURES = [
    "closest_feature_type",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "label"
GROUP_COLUMN = "gene"
ID_COLUMN = "VariationID"

META_COLUMNS = [
    ID_COLUMN,
    GROUP_COLUMN,
    "Entry",
    "Name",
    "GeneSymbol",
    "ClinicalSignificance",
    "match_type",
]

DEFAULT_XGB_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
}


def prepare_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to matched labelled variants and deduplicate to one row per VariationID."""
    frame = df[df["match_type"] == "both"].copy()
    frame = frame[frame[TARGET_COLUMN].isin(["pathogenic", "benign"])].copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype("Int64")

    frame["_rank_pos"] = (~frame["has_protein_position"]).astype(int)
    frame = (
        frame.sort_values([ID_COLUMN, "_rank_pos", "Entry"], kind="mergesort")
        .drop_duplicates(ID_COLUMN, keep="first")
        .drop(columns="_rank_pos")
        .reset_index(drop=True)
    )

    length = pd.to_numeric(frame["Length"], errors="coerce")
    pos = pd.to_numeric(frame["protein_position"], errors="coerce")
    frame["relative_protein_position"] = (pos / length).where(length > 0)

    if frame[GROUP_COLUMN].isna().any():
        raise ValueError("Modeling rows are missing gene labels needed for the group split.")

    return frame


def gene_stats(df: pd.DataFrame, gene_col: str, label_col: str, seed: int) -> pd.DataFrame:
    """Per-gene size / label counts, largest first, seeded shuffle for equal sizes."""
    rng = np.random.default_rng(seed)
    stats = (
        df.groupby(gene_col, sort=False)
        .agg(
            n=(label_col, "size"),
            n_pathogenic=(label_col, lambda s: int((s == "pathogenic").sum())),
        )
        .reset_index()
    )
    stats["_shuffle"] = rng.random(len(stats))
    return stats.sort_values(["n", "_shuffle"], ascending=[False, True])


def split_by_gene(
    df: pd.DataFrame,
    *,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
    gene_col: str = GROUP_COLUMN,
    label_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Assign every gene to train, valid, or test; approximate size and label balance."""
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1, got {ratios}")

    out = df.copy()
    stats = gene_stats(out, gene_col, label_col, seed)

    n_total = len(out)
    n_path_total = int((out[label_col] == "pathogenic").sum())
    targets = {
        "train": ratios[0] * n_total,
        "valid": ratios[1] * n_total,
        "test": ratios[2] * n_total,
    }
    path_targets = {
        "train": ratios[0] * n_path_total,
        "valid": ratios[1] * n_path_total,
        "test": ratios[2] * n_path_total,
    }
    counts = {"train": 0.0, "valid": 0.0, "test": 0.0}
    path_counts = {"train": 0.0, "valid": 0.0, "test": 0.0}
    assignment: dict[str, str] = {}

    for row in stats.itertuples(index=False):
        def score(split: str) -> tuple[float, float]:
            return (
                targets[split] - counts[split],
                path_targets[split] - path_counts[split],
            )

        split = max(("train", "valid", "test"), key=score)
        assignment[row[0]] = split
        counts[split] += row.n
        path_counts[split] += row.n_pathogenic

    out["split"] = out[gene_col].map(assignment)
    if out["split"].isna().any():
        raise RuntimeError("Some rows did not receive a split assignment.")
    return out


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model matrix X and binary target y (1 = pathogenic)."""
    X = df[FEATURE_COLUMNS].copy()

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in BOOLEAN_FEATURES:
        X[col] = X[col].fillna(False).astype(bool)

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string").fillna("__MISSING__").astype("category")

    y = (df[TARGET_COLUMN] == "pathogenic").astype(int)
    y.name = TARGET_COLUMN
    return X, y


def align_categories_to_train(
    train: pd.DataFrame,
    *others: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Restrict categorical levels to those seen in ``train``; map others to ``__MISSING__``."""
    train = train.copy()
    others = tuple(frame.copy() for frame in others)
    for col in CATEGORICAL_FEATURES:
        cats = list(train[col].cat.categories)
        if "__MISSING__" not in cats:
            cats = cats + ["__MISSING__"]
        train[col] = train[col].cat.set_categories(cats)
        for frame in others:
            values = frame[col].astype("string")
            values = values.where(values.isin(cats), "__MISSING__")
            frame[col] = pd.Categorical(values, categories=cats)
    return (train, *others)


def split_summary(df: pd.DataFrame, *, partition_col: str = "split") -> pd.DataFrame:
    """Per-partition counts useful for logging and notebooks."""
    rows = []
    for partition, part in df.groupby(partition_col, sort=True):
        rows.append(
            {
                partition_col: partition,
                "n_variants": len(part),
                "n_genes": part[GROUP_COLUMN].nunique(),
                "pct_pathogenic": 100 * (part[TARGET_COLUMN] == "pathogenic").mean(),
                "pct_with_protein_position": 100 * part["has_protein_position"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    holdout_order = {"train": 0, "valid": 1, "test": 2}
    if partition_col == "split" and summary[partition_col].isin(holdout_order).all():
        summary = summary.sort_values(
            partition_col, key=lambda s: s.map(holdout_order)
        ).reset_index(drop=True)
    return summary


def _keep_columns(*extra: str) -> list[str]:
    # Keep Length on disk so relative_protein_position can be recomputed if needed;
    # Length itself is not a model feature.
    return list(
        dict.fromkeys(META_COLUMNS + list(extra) + FEATURE_COLUMNS + [TARGET_COLUMN, "Length"])
    )


def run_split(
    *,
    input_path: Path = POSITION_MATCHED_IN,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """Prepare, holdout-split, write parquets, and return the three frames."""
    raw = pd.read_parquet(input_path)
    modeling = prepare_modeling_frame(raw)
    split_df = split_by_gene(modeling, ratios=ratios, seed=seed)
    split_df = split_df[_keep_columns("split")].copy()

    data_processed.mkdir(parents=True, exist_ok=True)
    frames = {
        "train": split_df[split_df["split"] == "train"].reset_index(drop=True),
        "valid": split_df[split_df["split"] == "valid"].reset_index(drop=True),
        "test": split_df[split_df["split"] == "test"].reset_index(drop=True),
    }
    frames["train"].to_parquet(TRAIN_OUT, index=False)
    frames["valid"].to_parquet(VALID_OUT, index=False)
    frames["test"].to_parquet(TEST_OUT, index=False)
    return frames


def make_xgb_model(
    *,
    seed: int = DEFAULT_SEED,
    n_estimators: int = 500,
    early_stopping_rounds: int | None = 50,
    n_jobs: int = 4,
    **params,
) -> XGBClassifier:
    """Build an XGBClassifier with project defaults; ``params`` override defaults."""
    model_params = {**DEFAULT_XGB_PARAMS, **params}
    kwargs: dict = dict(
        n_estimators=n_estimators,
        objective="binary:logistic",
        eval_metric="auc",
        enable_categorical=True,
        tree_method="hist",
        random_state=seed,
        n_jobs=n_jobs,
        **model_params,
    )
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    return XGBClassifier(**kwargs)


def fit_predict(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict | None = None,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[XGBClassifier, np.ndarray, float, float]:
    """Fit on train (early stop on valid); return model, test proba, valid AUC, test AUC."""
    params = dict(params or {})
    X_train, y_train = build_feature_matrix(train_df)
    X_valid, y_valid = build_feature_matrix(valid_df)
    X_test, y_test = build_feature_matrix(test_df)
    X_train, X_valid, X_test = align_categories_to_train(X_train, X_valid, X_test)

    model = make_xgb_model(seed=seed, **params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    proba_valid = model.predict_proba(X_valid)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]
    valid_auc = float(roc_auc_score(y_valid, proba_valid))
    test_auc = float(roc_auc_score(y_test, proba_test))
    return model, proba_test, valid_auc, test_auc


def fit_predict_fixed_trees(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict | None = None,
    *,
    n_estimators: int,
    seed: int = DEFAULT_SEED,
) -> tuple[XGBClassifier, np.ndarray, float]:
    """Fit on all of ``train_df`` with a fixed tree budget (no early-stopping holdout)."""
    params = dict(params or {})
    n_estimators = max(1, int(n_estimators))
    X_train, y_train = build_feature_matrix(train_df)
    X_test, y_test = build_feature_matrix(test_df)
    X_train, X_test = align_categories_to_train(X_train, X_test)

    model = make_xgb_model(
        seed=seed,
        n_estimators=n_estimators,
        early_stopping_rounds=None,
        **params,
    )
    model.fit(X_train, y_train, verbose=False)
    proba_test = model.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, proba_test))
    return model, proba_test, test_auc


def pick_threshold(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    *,
    method: str = "youden",
) -> float:
    """Choose a probability cutoff from labelled scores (no test leakage if scores are OOF).

    Methods:
      - ``youden``: maximize TPR - FPR (Youden's J)
      - ``f1``: maximize F1 on the precision–recall curve
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.5

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        # sklearn may append an extra threshold; align to finite scores.
        j = tpr - fpr
        return float(thresholds[int(np.argmax(j))])
    if method == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        if thresholds.size == 0:
            return 0.5
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        return float(thresholds[int(np.argmax(f1))])
    raise ValueError(f"Unknown threshold method: {method}")


def classification_metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> dict:
    """Precision / recall / accuracy / confusion matrix at a fixed cutoff (pathogenic=1)."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float((pred == y_true).mean()),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_predictions(
    name: str,
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    *,
    threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Compute ROC/PR/accuracy for a split; optionally print a classification report."""
    y_true = pd.Series(y_true).astype(int)
    pred = (proba >= threshold).astype(int)
    metrics = {
        "split": name,
        "n": int(len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float((pred == y_true).mean()),
    }
    if verbose:
        print(f"\n=== {name} ===")
        print(
            f"ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
            f"accuracy@{threshold}={metrics['accuracy']:.3f}"
        )
        print(
            classification_report(
                y_true, pred, target_names=["benign", "pathogenic"], digits=3
            )
        )
    return metrics


def train_holdout_model(
    frames: dict[str, pd.DataFrame] | None = None,
    *,
    params: dict | None = None,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run holdout split (unless ``frames`` given), fit XGBoost, return metrics and model."""
    if frames is None:
        frames = run_split(seed=seed)
    train_df, valid_df, test_df = frames["train"], frames["valid"], frames["test"]

    model, proba_test, valid_auc, test_auc = fit_predict(
        train_df, valid_df, test_df, params=params, seed=seed
    )

    X_train, y_train = build_feature_matrix(train_df)
    X_valid, y_valid = build_feature_matrix(valid_df)
    X_test, y_test = build_feature_matrix(test_df)
    X_train, X_valid, X_test = align_categories_to_train(X_train, X_valid, X_test)

    proba_train = model.predict_proba(X_train)[:, 1]
    proba_valid = model.predict_proba(X_valid)[:, 1]

    metrics_table = pd.DataFrame(
        [
            evaluate_predictions("train", y_train, proba_train),
            evaluate_predictions("valid", y_valid, proba_valid),
            evaluate_predictions("test", y_test, proba_test),
        ]
    ).set_index("split")

    importance = pd.Series(
        model.feature_importances_, index=FEATURE_COLUMNS, name="gain"
    ).sort_values(ascending=False)

    return {
        "frames": frames,
        "model": model,
        "matrices": {
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "X_test": X_test,
            "y_test": y_test,
        },
        "proba": {
            "train": proba_train,
            "valid": proba_valid,
            "test": proba_test,
        },
        "metrics": metrics_table,
        "importance": importance,
        "valid_auc": valid_auc,
        "test_auc": test_auc,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    frames = run_split(seed=args.seed)
    combined = pd.concat(frames.values(), ignore_index=True)
    print("Read:", POSITION_MATCHED_IN)
    print("Wrote:", TRAIN_OUT)
    print("Wrote:", VALID_OUT)
    print("Wrote:", TEST_OUT)
    print(split_summary(combined).to_string(index=False))
    train_genes = set(frames["train"][GROUP_COLUMN])
    valid_genes = set(frames["valid"][GROUP_COLUMN])
    test_genes = set(frames["test"][GROUP_COLUMN])
    assert train_genes.isdisjoint(valid_genes)
    assert train_genes.isdisjoint(test_genes)
    assert valid_genes.isdisjoint(test_genes)
    print("Holdout gene overlap across splits: none")


if __name__ == "__main__":
    main()
