"""Gene-grouped nested CV for non-XGBoost baselines.

Same outer folds / inner Youden protocol as ``cv_train_eval.run_nested_cv``.
Elastic-net logistic and random forest one-hot encode categoricals (fit on the
training frame only). CatBoost uses native categoricals, like XGBoost.

XGBoost is unchanged: pandas ``category`` + ``enable_categorical=True``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler

from cv_train_eval import DEFAULT_N_FOLDS, DEFAULT_N_TUNING_TRIALS, run_cv_folds
from features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_SEED,
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    TARGET_COLUMN,
    align_categories_to_train,
    build_feature_matrix,
    classification_metrics_at_threshold,
    encode_for_sklearn,
    pick_threshold,
    project_root,
)

# CatBoost otherwise dumps TensorBoard/error logs into cwd/catboost_info.
CATBOOST_TRAIN_DIR = project_root / "models" / "catboost_info"


def catboost_train_dir() -> str:
    """Directory for CatBoost training logs (not the saved .cbm)."""
    CATBOOST_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    return str(CATBOOST_TRAIN_DIR)

LOGISTIC_PARAM_DISTRIBUTIONS = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.0, 0.5, 1.0],  # 0 = ridge, 1 = lasso
}

RF_PARAM_DISTRIBUTIONS = {
    "n_estimators": [200, 400, 800],
    "max_depth": [4, 6, 10, None],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", 0.5, 0.8],
}

CATBOOST_PARAM_DISTRIBUTIONS = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "l2_leaf_reg": [1.0, 3.0, 5.0],
    "subsample": [0.7, 0.8, 0.9],
}

PARAM_DISTRIBUTIONS = {
    "logistic": LOGISTIC_PARAM_DISTRIBUTIONS,
    "random_forest": RF_PARAM_DISTRIBUTIONS,
    "catboost": CATBOOST_PARAM_DISTRIBUTIONS,
}

FitFn = Callable[..., dict]


def _concat_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    nonempty = [f for f in frames if not f.empty]
    if not nonempty:
        raise ValueError("Need at least one non-empty frame to fit.")
    return pd.concat(nonempty, ignore_index=True)


def fit_logistic(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    *,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Elastic-net logistic; one-hot cats. Fits on train+valid (no early stopping)."""
    # Concatenate train+valid: no native early stopping, so do not waste a gene fold.
    fit_df = _concat_frames(train_df, valid_df)
    X_fit, (X_test,), y_fit, (y_test,), names = encode_for_sklearn(fit_df, test_df)
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
        C=float(params["C"]),
        l1_ratio=float(params["l1_ratio"]),
    )
    model.fit(X_fit, y_fit)
    proba_test = model.predict_proba(X_test)[:, 1]
    coef = np.abs(model.coef_.ravel())
    return {
        "proba_test": proba_test,
        "test_auc": float(roc_auc_score(y_test, proba_test)),
        "importance": pd.Series(coef, index=names),
        "n_estimators": None,
        "model": model,
    }


def fit_random_forest(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    *,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Random forest; one-hot cats. Fits on train+valid (no early stopping)."""
    fit_df = _concat_frames(train_df, valid_df)
    X_fit, (X_test,), y_fit, (y_test,), names = encode_for_sklearn(fit_df, test_df)
    model = RandomForestClassifier(
        class_weight="balanced_subsample",
        n_jobs=4,
        random_state=seed,
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
    )
    model.fit(X_fit, y_fit)
    proba_test = model.predict_proba(X_test)[:, 1]
    return {
        "proba_test": proba_test,
        "test_auc": float(roc_auc_score(y_test, proba_test)),
        "importance": pd.Series(model.feature_importances_, index=names),
        "n_estimators": int(params["n_estimators"]),
        "model": model,
    }


def _catboost_matrices(
    train_df: pd.DataFrame,
    *others: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[pd.DataFrame, ...], pd.Series, tuple[pd.Series, ...], list[int]]:
    frames = [build_feature_matrix(train_df)[0]]
    ys = [(train_df[TARGET_COLUMN] == "pathogenic").astype(int)]
    for frame in others:
        X, y = build_feature_matrix(frame)
        frames.append(X)
        ys.append(y)
    aligned = align_categories_to_train(*frames)
    out_x = []
    for X in aligned:
        X = X.copy()
        for col in BOOLEAN_FEATURES:
            X[col] = X[col].astype(int)
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype("string")
        out_x.append(X)
    cat_idx = [out_x[0].columns.get_loc(col) for col in CATEGORICAL_FEATURES]
    return out_x[0], tuple(out_x[1:]), ys[0], tuple(ys[1:]), cat_idx


def prepare_catboost_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    """Single-frame CatBoost matrix: bools as int, cats as strings, plus cat indices."""
    X, y = build_feature_matrix(df)
    X = X.copy()
    for col in BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string").fillna("__MISSING__")
    cat_idx = [int(X.columns.get_loc(col)) for col in CATEGORICAL_FEATURES]
    return X, y, cat_idx


def fit_catboost(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    *,
    seed: int = DEFAULT_SEED,
    n_estimators: int | None = None,
    early_stopping_rounds: int | None = 50,
) -> dict:
    """CatBoost with native categoricals. Early-stops on valid when a valid fold exists."""
    from catboost import CatBoostClassifier

    params = dict(params)
    iterations = int(n_estimators) if n_estimators is not None else 500
    use_es = (
        early_stopping_rounds is not None
        and not valid_df.empty
        and n_estimators is None
    )
    if use_es:
        X_train, (X_valid, X_test), y_train, (y_valid, y_test), cat_idx = _catboost_matrices(
            train_df, valid_df, test_df
        )
        eval_set = (X_valid, y_valid)
    else:
        fit_df = _concat_frames(train_df, valid_df)
        X_train, (X_test,), y_train, (y_test,), cat_idx = _catboost_matrices(fit_df, test_df)
        eval_set = None

    model_kwargs: dict = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=seed,
        thread_count=4,
        verbose=False,
        iterations=iterations,
        use_best_model=bool(use_es),
        bootstrap_type="Bernoulli",
        depth=int(params["depth"]),
        learning_rate=float(params["learning_rate"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        subsample=float(params["subsample"]),
        train_dir=catboost_train_dir(),
        allow_writing_files=True,
    )
    if use_es:
        model_kwargs["od_type"] = "Iter"
        model_kwargs["od_wait"] = int(early_stopping_rounds)
    model = CatBoostClassifier(**model_kwargs)
    fit_kwargs: dict = {"cat_features": cat_idx}
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set
    model.fit(X_train, y_train, **fit_kwargs)
    proba_test = model.predict_proba(X_test)[:, 1]
    best_iter = model.get_best_iteration()
    if use_es and best_iter is not None and int(best_iter) >= 0:
        trees = int(best_iter) + 1
    else:
        trees = int(getattr(model, "tree_count_", iterations))
    importance = pd.Series(model.get_feature_importance(), index=FEATURE_COLUMNS)
    return {
        "proba_test": proba_test,
        "test_auc": float(roc_auc_score(y_test, proba_test)),
        "importance": importance,
        "n_estimators": trees,
        "model": model,
    }


def fit_catboost_fixed(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    *,
    seed: int = DEFAULT_SEED,
    n_estimators: int,
) -> dict:
    return fit_catboost(
        train_df,
        valid_df,
        test_df,
        params,
        seed=seed,
        n_estimators=max(1, int(n_estimators)),
        early_stopping_rounds=None,
    )


ESTIMATORS: dict[str, dict] = {
    "logistic": {
        "fit": fit_logistic,
        "fit_outer": fit_logistic,
        "uses_early_stopping": False,
        "param_distributions": LOGISTIC_PARAM_DISTRIBUTIONS,
    },
    "random_forest": {
        "fit": fit_random_forest,
        "fit_outer": fit_random_forest,
        "uses_early_stopping": False,
        "param_distributions": RF_PARAM_DISTRIBUTIONS,
    },
    "catboost": {
        "fit": fit_catboost,
        "fit_outer": fit_catboost_fixed,
        "uses_early_stopping": True,
        "param_distributions": CATBOOST_PARAM_DISTRIBUTIONS,
    },
}


def nested_tune_baseline(
    pool: pd.DataFrame,
    fit_fn: FitFn,
    param_distributions: dict,
    *,
    n_trials: int = DEFAULT_N_TUNING_TRIALS,
    seed: int = DEFAULT_SEED,
    threshold_method: str = "youden",
) -> tuple[dict, float, int | None, float, pd.DataFrame]:
    """Inner gene-grouped CV on ``pool`` for a non-XGBoost estimator."""
    inner_folds = sorted(int(f) for f in pool["fold"].unique())
    if len(inner_folds) < 2:
        raise ValueError("Need >= 2 folds in the outer-train pool for inner CV.")

    sampler = ParameterSampler(param_distributions, n_iter=n_trials, random_state=seed)
    trial_rows = []
    best_params: dict | None = None
    best_inner_auc = -np.inf
    best_n_estimators: int | None = None
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

            result = fit_fn(train_df, valid_df, test_df, params, seed=seed)
            fold_aucs.append(result["test_auc"])
            if result.get("n_estimators") is not None:
                fold_n_trees.append(int(result["n_estimators"]))
            y_inner = (test_df[TARGET_COLUMN] == "pathogenic").astype(int).to_numpy()
            fold_thresholds.append(
                pick_threshold(y_inner, result["proba_test"], method=threshold_method)
            )

        mean_auc = float(np.mean(fold_aucs))
        median_trees = int(np.median(fold_n_trees)) if fold_n_trees else None
        threshold = float(np.median(fold_thresholds))
        row = {
            "trial": trial_idx,
            "inner_mean_roc_auc": mean_auc,
            "inner_std_roc_auc": float(np.std(fold_aucs)),
            "inner_median_n_estimators": median_trees,
            "inner_threshold": threshold,
            **params,
        }
        trial_rows.append(row)
        if mean_auc > best_inner_auc:
            best_inner_auc = mean_auc
            best_params = params
            best_n_estimators = median_trees
            best_threshold = threshold

    assert best_params is not None
    return best_params, best_inner_auc, best_n_estimators, best_threshold, pd.DataFrame(trial_rows)


def run_nested_cv_baseline(
    estimator: str,
    folded: pd.DataFrame | None = None,
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    n_trials: int = DEFAULT_N_TUNING_TRIALS,
    seed: int = DEFAULT_SEED,
    param_distributions: dict | None = None,
    threshold_method: str = "youden",
    verbose: bool = True,
) -> dict:
    """Nested gene-grouped CV for ``logistic``, ``random_forest``, or ``catboost``."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator {estimator!r}; expected one of {list(ESTIMATORS)}")
    spec = ESTIMATORS[estimator]
    distributions = param_distributions or spec["param_distributions"]
    fit_inner: FitFn = spec["fit"]
    fit_outer: FitFn = spec["fit_outer"]

    if folded is None:
        folded = run_cv_folds(n_folds=n_folds, seed=seed)

    fold_rows = []
    oof_parts = []
    importances = []
    tuning_logs = []

    for outer_fold in range(n_folds):
        outer_test_df = folded[folded["fold"] == outer_fold].reset_index(drop=True)
        pool = folded[folded["fold"] != outer_fold].reset_index(drop=True)

        best_params, best_inner_auc, n_estimators, threshold, trials = nested_tune_baseline(
            pool,
            fit_inner,
            distributions,
            n_trials=n_trials,
            seed=seed + outer_fold,
            threshold_method=threshold_method,
        )
        trials = trials.copy()
        trials.insert(0, "outer_fold", outer_fold)
        tuning_logs.append(trials)

        outer_kwargs: dict = dict(params=best_params, seed=seed)
        if spec["uses_early_stopping"]:
            outer_kwargs["n_estimators"] = n_estimators if n_estimators is not None else 1
        result = fit_outer(
            pool,
            pool.iloc[0:0].copy(),
            outer_test_df,
            **outer_kwargs,
        )
        y_test = (outer_test_df[TARGET_COLUMN] == "pathogenic").astype(int)
        cls = classification_metrics_at_threshold(y_test, result["proba_test"], threshold)
        used_trees = result.get("n_estimators")
        if spec["uses_early_stopping"] and n_estimators is not None:
            used_trees = n_estimators

        row = {
            "fold": outer_fold,
            "n_train": len(pool),
            "n_test": len(outer_test_df),
            "n_test_genes": outer_test_df[GROUP_COLUMN].nunique(),
            "best_inner_roc_auc": best_inner_auc,
            "n_estimators": used_trees,
            "threshold": threshold,
            "roc_auc": result["test_auc"],
            "pr_auc": float(average_precision_score(y_test, result["proba_test"])),
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
            trees_msg = f"  n_estimators={used_trees}" if used_trees is not None else ""
            print(
                f"outer {outer_fold}: thr={threshold:.3f}  "
                f"ROC={row['roc_auc']:.3f}  P={row['precision']:.3f}  "
                f"R={row['recall']:.3f}  Acc={row['accuracy']:.3f}"
                f"{trees_msg}"
            )

        oof_parts.append(
            pd.DataFrame(
                {
                    "VariationID": outer_test_df["VariationID"].values,
                    "gene": outer_test_df[GROUP_COLUMN].values,
                    "fold": outer_fold,
                    "y_true": y_test.values,
                    "y_proba": result["proba_test"],
                    "threshold": threshold,
                    "y_pred": (result["proba_test"] >= threshold).astype(int),
                }
            )
        )
        imp = result["importance"].rename(f"fold_{outer_fold}")
        importances.append(imp)

    fold_metrics = pd.DataFrame(fold_rows).set_index("fold")
    oof = pd.concat(oof_parts, ignore_index=True)
    tuning_log = pd.concat(tuning_logs, ignore_index=True)
    importance_df = pd.concat(importances, axis=1).fillna(0.0)
    importance_df["mean"] = importance_df.mean(axis=1)
    importance_df["std"] = importance_df.std(axis=1)
    importance_df = importance_df.sort_values("mean", ascending=False)

    confusion_total = fold_metrics[["tn", "fp", "fn", "tp"]].sum().astype(int)
    cls_summary = fold_metrics[
        ["threshold", "accuracy", "precision", "recall", "f1"]
    ].agg(["mean", "std", "min", "max"])

    return {
        "estimator": estimator,
        "folded": folded,
        "fold_metrics": fold_metrics,
        "oof": oof,
        "tuning_log": tuning_log,
        "importance": importance_df,
        "confusion_total": confusion_total,
        "classification_summary": cls_summary,
        "threshold_method": threshold_method,
    }
