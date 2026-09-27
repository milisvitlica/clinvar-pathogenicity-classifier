"""Train a deployable final CatBoost classifier and run inference.

Two-stage protocol:
  1. Nested gene CV (`cv_baselines.run_nested_cv_baseline("catboost")` /
     ``notebooks/modeling/cv/cv_catboost.ipynb``) → honest outer-fold KPIs.
  2. ``train_final_catboost`` runs a **single-loop** gene CV on the *full*
     labelled table (`nested_tune_baseline`) to pick one ``θ*``, ``n_estimators*``,
     and Youden threshold, then fits on **all** labelled rows (no holdout).

The tune's inner mean ROC-AUC is a selection diagnostic only — do not report it
as model performance. Nested-CV comparison chose CatBoost for deploy
(``notebooks/modeling/cv/conclusions.md``).

Artifacts (under ``models/`` by default):
  catboost_final_pathogenicity.cbm        — CatBoost model
  catboost_final_pathogenicity_meta.json  — threshold, params, feature schema
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cv_baselines import (
    CATBOOST_PARAM_DISTRIBUTIONS,
    fit_catboost,
    nested_tune_baseline,
    prepare_catboost_matrix,
)
from cv_train_eval import DEFAULT_N_FOLDS, DEFAULT_N_TUNING_TRIALS, assign_cv_folds
from features import (
    CATEGORICAL_FEATURES,
    DEFAULT_SEED,
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    ID_COLUMN,
    TARGET_COLUMN,
    prepare_modeling_frame,
    project_root,
    resolve_position_matched_path,
)

MODELS_DIR = project_root / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "catboost_final_pathogenicity.cbm"
DEFAULT_META_PATH = MODELS_DIR / "catboost_final_pathogenicity_meta.json"


def _ensure_relative_position(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ``relative_protein_position`` when Length + protein_position are present."""
    out = df.copy()
    if "relative_protein_position" not in out.columns or out["relative_protein_position"].isna().all():
        if "Length" in out.columns and "protein_position" in out.columns:
            length = pd.to_numeric(out["Length"], errors="coerce")
            pos = pd.to_numeric(out["protein_position"], errors="coerce")
            out["relative_protein_position"] = (pos / length).where(length > 0)
    return out


def _jsonable(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _matrix_with_saved_categories(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Build the CatBoost frame and force category levels from training meta."""
    X, _, _ = prepare_catboost_matrix(_ensure_relative_position(df))
    for col, cats in meta.get("categorical_levels", {}).items():
        if col not in X.columns:
            continue
        values = X[col].astype("string").fillna("__MISSING__")
        values = values.where(values.isin(cats), "__MISSING__")
        X[col] = values
    return X[list(meta["feature_columns"])]  # drop columns added after this model was saved


def train_final_catboost(
    *,
    input_path: Path | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    n_trials: int = DEFAULT_N_TUNING_TRIALS,
    seed: int = DEFAULT_SEED,
    threshold_method: str = "youden",
    param_distributions: dict | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    meta_path: Path = DEFAULT_META_PATH,
    verbose: bool = True,
) -> dict:
    """Tune on full labelled data (gene CV), fit on all rows, persist model + meta."""
    from catboost import CatBoostClassifier

    raw = pd.read_parquet(input_path or resolve_position_matched_path())
    modeling = prepare_modeling_frame(raw)
    # Single-loop gene CV on ALL labelled rows (not nested): pick one θ* for deploy.
    folded = assign_cv_folds(modeling, n_folds=n_folds, seed=seed)

    best_params, best_inner_auc, n_estimators, threshold, trials = nested_tune_baseline(
        folded,
        fit_catboost,
        param_distributions or CATBOOST_PARAM_DISTRIBUTIONS,
        n_trials=n_trials,
        seed=seed,
        threshold_method=threshold_method,
    )
    n_estimators = max(1, int(n_estimators or 1))

    X, y, cat_idx = prepare_catboost_matrix(folded)
    params = dict(best_params)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=seed,
        thread_count=4,
        verbose=False,
        iterations=n_estimators,
        use_best_model=False,
        bootstrap_type="Bernoulli",
        depth=int(params["depth"]),
        learning_rate=float(params["learning_rate"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        subsample=float(params["subsample"]),
    )
    model.fit(X, y, cat_features=cat_idx)

    categorical_levels = {
        col: sorted({str(v) for v in X[col].dropna().unique()})
        for col in CATEGORICAL_FEATURES
    }
    for levels in categorical_levels.values():
        if "__MISSING__" not in levels:
            levels.append("__MISSING__")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    meta = {
        "estimator": "catboost",
        "model_path": str(model_path),
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categorical_levels": categorical_levels,
        "params": {k: _jsonable(v) for k, v in params.items()},
        "n_estimators": int(n_estimators),
        "threshold": float(threshold),
        "threshold_method": threshold_method,
        "seed": int(seed),
        "n_folds_tune": int(n_folds),
        "n_trials": int(n_trials),
        "n_train": int(len(folded)),
        "n_genes": int(folded[GROUP_COLUMN].nunique()),
        "tune_inner_mean_roc_auc": float(best_inner_auc),
        "label_positive": "pathogenic",
        "label_negative": "benign",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    if verbose:
        print(f"Tuned on {meta['n_train']} variants / {meta['n_genes']} genes")
        print(f"Inner mean ROC-AUC (full-data tune CV): {best_inner_auc:.3f}")
        print(f"n_estimators={n_estimators}  threshold={threshold:.4f} ({threshold_method})")
        print(f"params={params}")
        print(f"Saved model: {model_path}")
        print(f"Saved meta:  {meta_path}")

    return {
        "model": model,
        "meta": meta,
        "trials": trials,
        "train_frame": folded,
        "model_path": model_path,
        "meta_path": meta_path,
    }


def load_final_catboost(
    model_path: Path = DEFAULT_MODEL_PATH,
    meta_path: Path = DEFAULT_META_PATH,
):
    """Load persisted CatBoost model and metadata."""
    from catboost import CatBoostClassifier

    meta = json.loads(Path(meta_path).read_text())
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model, meta


def predict_pathogenicity(
    df: pd.DataFrame,
    *,
    model=None,
    meta: dict | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    meta_path: Path = DEFAULT_META_PATH,
) -> pd.DataFrame:
    """Score rows with the final CatBoost classifier; probabilities and hard labels."""
    if model is None or meta is None:
        model, meta = load_final_catboost(model_path, meta_path)

    X = _matrix_with_saved_categories(df, meta)
    proba = model.predict_proba(X)[:, 1]
    threshold = float(meta["threshold"])
    pred = (proba >= threshold).astype(int)

    out = pd.DataFrame(index=df.index)
    if ID_COLUMN in df.columns:
        out[ID_COLUMN] = df[ID_COLUMN].values
    if GROUP_COLUMN in df.columns:
        out[GROUP_COLUMN] = df[GROUP_COLUMN].values
    if "Name" in df.columns:
        out["Name"] = df["Name"].values
    if "ClinicalSignificance" in df.columns:
        out["ClinicalSignificance"] = df["ClinicalSignificance"].values
    if TARGET_COLUMN in df.columns:
        out[TARGET_COLUMN] = df[TARGET_COLUMN].values
    out["y_proba"] = proba
    out["threshold"] = threshold
    out["y_pred"] = pred
    out["pred_label"] = np.where(pred == 1, "pathogenic", "benign")
    return out.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TUNING_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threshold-method", choices=["f1", "youden"], default="youden")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--meta-path", type=Path, default=DEFAULT_META_PATH)
    args = parser.parse_args(argv)

    train_final_catboost(
        input_path=args.input,
        n_folds=args.n_folds,
        n_trials=args.n_trials,
        seed=args.seed,
        threshold_method=args.threshold_method,
        model_path=args.model_path,
        meta_path=args.meta_path,
    )


if __name__ == "__main__":
    main()
