"""Shared feature matrices, encoding, thresholds, and optional holdout split.

Reads data/processed/clinvar_uniprot_position_matched.parquet, keeps matched
labelled variants (one row per VariationID), and provides helpers to build
model matrices (native categoricals for XGBoost; one-hot for sklearn) and fit.

gnomAD frequencies are **EDA-only** (circular with ClinVar ACMG labels); they
are not in ``FEATURE_COLUMNS``. See ``add_gnomad_features`` and
``notebooks/eda/gnomad_eda.ipynb``.

PhyloP100way (UCSC hg38) **is** a model feature. Conservation is sometimes
ACMG PP3/BP4 (supporting/moderate computational evidence), not stand-alone
like BA1 — milder leakage than AF. See ``add_phylop_features``.

Optional CLI writes a gene-grouped train/valid/test split:
  data/processed/train.parquet
  data/processed/valid.parquet
  data/processed/test.parquet

Honest KPIs come from ``cv_train_eval`` / ``cv_baselines``, not this split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data/processed"

POSITION_MATCHED_IN = data_processed / "clinvar_uniprot_position_matched.parquet"
POSITION_MATCHED_VUS_IN = data_processed / "clinvar_uniprot_position_matched_vus.parquet"
PHYLOP_CLEAN = data_processed / "phylop_clean.parquet"
TRAIN_OUT = data_processed / "train.parquet"
VALID_OUT = data_processed / "valid.parquet"
TEST_OUT = data_processed / "test.parquet"

DEFAULT_RATIOS = (0.70, 0.15, 0.15)  # optional holdout only; nested CV does not use this
DEFAULT_SEED = 42

# Columns the models see. Omitted on purpose under gene-grouped CV:
#   Chromosome / Length — nearly constant within a gene (identity leak).
# gnomAD AF is EDA-only: ClinVar P/B labels already use ACMG BA1/BS1/PM2.
# phylop_100way: PP3/BP4-style conservation (supporting/moderate, not BA1).
NUMERIC_FEATURES = [
    "protein_position",
    "relative_protein_position",
    "distance_to_closest_feature",
    "phylop_100way",
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

GNOMAD_AF_EPS = 1e-6  # used by add_gnomad_features (EDA), not the model matrix

FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "label"  # pathogenic vs benign
GROUP_COLUMN = "gene"  # splits are by gene, never random rows
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


def _dedupe_matched_variants(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per VariationID, preferring rows with a parsed protein position."""
    frame = frame.copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype("Int64")
    frame["_rank_pos"] = (~frame["has_protein_position"]).astype(int)
    return (
        frame.sort_values([ID_COLUMN, "_rank_pos", "Entry"], kind="mergesort")
        .drop_duplicates(ID_COLUMN, keep="first")
        .drop(columns="_rank_pos")
        .reset_index(drop=True)
    )


def _add_relative_protein_position(frame: pd.DataFrame) -> pd.DataFrame:
    length = pd.to_numeric(frame["Length"], errors="coerce")
    pos = pd.to_numeric(frame["protein_position"], errors="coerce")
    frame = frame.copy()
    frame["relative_protein_position"] = (pos / length).where(length > 0)
    return frame


def _gnomad_af_bin(in_gnomad: bool, af: float) -> str:
    """ACMG-inspired AF bins (BA1 ≥ 5%, BS1 ≥ 1%; absence is PM2-like)."""
    if not in_gnomad:
        return "absent"
    if af >= 0.05:
        return "common_ba1"
    if af >= 0.01:
        return "common_bs1"
    if af >= 0.001:
        return "low_freq"
    if af >= 1e-4:
        return "rare"
    if af >= 1e-5:
        return "ultra_rare"
    return "singleton_or_private"


def add_gnomad_features(frame: pd.DataFrame) -> pd.DataFrame:
    """EDA helper: fill absent AF to 0, add ``log10_gnomad_af`` and ``gnomad_af_bin``.

    Not used by ``FEATURE_COLUMNS``. ClinVar P/B labels already use BA1/BS1/PM2.
    """
    frame = frame.copy()
    if "in_gnomad" not in frame.columns:
        print(
            "Warning: gnomAD columns missing; filling absent-frequency defaults. "
            "Run python src/join_clinvar_gnomad.py"
        )
        frame["in_gnomad"] = False
        frame["gnomad_af"] = pd.NA
        frame["gnomad_af_popmax"] = pd.NA
        frame["gnomad_nhomalt"] = pd.NA

    in_g = frame["in_gnomad"].fillna(False).astype(bool)
    if "gnomad_af" not in frame.columns:
        frame["gnomad_af"] = pd.NA
    if "gnomad_af_popmax" not in frame.columns:
        frame["gnomad_af_popmax"] = pd.NA
    if "gnomad_nhomalt" not in frame.columns:
        frame["gnomad_nhomalt"] = pd.NA

    af = pd.to_numeric(frame["gnomad_af"], errors="coerce")
    af_popmax = pd.to_numeric(frame["gnomad_af_popmax"], errors="coerce")
    nhom = pd.to_numeric(frame["gnomad_nhomalt"], errors="coerce")

    # Absent ≠ "typical rare": fill 0 (PM2-style) so sklearn median-impute cannot
    # replace missing AF with the training median.
    af = af.where(in_g, 0.0).fillna(0.0)
    af_popmax = af_popmax.where(in_g, 0.0).fillna(af)
    nhom = nhom.where(in_g, 0.0).fillna(0.0)

    frame["in_gnomad"] = in_g
    frame["gnomad_af"] = af
    frame["gnomad_af_popmax"] = af_popmax
    frame["gnomad_nhomalt"] = nhom
    frame["log10_gnomad_af"] = np.log10(af + GNOMAD_AF_EPS)
    if "gnomad_filter_pass" in frame.columns:
        frame["gnomad_filter_pass"] = frame["gnomad_filter_pass"].fillna(False).astype(bool)
    frame["gnomad_af_bin"] = [
        _gnomad_af_bin(flag, float(freq)) for flag, freq in zip(in_g, af)
    ]
    return frame


def _clinvar_chrom_key(chrom) -> str:
    return str(chrom).removeprefix("chr")


def add_phylop_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Left-join UCSC phyloP100way (reference-base conservation) for the model.

    Requires ``data/processed/phylop_clean.parquet`` from ``ingest_phylop.py``.
    Missing sites (e.g. MT) stay NA; trees use native NA, sklearn median-imputes.
    """
    frame = frame.copy()
    if "phylop_100way" in frame.columns and frame["phylop_100way"].notna().any():
        frame["phylop_100way"] = pd.to_numeric(frame["phylop_100way"], errors="coerce")
        return frame
    if not PHYLOP_CLEAN.exists():
        raise FileNotFoundError(
            f"Missing {PHYLOP_CLEAN}. Run python src/ingest_phylop.py"
        )
    phy = pd.read_parquet(PHYLOP_CLEAN, columns=["chrom", "pos", "phylop_100way"])
    phy = phy.drop_duplicates(["chrom", "pos"])
    phy["chrom"] = phy["chrom"].map(_clinvar_chrom_key)
    phy["pos"] = pd.to_numeric(phy["pos"], errors="coerce").astype("Int64")
    keys = pd.DataFrame({
        "chrom": frame["Chromosome"].map(_clinvar_chrom_key),
        "pos": pd.to_numeric(frame["Start"], errors="coerce").astype("Int64"),
    })
    scores = keys.merge(phy, on=["chrom", "pos"], how="left")["phylop_100way"]
    frame["phylop_100way"] = pd.to_numeric(scores.to_numpy(), errors="coerce")
    return frame


def resolve_position_matched_path(*, vus: bool = False) -> Path:
    """UniProt position-matched table used for training / CV / inference."""
    return POSITION_MATCHED_VUS_IN if vus else POSITION_MATCHED_IN


def prepare_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Matched P/B variants, one row per VariationID (no gnomAD features)."""
    frame = df[df["match_type"] == "both"].copy()  # need UniProt + ClinVar
    frame = frame[frame[TARGET_COLUMN].isin(["pathogenic", "benign"])].copy()
    frame = _dedupe_matched_variants(frame)
    frame = _add_relative_protein_position(frame)
    frame = add_phylop_features(frame)

    if frame[GROUP_COLUMN].isna().any():
        raise ValueError("Modeling rows are missing gene labels needed for the group split.")

    return frame


def prepare_inference_frame(
    df: pd.DataFrame,
    *,
    labels: set[str] | None = None,
) -> pd.DataFrame:
    """Matched UniProt–ClinVar rows ready for scoring (one row per VariationID).

    Unlike ``prepare_modeling_frame``, this keeps non-training labels (e.g. ``vus``).
    Pass ``labels=None`` to keep any label among matched rows.
    """
    frame = df[df["match_type"] == "both"].copy()
    if labels is not None:
        frame = frame[frame[TARGET_COLUMN].isin(labels)].copy()
    if frame.empty:
        raise ValueError("No matched variants available for inference.")
    frame = _dedupe_matched_variants(frame)
    frame = _add_relative_protein_position(frame)
    frame = add_phylop_features(frame)

    if frame[GROUP_COLUMN].isna().any():
        raise ValueError("Inference rows are missing gene labels.")

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

    y = (df[TARGET_COLUMN] == "pathogenic").astype(int)  # 1 = pathogenic, 0 = benign
    y.name = TARGET_COLUMN
    return X, y


def align_categories_to_train(
    train: pd.DataFrame,
    *others: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Restrict categorical levels to those seen in ``train``; map others to ``__MISSING__``.

    Prevents test-set alleles/bins from leaking into training category codes.
    """
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


def sklearn_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Feature frame for sklearn: numeric NaNs kept, bools as 0/1, cats as strings."""
    X = df[FEATURE_COLUMNS].copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in BOOLEAN_FEATURES:
        X[col] = X[col].fillna(False).astype(int)
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string").fillna("__MISSING__")
    return X


def make_sklearn_preprocessor() -> ColumnTransformer:
    """Median-impute numerics; one-hot cats fitted on train only (``handle_unknown=ignore``).

    XGBoost does **not** use this path: it keeps pandas ``category`` columns and
    ``enable_categorical=True``. Numerics are median-imputed and standardized
    (needed for elastic-net; harmless for random forest).
    """
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
            ("cat", onehot, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def encode_for_sklearn(
    train_df: pd.DataFrame,
    *others: pd.DataFrame,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray, tuple[np.ndarray, ...], list[str]]:
    """Fit one-hot / impute on ``train_df`` only; transform other frames.

    Returns ``X_train, X_others, y_train, y_others, feature_names``.
    """
    preprocessor = make_sklearn_preprocessor()
    X_train = preprocessor.fit_transform(sklearn_input_frame(train_df))
    y_train = (train_df[TARGET_COLUMN] == "pathogenic").astype(int).to_numpy()
    X_others = tuple(
        preprocessor.transform(sklearn_input_frame(frame)) for frame in others
    )
    y_others = tuple(
        (frame[TARGET_COLUMN] == "pathogenic").astype(int).to_numpy()
        for frame in others
    )
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    return X_train, X_others, y_train, y_others, feature_names


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
    input_path: Path | None = None,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """Prepare, holdout-split, write parquets, and return the three frames."""
    input_path = input_path or resolve_position_matched_path()
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


def scale_pos_weight_from_labels(y: pd.Series | np.ndarray) -> float:
    """XGBoost ``scale_pos_weight`` ≈ n_negative / n_positive (benign / pathogenic)."""
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


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
        eval_metric="aucpr",
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

    if "scale_pos_weight" not in params:
        params["scale_pos_weight"] = scale_pos_weight_from_labels(y_train)

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

    if "scale_pos_weight" not in params:
        params["scale_pos_weight"] = scale_pos_weight_from_labels(y_train)

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
    print("Read:", resolve_position_matched_path())
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
