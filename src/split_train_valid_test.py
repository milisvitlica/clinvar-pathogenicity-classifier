"""Build a modeling table and gene-grouped holdout / CV splits.

Reads data/processed/clinvar_uniprot_position_matched.parquet, keeps ClinVar–UniProt
matched variants (one row per VariationID), and assigns each *gene* wholly to a
holdout split or CV fold so the same gene never leaks across partitions.

Holdout writes:
  data/processed/train.parquet
  data/processed/valid.parquet
  data/processed/test.parquet

CV writes:
  data/processed/cv_folds.parquet   (column ``fold`` ∈ {0 .. n_folds-1})
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data/processed"

POSITION_MATCHED_IN = data_processed / "clinvar_uniprot_position_matched.parquet"
TRAIN_OUT = data_processed / "train.parquet"
VALID_OUT = data_processed / "valid.parquet"
TEST_OUT = data_processed / "test.parquet"
CV_FOLDS_OUT = data_processed / "cv_folds.parquet"

DEFAULT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_SEED = 42
DEFAULT_N_FOLDS = 5

# Structured features available from the position-matched join (no IDs, no label leaks).
NUMERIC_FEATURES = [
    "Length",
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
    "closest_feature_description",
    "domain_names",
    "region_names",
    "Chromosome",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "OriginSimple",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "label"
GROUP_COLUMN = "gene"
ID_COLUMN = "VariationID"

# Kept on written parquets for inspection / debugging (not used as model inputs).
META_COLUMNS = [
    ID_COLUMN,
    GROUP_COLUMN,
    "Entry",
    "Name",
    "GeneSymbol",
    "ClinicalSignificance",
    "match_type",
]


def prepare_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to matched labelled variants and deduplicate to one row per VariationID."""
    frame = df[df["match_type"] == "both"].copy()
    frame = frame[frame[TARGET_COLUMN].isin(["pathogenic", "benign"])].copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype("Int64")

    # Prefer rows with a protein position; break remaining ties on Entry for stability.
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


def _gene_stats(df: pd.DataFrame, gene_col: str, label_col: str, seed: int) -> pd.DataFrame:
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
    gene_stats = _gene_stats(out, gene_col, label_col, seed)

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

    for row in gene_stats.itertuples(index=False):
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
    gene_stats = _gene_stats(out, gene_col, label_col, seed)

    n_total = len(out)
    n_path_total = int((out[label_col] == "pathogenic").sum())
    target = n_total / n_folds
    path_target = n_path_total / n_folds
    counts = np.zeros(n_folds, dtype=float)
    path_counts = np.zeros(n_folds, dtype=float)
    assignment: dict[str, int] = {}

    for row in gene_stats.itertuples(index=False):
        # Pack into the fold furthest below its size target; break ties on label deficit.
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

    For fold ``k``, the test set is fold ``k``. Validation is a *deterministic*
    neighbouring fold ``(k + valid_fold_offset) % n_folds`` (not a random row
    subsample) — used for hyperparameter tuning / early stopping. Remaining
    folds form the train set. Set ``valid_fold_offset=0`` to skip a dedicated
    valid fold (train = all non-test).
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
    return list(dict.fromkeys(META_COLUMNS + list(extra) + FEATURE_COLUMNS + [TARGET_COLUMN]))


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


def run_cv_folds(
    *,
    input_path: Path = POSITION_MATCHED_IN,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Prepare, assign gene-grouped CV folds, write parquet, and return the frame."""
    raw = pd.read_parquet(input_path)
    modeling = prepare_modeling_frame(raw)
    folded = assign_cv_folds(modeling, n_folds=n_folds, seed=seed)
    folded = folded[_keep_columns("fold")].copy()

    data_processed.mkdir(parents=True, exist_ok=True)
    folded.to_parquet(CV_FOLDS_OUT, index=False)
    return folded


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["holdout", "cv", "both"],
        default="both",
        help="Which gene-grouped partition(s) to write (default: both).",
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    print("Read:", POSITION_MATCHED_IN)

    if args.mode in {"holdout", "both"}:
        frames = run_split(seed=args.seed)
        combined = pd.concat(frames.values(), ignore_index=True)
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

    if args.mode in {"cv", "both"}:
        folded = run_cv_folds(n_folds=args.n_folds, seed=args.seed)
        print("Wrote:", CV_FOLDS_OUT)
        print(split_summary(folded, partition_col="fold").to_string(index=False))
        # Genes must appear in exactly one fold.
        gene_folds = folded.groupby(GROUP_COLUMN)["fold"].nunique()
        assert (gene_folds == 1).all()
        print(f"CV gene overlap across {args.n_folds} folds: none")


if __name__ == "__main__":
    main()
