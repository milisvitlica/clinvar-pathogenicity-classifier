"""Build a modeling table and gene-grouped train/valid/test split.

Reads data/processed/clinvar_uniprot_position_matched.parquet, keeps ClinVar–UniProt
matched variants (one row per VariationID), and assigns each *gene* wholly to
train, valid, or test so the same gene never leaks across splits.

Writes:
  data/processed/train.parquet
  data/processed/valid.parquet
  data/processed/test.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data/processed"

POSITION_MATCHED_IN = data_processed / "clinvar_uniprot_position_matched.parquet"
TRAIN_OUT = data_processed / "train.parquet"
VALID_OUT = data_processed / "valid.parquet"
TEST_OUT = data_processed / "test.parquet"

DEFAULT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_SEED = 42

# Structured features available from the position-matched join (no IDs, no label leaks).
NUMERIC_FEATURES = [
    "Length",
    "protein_position",
    "relative_protein_position",
    "distance_to_closest_feature"
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

# Kept on the split parquets for inspection / debugging (not used as model inputs).
META_COLUMNS = [
    ID_COLUMN,
    GROUP_COLUMN,
    "Entry",
    "Name",
    "GeneSymbol",
    "ClinicalSignificance",
    "match_type",
    "split",
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
    rng = np.random.default_rng(seed)

    gene_stats = (
        out.groupby(gene_col, sort=False)
        .agg(
            n=(label_col, "size"),
            n_pathogenic=(label_col, lambda s: int((s == "pathogenic").sum())),
        )
        .reset_index()
    )
    # Largest genes first (better packing), with a seeded shuffle for equal sizes.
    gene_stats["_shuffle"] = rng.random(len(gene_stats))
    gene_stats = gene_stats.sort_values(["n", "_shuffle"], ascending=[False, True])

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
        # Prefer the split furthest below its size target; break ties with label deficit.
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


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-split counts useful for logging and notebooks."""
    rows = []
    for split, part in df.groupby("split", sort=False):
        rows.append(
            {
                "split": split,
                "n_variants": len(part),
                "n_genes": part[GROUP_COLUMN].nunique(),
                "pct_pathogenic": 100 * (part[TARGET_COLUMN] == "pathogenic").mean(),
                "pct_with_protein_position": 100 * part["has_protein_position"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    order = {"train": 0, "valid": 1, "test": 2}
    return summary.sort_values("split", key=lambda s: s.map(order)).reset_index(drop=True)


def run_split(
    *,
    input_path: Path = POSITION_MATCHED_IN,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """Prepare, split, write parquets, and return the three frames."""
    raw = pd.read_parquet(input_path)
    modeling = prepare_modeling_frame(raw)
    split_df = split_by_gene(modeling, ratios=ratios, seed=seed)

    keep_cols = list(
        dict.fromkeys(
            META_COLUMNS
            + FEATURE_COLUMNS
            + [TARGET_COLUMN]
        )
    )
    # relative_protein_position is in FEATURE_COLUMNS; ensure engineered cols exist.
    split_df = split_df[keep_cols].copy()

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


def main() -> None:
    frames = run_split()
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
    print("Gene overlap across splits: none")


if __name__ == "__main__":
    main()
