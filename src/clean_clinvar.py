"""Clean the raw ClinVar parquet into a labelled, classifier-ready table.

Reads data/raw/clinvar_reliable_grch38.parquet (from ingest_clinvar.py) and writes
data/processed/clinvar_clean.parquet. Keeps only expert-reviewed single-nucleotide
variants with a definite (non-VUS) clinical significance, and adds a binary
`label` (pathogenic / benign) used as the classification target.
"""

from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / "data/raw"
data_processed = project_root / "data/processed"

CLINVAR_RAW = data_raw / "clinvar_reliable_grch38.parquet"
CLINVAR_CLEAN = data_processed / "clinvar_clean.parquet"

BENIGN = {"Likely benign", "Benign", "Benign/Likely benign"}
PATHOGENIC = {"Pathogenic", "Pathogenic/Likely pathogenic", "Likely pathogenic"}
ALLOWED_CLINICAL_SIGNIFICANCE = BENIGN | PATHOGENIC


def _label(clinical_significance: str) -> str:
    if clinical_significance in BENIGN:
        return "benign"
    if clinical_significance in PATHOGENIC:
        return "pathogenic"
    return "unknown"


def main() -> None:
    df = pd.read_parquet(CLINVAR_RAW)

    df["PhenotypeList"] = df["PhenotypeList"].apply(
        lambda x: "not provided"
        if pd.isna(x)
        else str(x)
        .replace("not provided|", "")
        .replace("not specified|", "")
        .replace("|not provided", "")
        .replace("|not specified", "")
        .replace("not specified", "not provided")
        .replace("|", ". ")
    )

    df = df[df["ReviewStatus"].isin(["practice guideline", "reviewed by expert panel"])].copy()
    df = df[df["ClinicalSignificance"].isin(ALLOWED_CLINICAL_SIGNIFICANCE)].copy()
    df = df[df["Type"] == "single nucleotide variant"].copy()

    df["label"] = df["ClinicalSignificance"].astype(str).map(_label)

    data_processed.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLINVAR_CLEAN, index=False)
    print("Saved cleaned parquet:", CLINVAR_CLEAN, df.shape)
    print("label breakdown:", df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
