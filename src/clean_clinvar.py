"""Clean raw ClinVar into labelled (P/B) and VUS tables.

Reads data/raw/clinvar_reliable_grch38.parquet (from ingest_clinvar.py) and writes:

  data/processed/clinvar_clean.parquet
      Expert-reviewed SNVs with definite clinical significance + binary
      ``label`` ∈ {pathogenic, benign} (training / evaluation target).

  data/processed/clinvar_clean_vus.parquet
      Same review/type filters, but ClinicalSignificance = Uncertain significance;
      ``label`` = vus (held out of training; for inference / triage demos).
"""

from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / "data/raw"
data_processed = project_root / "data/processed"

CLINVAR_RAW = data_raw / "clinvar_reliable_grch38.parquet"
CLINVAR_CLEAN = data_processed / "clinvar_clean.parquet"
CLINVAR_CLEAN_VUS = data_processed / "clinvar_clean_vus.parquet"

# Collapse ClinVar's graded terms into a binary training target.
# "Likely *" stays in the same bucket as the definite call (standard for this task).
BENIGN = {"Likely benign", "Benign", "Benign/Likely benign"}
PATHOGENIC = {"Pathogenic", "Pathogenic/Likely pathogenic", "Likely pathogenic"}
ALLOWED_CLINICAL_SIGNIFICANCE = BENIGN | PATHOGENIC
VUS_CLINICAL_SIGNIFICANCE = {"Uncertain significance"}  # scored later, never used as y

EXPERT_REVIEW = {"practice guideline", "reviewed by expert panel"}


def _normalize_phenotype_list(value) -> str:
    # ClinVar pipes several disease names together; drop placeholder tokens.
    if pd.isna(value):
        return "not provided"
    return (
        str(value)
        .replace("not provided|", "")
        .replace("not specified|", "")
        .replace("|not provided", "")
        .replace("|not specified", "")
        .replace("not specified", "not provided")
        .replace("|", ". ")
    )


def _label(clinical_significance: str) -> str:
    if clinical_significance in BENIGN:
        return "benign"
    if clinical_significance in PATHOGENIC:
        return "pathogenic"
    if clinical_significance in VUS_CLINICAL_SIGNIFICANCE:
        return "vus"
    return "unknown"


def _base_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Shared QC for labelled + VUS: expert SNVs only (indels/CNVs dropped)."""
    out = df.copy()
    out["PhenotypeList"] = out["PhenotypeList"].map(_normalize_phenotype_list)
    out = out[out["ReviewStatus"].isin(EXPERT_REVIEW)].copy()
    out = out[out["Type"] == "single nucleotide variant"].copy()
    return out


def clean_labelled(df: pd.DataFrame) -> pd.DataFrame:
    """Pathogenic / benign training table."""
    out = _base_clean(df)
    out = out[out["ClinicalSignificance"].isin(ALLOWED_CLINICAL_SIGNIFICANCE)].copy()
    out["label"] = out["ClinicalSignificance"].astype(str).map(_label)
    return out.reset_index(drop=True)


def clean_vus(df: pd.DataFrame) -> pd.DataFrame:
    """Uncertain-significance table (same QC filters as labelled set)."""
    out = _base_clean(df)
    out = out[out["ClinicalSignificance"].isin(VUS_CLINICAL_SIGNIFICANCE)].copy()
    out["label"] = "vus"
    return out.reset_index(drop=True)


def main() -> None:
    raw = pd.read_parquet(CLINVAR_RAW)
    labelled = clean_labelled(raw)
    vus = clean_vus(raw)

    data_processed.mkdir(parents=True, exist_ok=True)
    labelled.to_parquet(CLINVAR_CLEAN, index=False)
    vus.to_parquet(CLINVAR_CLEAN_VUS, index=False)

    print("Saved cleaned parquet:", CLINVAR_CLEAN, labelled.shape)
    print("label breakdown:", labelled["label"].value_counts().to_dict())
    print("Saved VUS parquet:", CLINVAR_CLEAN_VUS, vus.shape)
    print("VUS ClinicalSignificance:", vus["ClinicalSignificance"].value_counts().to_dict())


if __name__ == "__main__":
    main()
