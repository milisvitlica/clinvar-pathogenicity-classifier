"""Clean the raw UniProt parquet into a classifier-ready protein feature table.

Reads data/raw/uniprot_human_reviewed.parquet (from ingest_uniprot.py) and writes
data/processed/uniprot_rag.parquet: reviewed human proteins with disease
involvement, with the leading "FUNCTION:/DISEASE:/TISSUE SPECIFICITY:" prefixes
stripped from the free-text columns.
"""

from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / "data/raw"
data_processed = project_root / "data/processed"

UNIPROT_RAW = data_raw / "uniprot_human_reviewed.parquet"
UNIPROT_CLEAN = data_processed / "uniprot_rag.parquet"


def main() -> None:
    df = pd.read_parquet(UNIPROT_RAW)

    df["Function [CC]"] = df["Function [CC]"].str.replace("FUNCTION: ", "")
    df["Involvement in disease"] = df["Involvement in disease"].str.replace("DISEASE: ", "")
    df["Tissue specificity"] = df["Tissue specificity"].str.replace("TISSUE SPECIFICITY: ", "")

    df = df[df["Involvement in disease"].notna()].copy()

    data_processed.mkdir(parents=True, exist_ok=True)
    df.to_parquet(UNIPROT_CLEAN, index=False)
    print("Saved cleaned parquet:", UNIPROT_CLEAN, df.shape)


if __name__ == "__main__":
    main()
