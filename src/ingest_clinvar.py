"""Download ClinVar variant summary and write data/raw/clinvar_reliable_grch38.parquet."""

from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / "data/raw"
CLINVAR_PARQUET = data_raw / "clinvar_reliable_grch38.parquet"
CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

columns = [
    "VariationID", "Name", "GeneSymbol", "Chromosome", "Start", "Stop",
    "ClinicalSignificance", "ReviewStatus",
    "Type", "PhenotypeList", "PhenotypeIDS", "Assembly", "HGNC_ID",
    "NumberSubmitters", "LastEvaluated",
]
dtype_map = {
    "VariationID": "int32", "Name": "string", "GeneSymbol": "category",
    "Chromosome": "category", "Start": "Int64", "Stop": "Int64",
    "ClinicalSignificance": "category", "ReviewStatus": "category",
    "Type": "category", "PhenotypeList": "string", "PhenotypeIDS": "string",
    "Assembly": "category", "HGNC_ID": "string", "NumberSubmitters": "Int16",
}
allowed_review = {
    "practice guideline",
    "reviewed by expert panel",
    # "criteria provided, multiple submitters, no conflicts", # too many...
}

chunks = []
for chunk in pd.read_csv(
    CLINVAR_URL, sep="\t", compression="gzip", usecols=columns,
    dtype=dtype_map, chunksize=100_000, low_memory=True,
):
    filtered = chunk[
        (chunk["Assembly"] == "GRCh38")
        & (chunk["ReviewStatus"].isin(allowed_review))
    ]
    chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True).drop_duplicates()
data_raw.mkdir(parents=True, exist_ok=True)
df.to_parquet(CLINVAR_PARQUET, index=False)
print("Saved:", CLINVAR_PARQUET, df.shape)
