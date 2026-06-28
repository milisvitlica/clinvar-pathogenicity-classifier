"""Download reviewed human UniProt entries and write data/raw/uniprot_human_reviewed.parquet."""

from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / "data/raw"
UNIPROT_PARQUET = data_raw / "uniprot_human_reviewed.parquet"
UNIPROT_STREAM_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?compressed=true"
    "&fields=accession,id,gene_names,protein_name,length,"
    "cc_function,cc_disease,cc_subcellular_location,cc_interaction,"
    "ft_domain,ft_region,ft_zn_fing,cc_domain,"
    "ft_variant,cc_polymorphism,"
    "cc_tissue_specificity,cc_developmental_stage"
    "&format=tsv"
    "&query=(organism_id:9606)%20AND%20(reviewed:true)"
)

df = pd.read_csv(UNIPROT_STREAM_URL, sep="\t", compression="gzip", low_memory=False)
data_raw.mkdir(parents=True, exist_ok=True)
df.to_parquet(UNIPROT_PARQUET, index=False)
print("Saved:", UNIPROT_PARQUET, df.shape)
