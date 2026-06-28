"""Outer-join cleaned UniProt + ClinVar parquets on gene and write data/processed/joined.parquet.

Join key: UniProt primary gene (first token of "Gene Names") == ClinVar gene.
ClinVar "GeneSymbol" is a ";"-delimited list of genes (e.g. "KLLN;LOC130004273;MLDHR;PTEN"),
so each variant is first exploded to one row per gene; the original list is kept in
"GeneSymbolFull". The result is row-level: one row per (protein, variant-gene) pair for
shared genes, plus unmatched UniProt proteins (clinvar columns null) and unmatched ClinVar
variant-genes (uniprot columns null). A unified `gene`, a `match_type` flag, and a synthetic
`join_id` are added for EDA.
"""

from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data/processed"

UNIPROT_CLEAN = data_processed / "uniprot_rag.parquet"
CLINVAR_CLEAN = data_processed / "clinvar_rag.parquet"
JOINED_OUT = data_processed / "joined.parquet"

GENE_KEY = "gene_key"


def _first_gene(gene_names) -> str:
    if pd.isna(gene_names):
        return ""
    tokens = str(gene_names).split()
    return tokens[0] if tokens else ""


def _explode_clinvar_genes(clinvar: pd.DataFrame) -> pd.DataFrame:
    # ClinVar GeneSymbol is a ";"-delimited gene list; explode to one gene per row.
    clinvar = clinvar.copy()
    clinvar["GeneSymbolFull"] = clinvar["GeneSymbol"].astype("string")
    clinvar["GeneSymbol"] = (
        clinvar["GeneSymbolFull"].str.split(";").map(
            lambda genes: [g.strip() for g in genes if g and g.strip()]
            if isinstance(genes, list)
            else genes
        )
    )
    clinvar = clinvar.explode("GeneSymbol", ignore_index=True)
    clinvar["GeneSymbol"] = clinvar["GeneSymbol"].astype("string")
    return clinvar


def build_joined_dataframe() -> pd.DataFrame:
    uniprot = pd.read_parquet(UNIPROT_CLEAN)
    clinvar = pd.read_parquet(CLINVAR_CLEAN)

    uniprot = uniprot.copy()
    uniprot[GENE_KEY] = uniprot["Gene Names"].map(_first_gene)

    clinvar = _explode_clinvar_genes(clinvar)

    merged = uniprot.merge(
        clinvar,
        left_on=GENE_KEY,
        right_on="GeneSymbol",
        how="outer",
    )

    has_uniprot = merged["Entry"].notna()
    has_clinvar = merged["VariationID"].notna()
    merged["match_type"] = "clinvar_only"
    merged.loc[has_uniprot & ~has_clinvar, "match_type"] = "uniprot_only"
    merged.loc[has_uniprot & has_clinvar, "match_type"] = "both"

    # Unified gene label (UniProt primary gene, falling back to ClinVar symbol)
    merged["gene"] = merged[GENE_KEY].where(merged[GENE_KEY].astype(bool), pd.NA)
    merged["gene"] = merged["gene"].fillna(merged["GeneSymbol"])

    merged = merged.reset_index(drop=True)
    merged.insert(0, "join_id", merged.index.astype(str))
    return merged


def main() -> None:
    merged = build_joined_dataframe()
    data_processed.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(JOINED_OUT, index=False)
    counts = merged["match_type"].value_counts().to_dict()
    print("Saved joined parquet:", JOINED_OUT, merged.shape)
    print("match_type breakdown:", counts)


if __name__ == "__main__":
    main()
