"""Outer-join cleaned UniProt + ClinVar parquets on gene.

Join key: UniProt primary gene (first token of "Gene Names") == ClinVar gene.
ClinVar "GeneSymbol" is a ";"-delimited list of genes (e.g. "KLLN;LOC130004273;MLDHR;PTEN"),
so each variant is first exploded to one row per gene; the original list is kept in
"GeneSymbolFull". The result is row-level: one row per (protein, variant-gene) pair for
shared genes, plus unmatched UniProt proteins (clinvar columns null) and unmatched ClinVar
variant-genes (uniprot columns null). A unified `gene`, a `match_type` flag, and a synthetic
`join_id` are added for EDA.

Default: labelled ClinVar -> clinvar_uniprot_joined.parquet
VUS:     ``--clinvar .../clinvar_clean_vus.parquet --out .../clinvar_uniprot_joined_vus.parquet``
         (left join: keep all VUS rows; skip unmatched UniProt-only proteins).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data/processed"

UNIPROT_CLEAN = data_processed / "uniprot_clean.parquet"
CLINVAR_CLEAN = data_processed / "clinvar_clean.parquet"
CLINVAR_CLEAN_VUS = data_processed / "clinvar_clean_vus.parquet"
JOINED_OUT = data_processed / "clinvar_uniprot_joined.parquet"
JOINED_VUS_OUT = data_processed / "clinvar_uniprot_joined_vus.parquet"

GENE_KEY = "gene_key"


def _first_gene(gene_names) -> str:
    # UniProt "Gene Names" is space-delimited; the first token is the primary symbol.
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


def build_joined_dataframe(
    *,
    clinvar_path: Path = CLINVAR_CLEAN,
    uniprot_path: Path = UNIPROT_CLEAN,
    how: str = "outer",
) -> pd.DataFrame:
    """Join UniProt proteins to ClinVar variants on gene.

    ``how="outer"`` — full EDA join (labelled training path).
    ``how="left"`` — ClinVar-centric (VUS inference): all variants kept, UniProt-only dropped.
    """
    uniprot = pd.read_parquet(uniprot_path)
    clinvar = pd.read_parquet(clinvar_path)

    uniprot = uniprot.copy()
    uniprot[GENE_KEY] = uniprot["Gene Names"].map(_first_gene)

    clinvar = _explode_clinvar_genes(clinvar)

    if how == "left":
        merged = clinvar.merge(
            uniprot,
            left_on="GeneSymbol",
            right_on=GENE_KEY,
            how="left",
        )
    elif how == "outer":
        merged = uniprot.merge(
            clinvar,
            left_on=GENE_KEY,
            right_on="GeneSymbol",
            how="outer",
        )
    else:
        raise ValueError(f"Unsupported join how={how!r}; use 'outer' or 'left'")

    has_uniprot = merged["Entry"].notna()
    has_clinvar = merged["VariationID"].notna()
    # both = training rows; the *_only rows exist for EDA coverage checks.
    merged["match_type"] = "clinvar_only"
    merged.loc[has_uniprot & ~has_clinvar, "match_type"] = "uniprot_only"
    merged.loc[has_uniprot & has_clinvar, "match_type"] = "both"

    # Unified gene label (UniProt primary gene, falling back to ClinVar symbol)
    merged["gene"] = merged[GENE_KEY].where(merged[GENE_KEY].astype(bool), pd.NA)
    merged["gene"] = merged["gene"].fillna(merged["GeneSymbol"])

    merged = merged.reset_index(drop=True)
    merged.insert(0, "join_id", merged.index.astype(str))
    return merged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinvar", type=Path, default=CLINVAR_CLEAN)
    parser.add_argument("--uniprot", type=Path, default=UNIPROT_CLEAN)
    parser.add_argument("--out", type=Path, default=JOINED_OUT)
    parser.add_argument(
        "--how",
        choices=["outer", "left"],
        default="outer",
        help="outer = labelled EDA join; left = ClinVar-centric (VUS)",
    )
    parser.add_argument(
        "--vus",
        action="store_true",
        help=f"Shortcut: clinvar={CLINVAR_CLEAN_VUS.name}, out={JOINED_VUS_OUT.name}, how=left",
    )
    args = parser.parse_args(argv)

    clinvar_path = CLINVAR_CLEAN_VUS if args.vus else args.clinvar
    out_path = JOINED_VUS_OUT if args.vus else args.out
    how = "left" if args.vus else args.how

    merged = build_joined_dataframe(
        clinvar_path=clinvar_path,
        uniprot_path=args.uniprot,
        how=how,
    )
    data_processed.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    counts = merged["match_type"].value_counts().to_dict()
    print("Saved joined parquet:", out_path, merged.shape)
    print("match_type breakdown:", counts)


if __name__ == "__main__":
    main()
