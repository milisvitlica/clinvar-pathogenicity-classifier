"""Left-join cleaned gnomAD frequencies onto ClinVar–UniProt position-matched tables.

Join key: gnomAD variant ID built from ClinVar GRCh38 ``Chromosome``, ``Start``,
``ReferenceAlleleVCF``, ``AlternateAlleleVCF`` (same encoding as ingest_gnomad.py).

Default: labelled position-matched table
  → data/processed/clinvar_uniprot_gnomad_position_matched.parquet

VUS: ``--vus``
  → data/processed/clinvar_uniprot_gnomad_position_matched_vus.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ingest_gnomad import gnomad_variant_id

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data/processed"

GNOMAD_CLEAN = data_processed / "gnomad_clean.parquet"
POSITION_MATCHED_IN = data_processed / "clinvar_uniprot_position_matched.parquet"
POSITION_MATCHED_VUS_IN = data_processed / "clinvar_uniprot_position_matched_vus.parquet"
JOINED_OUT = data_processed / "clinvar_uniprot_gnomad_position_matched.parquet"
JOINED_VUS_OUT = data_processed / "clinvar_uniprot_gnomad_position_matched_vus.parquet"

GNOMAD_KEEP = [
    "gnomad_variant_id",
    "in_gnomad",
    "gnomad_af",
    "gnomad_af_popmax",
    "gnomad_an",
    "gnomad_nhomalt",
    "gnomad_filter_pass",
    "rsids",
    "flags",
    "exome_af",
    "genome_af",
    "joint_af",
    "exome_an",
    "genome_an",
    "joint_an",
    "exome_nhomalt",
    "genome_nhomalt",
    "joint_nhomalt",
    "exome_af_afr",
    "exome_af_amr",
    "exome_af_asj",
    "exome_af_eas",
    "exome_af_fin",
    "exome_af_mid",
    "exome_af_nfe",
    "exome_af_sas",
    "joint_af_afr",
    "joint_af_amr",
    "joint_af_asj",
    "joint_af_eas",
    "joint_af_fin",
    "joint_af_mid",
    "joint_af_nfe",
    "joint_af_sas",
    "joint_faf95_popmax",
    "joint_faf95_popmax_pop",
]


def add_variant_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gnomad_variant_id"] = [
        gnomad_variant_id(c, p, r, a) if pd.notna(p) and pd.notna(r) and pd.notna(a) else pd.NA
        for c, p, r, a in zip(
            out["Chromosome"], out["Start"],
            out["ReferenceAlleleVCF"], out["AlternateAlleleVCF"],
        )
    ]
    return out


def join_gnomad(
    clinvar: pd.DataFrame,
    gnomad: pd.DataFrame,
) -> pd.DataFrame:
    gnomad_cols = [c for c in GNOMAD_KEEP if c in gnomad.columns]
    right = gnomad[gnomad_cols].drop_duplicates("gnomad_variant_id")
    left = add_variant_id(clinvar)
    merged = left.merge(right, on="gnomad_variant_id", how="left")
    merged["in_gnomad"] = merged["in_gnomad"].astype("boolean").fillna(False).astype(bool)
    merged["gnomad_filter_pass"] = (
        merged["gnomad_filter_pass"].astype("boolean").fillna(False).astype(bool)
    )
    return merged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinvar", type=Path, default=POSITION_MATCHED_IN)
    parser.add_argument("--gnomad", type=Path, default=GNOMAD_CLEAN)
    parser.add_argument("--out", type=Path, default=JOINED_OUT)
    parser.add_argument(
        "--vus",
        action="store_true",
        help=f"Shortcut: clinvar={POSITION_MATCHED_VUS_IN.name}, out={JOINED_VUS_OUT.name}",
    )
    args = parser.parse_args(argv)

    clinvar_path = POSITION_MATCHED_VUS_IN if args.vus else args.clinvar
    out_path = JOINED_VUS_OUT if args.vus else args.out

    clinvar = pd.read_parquet(clinvar_path)
    gnomad = pd.read_parquet(args.gnomad)
    merged = join_gnomad(clinvar, gnomad)

    data_processed.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    has_locus = merged["gnomad_variant_id"].notna()
    n_found = int(merged["in_gnomad"].sum())
    print("Read position-matched:", clinvar_path, clinvar.shape)
    print("Read gnomAD:", args.gnomad, gnomad.shape)
    print("Saved:", out_path, merged.shape)
    print(
        f"Rows with a genomic key: {int(has_locus.sum())}/{len(merged)}; "
        f"in_gnomad={n_found} ({n_found / max(len(merged), 1):.1%})"
    )


if __name__ == "__main__":
    main()
