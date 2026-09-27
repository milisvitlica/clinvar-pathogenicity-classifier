"""Clean raw gnomAD site lookups into a frequency table (EDA, not modeling).

Reads data/raw/gnomad_clinvar_sites.parquet (from ingest_gnomad.py) and writes
data/processed/gnomad_clean.parquet: one row per ClinVar SNV queried against
gnomAD v4, with joint/exome/genome AC/AN/AF, filtering allele frequency, and
continental genetic-ancestry frequencies.

Preferred AF is joint (exomes+genomes) when present, else exome, else genome.
Sites not found keep ``in_gnomad=False`` and null AFs. For EDA plots,
``add_gnomad_features`` (in features.py) fills those to 0 — that helper is not
on the training matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / "data/raw"
data_processed = project_root / "data/processed"

GNOMAD_RAW = data_raw / "gnomad_clinvar_sites.parquet"
GNOMAD_CLEAN = data_processed / "gnomad_clean.parquet"

# Genetic ancestry groups in gnomAD v4. Ignore sex-split ids like nfe_XX / nfe_XY.
POPULATIONS = ["afr", "ami", "amr", "asj", "eas", "fin", "mid", "nfe", "remaining", "sas"]


def _loads(value):
    # ingest stored nested GraphQL objects as JSON strings (or null).
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value)
    if text in {"", "null", "None"}:
        return None
    return json.loads(text)


def _af(ac, an) -> float | None:
    # Allele frequency = alt copies / chromosomes sequenced at this site.
    try:
        an_f = float(an)
        ac_f = float(ac)
    except (TypeError, ValueError):
        return None
    if an_f <= 0 or pd.isna(an_f) or pd.isna(ac_f):
        return None
    return ac_f / an_f


def _join_filters(values) -> str | None:
    """``None`` if the source is missing; empty string if FILTER is PASS."""
    if values is None:
        return None
    if not values:
        return ""
    return ";".join(str(v) for v in values)


def _pop_map(populations: list | None, hom_key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for pop in populations or []:
        pop_id = str(pop.get("id", ""))
        if pop_id not in POPULATIONS:
            continue
        ac, an = pop.get("ac"), pop.get("an")
        out[pop_id] = {
            "ac": ac,
            "an": an,
            "af": _af(ac, an),
            "nhomalt": pop.get(hom_key),
        }
    return out


def _flatten_source(prefix: str, payload: dict | None, *, hom_key: str) -> dict:
    """Turn one GraphQL block (exome / genome / joint) into flat columns.

    hom_key differs: exome/genome use ``ac_hom``, joint uses ``homozygote_count``.
    """
    cols = {
        f"{prefix}_ac": None,
        f"{prefix}_an": None,
        f"{prefix}_af": None,
        f"{prefix}_nhomalt": None,
        f"{prefix}_filters": None,
        f"{prefix}_flags": None,
        f"{prefix}_faf95_popmax": None,
        f"{prefix}_faf95_popmax_pop": None,
    }
    for pop in POPULATIONS:
        cols[f"{prefix}_af_{pop}"] = None
    if not payload:
        return cols

    ac, an = payload.get("ac"), payload.get("an")
    cols[f"{prefix}_ac"] = ac
    cols[f"{prefix}_an"] = an
    cols[f"{prefix}_af"] = payload.get("af") if payload.get("af") is not None else _af(ac, an)
    cols[f"{prefix}_nhomalt"] = payload.get(hom_key)
    cols[f"{prefix}_filters"] = _join_filters(payload.get("filters"))
    cols[f"{prefix}_flags"] = _join_filters(payload.get("flags"))
    faf95 = payload.get("faf95") or {}
    cols[f"{prefix}_faf95_popmax"] = faf95.get("popmax")
    cols[f"{prefix}_faf95_popmax_pop"] = faf95.get("popmax_population")
    for pop, stats in _pop_map(payload.get("populations"), hom_key).items():
        cols[f"{prefix}_af_{pop}"] = stats["af"]
    return cols


def _preferred_af(joint_af, exome_af, genome_af):
    # Joint has the largest sample size; fall back if that callset missed the site.
    for value in (joint_af, exome_af, genome_af):
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            return value
    return None


def _pass_filters(*filter_strings) -> bool:
    """True when every available sequencing source has an empty FILTER set."""
    seen = False
    for raw in filter_strings:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        seen = True
        if str(raw).strip():
            return False
    return seen


def clean(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in df.itertuples(index=False):
        found = bool(getattr(rec, "found", False))
        exome = _loads(getattr(rec, "exome_json", None)) if found else None
        genome = _loads(getattr(rec, "genome_json", None)) if found else None
        joint = _loads(getattr(rec, "joint_json", None)) if found else None

        row = {
            "gnomad_variant_id": rec.gnomad_variant_id,
            "in_gnomad": found and (exome is not None or genome is not None or joint is not None),
            "rsids": ";".join(_loads(getattr(rec, "rsids", "[]")) or []) or None,
            "flags": _join_filters(_loads(getattr(rec, "flags", "[]"))),
            "error": getattr(rec, "error", None) if not found else None,
        }
        row.update(_flatten_source("exome", exome, hom_key="ac_hom"))
        row.update(_flatten_source("genome", genome, hom_key="ac_hom"))
        row.update(_flatten_source("joint", joint, hom_key="homozygote_count"))

        # EDA columns; FEATURE_COLUMNS in features.py does not include these.
        row["gnomad_af"] = _preferred_af(row["joint_af"], row["exome_af"], row["genome_af"])
        row["gnomad_an"] = _preferred_af(row["joint_an"], row["exome_an"], row["genome_an"])
        row["gnomad_nhomalt"] = _preferred_af(
            row["joint_nhomalt"], row["exome_nhomalt"], row["genome_nhomalt"]
        )
        row["gnomad_af_popmax"] = _preferred_af(
            row["joint_faf95_popmax"],
            row["exome_faf95_popmax"],
            row["genome_faf95_popmax"],
        )
        if row["gnomad_af_popmax"] is None:
            # No FAF95: use the highest continental AF (not the "remaining" leftover group).
            pop_afs = [
                row[f"joint_af_{pop}"] if row[f"joint_af_{pop}"] is not None
                else row[f"exome_af_{pop}"] if row[f"exome_af_{pop}"] is not None
                else row[f"genome_af_{pop}"]
                for pop in POPULATIONS
                if pop != "remaining"
            ]
            pop_afs = [v for v in pop_afs if v is not None]
            row["gnomad_af_popmax"] = max(pop_afs) if pop_afs else row["gnomad_af"]

        row["gnomad_filter_pass"] = (
            _pass_filters(row["joint_filters"], row["exome_filters"], row["genome_filters"])
            if row["in_gnomad"]
            else False
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    float_cols = [c for c in out.columns if c.endswith("_af") or "af_" in c or c.endswith("_popmax")]
    for col in ["gnomad_af", "gnomad_af_popmax", "gnomad_an", "gnomad_nhomalt", *float_cols]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["in_gnomad"] = out["in_gnomad"].astype(bool)
    out["gnomad_filter_pass"] = out["gnomad_filter_pass"].astype(bool)
    return out.reset_index(drop=True)


def main() -> None:
    raw = pd.read_parquet(GNOMAD_RAW)
    cleaned = clean(raw)
    data_processed.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(GNOMAD_CLEAN, index=False)
    n = len(cleaned)
    n_found = int(cleaned["in_gnomad"].sum())
    print("Saved cleaned parquet:", GNOMAD_CLEAN, cleaned.shape)
    print(f"in_gnomad: {n_found}/{n} ({n_found / max(n, 1):.1%})")
    if n_found:
        af = cleaned.loc[cleaned["in_gnomad"], "gnomad_af"]
        print(
            f"AF among found — median {af.median():.2e}, "
            f"mean {af.mean():.2e}, max {af.max():.4f}"
        )
        print("filter_pass:", cleaned["gnomad_filter_pass"].value_counts().to_dict())


if __name__ == "__main__":
    main()
