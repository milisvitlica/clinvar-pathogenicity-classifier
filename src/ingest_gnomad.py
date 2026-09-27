"""Download gnomAD v4 frequencies for ClinVar SNVs → data/raw/gnomad_clinvar_sites.parquet.

EDA-only (not a model input). ClinVar P/B labels already use ACMG frequency
evidence, so AF is circular with the training target.

ClinVar/UniProt ingest pull one table. gnomAD’s public files are genome-wide
VCFs (hundreds of GB), so this script asks the GraphQL API only for the ~15k
ClinVar SNVs we already have. One row per queried variant, including absences.

Re-running skips variant IDs already in the output parquet.
"""

from __future__ import annotations  # postpone hint evaluation so list[str] / X | Y work on 3.9–3.10

import json
import time
from pathlib import Path

import pandas as pd
import requests

project_root = Path(__file__).resolve().parents[1]  # src/ -> repo root
data_raw = project_root / "data/raw"
data_processed = project_root / "data/processed"

GNOMAD_PARQUET = data_raw / "gnomad_clinvar_sites.parquet"
CLINVAR_CLEAN = data_processed / "clinvar_clean.parquet"
CLINVAR_CLEAN_VUS = data_processed / "clinvar_clean_vus.parquet"

GNOMAD_API = "https://gnomad.broadinstitute.org/api"
DATASET = "gnomad_r4"  # GRCh38, matches ClinVar Assembly filter
BATCH_SIZE = 10  # variants per GraphQL POST (larger batches 429 / time out)
MAX_RETRIES = 6

# One POST asks for several variants via aliases loc0, loc1, …
# Nested blocks: exome (~731k), genome (~76k), joint (combined counts).
FIELDS = """
    variant_id flags rsids
    exome { ac an af ac_hom filters flags
            faf95 { popmax popmax_population }
            populations { id ac an ac_hom } }
    genome { ac an af ac_hom filters flags
             faf95 { popmax popmax_population }
             populations { id ac an ac_hom } }
    joint { ac an homozygote_count filters
            faf95 { popmax popmax_population }
            populations { id ac an homozygote_count } }
"""


def gnomad_variant_id(chrom, pos, ref, alt) -> str:
    """gnomAD v4 id: `chrom-pos-ref-alt` (mito is `M`, no `chr` prefix)."""
    chrom_s = str(chrom).removeprefix("chr")
    if chrom_s == "MT":
        chrom_s = "M"
    return f"{chrom_s}-{int(pos)}-{ref}-{alt}"


def _post(query: str) -> dict:
    # gnomAD rate-limits; 429/5xx wait and retry with exponential backoff.
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                GNOMAD_API,
                json={"query": query},
                timeout=120,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(2 ** attempt, 60)
                print(f"  HTTP {resp.status_code}; retry {attempt}/{MAX_RETRIES} in {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data")
            if not data:
                raise RuntimeError(payload.get("errors") or "empty GraphQL data")
            return payload
        except (requests.RequestException, json.JSONDecodeError, RuntimeError) as exc:
            last_exc = exc
            wait = min(2 ** attempt, 60)
            print(f"  retry {attempt}/{MAX_RETRIES} ({exc!r}) in {wait}s")
            time.sleep(wait)
    raise last_exc or RuntimeError("gnomAD GraphQL request failed")


def fetch_batch(variant_ids: list[str]) -> list[dict]:
    # GraphQL: { loc0: variant(...) { fields } loc1: variant(...) { fields } }
    # Missing sites come back as null — we still write a row (found=False).
    parts = [
        f'loc{i}: variant(variantId: "{vid}", dataset: {DATASET}) {{ {FIELDS} }}'
        for i, vid in enumerate(variant_ids)
    ]
    payload = _post("query { " + " ".join(parts) + " }")
    data = payload.get("data") or {}
    rows = []
    for i, vid in enumerate(variant_ids):
        variant = data.get(f"loc{i}")
        # Keep exome/genome/joint as JSON strings; clean_gnomad.py flattens them.
        rows.append({
            "gnomad_variant_id": vid,
            "found": variant is not None,
            "error": None if variant is not None else "Variant not found",
            "rsids": json.dumps((variant or {}).get("rsids") or []),
            "flags": json.dumps((variant or {}).get("flags") or []),
            "exome_json": json.dumps((variant or {}).get("exome")),
            "genome_json": json.dumps((variant or {}).get("genome")),
            "joint_json": json.dumps((variant or {}).get("joint")),
        })
    return rows


def clinvar_variant_ids() -> list[str]:
    # Labelled P/B + VUS loci; dict.fromkeys keeps order and uniqueness.
    frames = []
    for path in (CLINVAR_CLEAN, CLINVAR_CLEAN_VUS):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run ingest_clinvar.py then clean_clinvar.py first."
            )
        frames.append(pd.read_parquet(path, columns=[
            "Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF",
        ]))
    loci = pd.concat(frames, ignore_index=True).dropna(
        subset=["Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF"]
    )
    ids = [
        gnomad_variant_id(c, p, r, a)
        for c, p, r, a in zip(
            loci["Chromosome"], loci["Start"],
            loci["ReferenceAlleleVCF"], loci["AlternateAlleleVCF"],
        )
    ]
    return list(dict.fromkeys(ids))


def main() -> None:
    todo = clinvar_variant_ids()
    existing = pd.DataFrame()
    if GNOMAD_PARQUET.exists():
        # Resume: a full run is ~15k lookups; skip IDs already on disk.
        existing = pd.read_parquet(GNOMAD_PARQUET)
        done = set(existing["gnomad_variant_id"].astype(str))
        todo = [vid for vid in todo if vid not in done]
        print(f"Resuming: {len(done)} already fetched, {len(todo)} remaining")
    if not todo:
        print("Nothing to fetch:", GNOMAD_PARQUET, existing.shape)
        return

    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    print(f"Querying gnomAD {DATASET} for {len(todo)} variants in {len(batches)} batches")
    new_rows: list[dict] = []
    for i, batch in enumerate(batches, start=1):
        rows = fetch_batch(batch)
        new_rows.extend(rows)
        found = sum(r["found"] for r in rows)
        print(f"  batch {i}/{len(batches)} found {found}/{len(rows)}")
        if i % 50 == 0:
            _save(existing, new_rows)  # checkpoint so a crash is not a full restart

    out = _save(existing, new_rows)
    print("Saved:", GNOMAD_PARQUET, out.shape, f"in_gnomad={int(out['found'].sum())}/{len(out)}")


def _save(existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    parts = []
    if not existing.empty:
        parts.append(existing)
    if new_rows:
        parts.append(pd.DataFrame(new_rows))
    out = pd.concat(parts, ignore_index=True).drop_duplicates("gnomad_variant_id", keep="last")
    data_raw.mkdir(parents=True, exist_ok=True)
    out.to_parquet(GNOMAD_PARQUET, index=False)
    return out


if __name__ == "__main__":
    main()
