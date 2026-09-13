"""Download gnomAD v4 frequencies for ClinVar SNVs → data/raw/gnomad_clinvar_sites.parquet.

The full gnomAD sites VCF is hundreds of GB. This project only needs allele
frequencies at expert-reviewed ClinVar SNVs (~15k GRCh38 loci), so ingest
queries the public gnomAD GraphQL API (`dataset: gnomad_r4`) for those sites
and stores one row per queried variant (including sites absent from gnomAD).

Resume-safe: re-running skips variant IDs already in the output parquet.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / "data/raw"
data_processed = project_root / "data/processed"

GNOMAD_PARQUET = data_raw / "gnomad_clinvar_sites.parquet"
CLINVAR_CLEAN = data_processed / "clinvar_clean.parquet"
CLINVAR_CLEAN_VUS = data_processed / "clinvar_clean_vus.parquet"

GNOMAD_API = "https://gnomad.broadinstitute.org/api"
DATASET = "gnomad_r4"
BATCH_SIZE = 10
MAX_WORKERS = 2
MAX_RETRIES = 6
USER_AGENT = "clinvar-pathogenicity-classifier/0.1 (gnomAD frequency ingest)"

VARIANT_FIELDS = """
    variant_id
    chrom
    pos
    ref
    alt
    flags
    rsids
    exome {
      ac an af ac_hom filters flags
      faf95 { popmax popmax_population }
      populations { id ac an ac_hom }
    }
    genome {
      ac an af ac_hom filters flags
      faf95 { popmax popmax_population }
      populations { id ac an ac_hom }
    }
    joint {
      ac an homozygote_count filters
      faf95 { popmax popmax_population }
      populations { id ac an homozygote_count }
    }
"""


def gnomad_variant_id(chrom, pos, ref, alt) -> str:
    """gnomAD v4 variant ID (`chrom-pos-ref-alt`, mito contig `M`, no `chr` prefix)."""
    chrom_s = str(chrom).removeprefix("chr")
    if chrom_s == "MT":
        chrom_s = "M"
    return f"{chrom_s}-{int(pos)}-{ref}-{alt}"


def _post(session: requests.Session, query: str) -> dict:
    last_exc: Exception | None = None
    headers = {
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(
                GNOMAD_API,
                params={"n": uuid.uuid4().hex},
                json={"query": query},
                timeout=120,
                headers=headers,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(2 ** attempt, 60)
                print(f"  HTTP {resp.status_code}; retry {attempt}/{MAX_RETRIES} in {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            payload = resp.json()
            if "data" not in payload and payload.get("errors"):
                raise RuntimeError(payload["errors"])
            return payload
        except (requests.RequestException, json.JSONDecodeError, RuntimeError) as exc:
            last_exc = exc
            wait = min(2 ** attempt, 60)
            print(f"  retry {attempt}/{MAX_RETRIES} ({exc!r}) in {wait}s")
            time.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("gnomAD GraphQL request failed")


def _row_from_payload(variant_id: str, variant: dict | None, error: str | None) -> dict:
    return {
        "gnomad_variant_id": variant_id,
        "found": variant is not None,
        "error": error,
        "rsids": json.dumps((variant or {}).get("rsids") or []),
        "flags": json.dumps((variant or {}).get("flags") or []),
        "exome_json": json.dumps((variant or {}).get("exome")),
        "genome_json": json.dumps((variant or {}).get("genome")),
        "joint_json": json.dumps((variant or {}).get("joint")),
    }


def fetch_batch(session: requests.Session, variant_ids: list[str]) -> list[dict]:
    aliases = [f"loc{i}" for i in range(len(variant_ids))]
    parts = [
        f'{alias}: variant(variantId: "{vid}", dataset: {DATASET}) {{ {VARIANT_FIELDS} }}'
        for alias, vid in zip(aliases, variant_ids)
    ]
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        query = f'query {{ {" ".join(parts)} }} # {uuid.uuid4().hex}'
        try:
            payload = _post(session, query)
            data = payload.get("data") or {}
            if data and not set(aliases).intersection(data):
                raise RuntimeError(
                    f"unexpected GraphQL aliases {sorted(data)[:8]} (wanted {aliases[:3]}…)"
                )
            errors = payload.get("errors") or []
            error_by_alias: dict[str, str] = {}
            unmatched: list[str] = []
            for err in errors:
                message = str(err.get("message", err))
                path = err.get("path") or []
                if path:
                    error_by_alias[str(path[0])] = message
                else:
                    unmatched.append(message)

            rows = []
            for alias, vid in zip(aliases, variant_ids):
                variant = data.get(alias)
                error = error_by_alias.get(alias)
                if variant is not None and variant.get("variant_id") not in {None, vid}:
                    raise RuntimeError(
                        f"GraphQL mismatch: asked {vid}, got {variant.get('variant_id')}"
                    )
                if variant is None and error is None and unmatched:
                    error = unmatched[0]
                if variant is None and error is None:
                    error = "Variant not found"
                rows.append(_row_from_payload(vid, variant, error))
            return rows
        except RuntimeError as exc:
            last_exc = exc
            wait = min(2 ** attempt, 60)
            print(f"  batch retry {attempt}/{MAX_RETRIES} ({exc}) in {wait}s")
            time.sleep(wait)
    raise last_exc or RuntimeError("gnomAD batch failed")


def loci_from_clinvar(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_parquet(path, columns=[
            "Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF",
        ])
        frames.append(df)
    loci = pd.concat(frames, ignore_index=True).drop_duplicates()
    loci = loci.dropna(subset=["Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF"])
    loci["gnomad_variant_id"] = [
        gnomad_variant_id(c, p, r, a)
        for c, p, r, a in zip(
            loci["Chromosome"], loci["Start"],
            loci["ReferenceAlleleVCF"], loci["AlternateAlleleVCF"],
        )
    ]
    return loci.drop_duplicates("gnomad_variant_id").reset_index(drop=True)


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def ingest(
    *,
    clinvar_paths: list[Path] | None = None,
    out_path: Path = GNOMAD_PARQUET,
    batch_size: int = BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
    limit: int | None = None,
) -> pd.DataFrame:
    clinvar_paths = clinvar_paths or [CLINVAR_CLEAN, CLINVAR_CLEAN_VUS]
    missing = [p for p in clinvar_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Cleaned ClinVar tables are required so we only query ClinVar SNVs. "
            f"Missing: {missing}. Run ingest_clinvar.py then clean_clinvar.py first."
        )

    loci = loci_from_clinvar(clinvar_paths)
    if limit is not None:
        loci = loci.head(int(limit)).copy()
    todo = list(loci["gnomad_variant_id"])
    existing = pd.DataFrame()
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        retry = existing["error"].fillna("").astype(str).isin(
            {"missing from GraphQL data", "unexpected GraphQL aliases"}
        )
        existing = existing.loc[~retry].copy()
        done = set(existing["gnomad_variant_id"].astype(str))
        todo = [vid for vid in todo if vid not in done]
        print(f"Resuming: {len(done)} already fetched, {len(todo)} remaining")

    if not todo:
        print("Nothing to fetch:", out_path, existing.shape)
        return existing

    batches = _chunks(todo, batch_size)
    print(f"Querying gnomAD {DATASET} for {len(todo)} variants in {len(batches)} batches")
    new_rows: list[dict] = []

    def run_batch(batch: list[str]) -> list[dict]:
        # One session per worker; requests.Session is not thread-safe.
        worker_session = requests.Session()
        return fetch_batch(worker_session, batch)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_batch, batch): batch for batch in batches}
        for fut in as_completed(futures):
            rows = fut.result()
            new_rows.extend(rows)
            completed += 1
            found = sum(r["found"] for r in rows)
            print(
                f"  batch {completed}/{len(batches)} "
                f"found {found}/{len(rows)} (total new {len(new_rows)}/{len(todo)})"
            )
            if completed % 20 == 0:
                _write(out_path, existing, new_rows)

    out = _write(out_path, existing, new_rows)
    n_found = int(out["found"].sum()) if "found" in out.columns else 0
    print("Saved:", out_path, out.shape, f"in_gnomad={n_found}/{len(out)}")
    return out


def _write(out_path: Path, existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    parts = []
    if not existing.empty:
        parts.append(existing)
    if new_rows:
        parts.append(pd.DataFrame(new_rows))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out = out.drop_duplicates("gnomad_variant_id", keep="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--out", type=Path, default=GNOMAD_PARQUET)
    parser.add_argument("--limit", type=int, default=None, help="Query at most N loci (smoke test)")
    args = parser.parse_args(argv)
    ingest(
        out_path=args.out,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
