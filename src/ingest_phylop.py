"""Download UCSC hg38 phyloP100way at ClinVar SNV positions.

PhyloP is a **model** feature (cross-species constraint at the reference base).
ClinVar labels sometimes use conservation as ACMG PP3/BP4 — supporting/moderate
computational evidence, not stand-alone like BA1 — so there is some leakage,
weaker than gnomAD AF.

Writes data/processed/phylop_clean.parquet (one row per chrom-pos).
Re-runs skip positions already on disk. Mito/unmapped sites stay ``found=False``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data/processed"

PHYLOP_CLEAN = data_processed / "phylop_clean.parquet"
CLINVAR_CLEAN = data_processed / "clinvar_clean.parquet"
CLINVAR_CLEAN_VUS = data_processed / "clinvar_clean_vus.parquet"

UCSC_TRACK = "https://api.genome.ucsc.edu/getData/track"
MAX_WORKERS = 16
MAX_RETRIES = 6
CHECKPOINT_EVERY = 400


def clinvar_chrom(chrom) -> str:
    """ClinVar-style contig: no ``chr`` prefix (``MT`` stays ``MT``)."""
    return str(chrom).removeprefix("chr")


def ucsc_chrom(chrom) -> str:
    """UCSC hg38 name (``chr12``, mito ``chrM``)."""
    s = clinvar_chrom(chrom)
    if s == "MT":
        return "chrM"
    return f"chr{s}"


def clinvar_positions() -> pd.DataFrame:
    frames = []
    for path in (CLINVAR_CLEAN, CLINVAR_CLEAN_VUS):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run ingest_clinvar.py then clean_clinvar.py first."
            )
        frames.append(pd.read_parquet(path, columns=["Chromosome", "Start"]))
    loci = pd.concat(frames, ignore_index=True).dropna(subset=["Chromosome", "Start"])
    loci["chrom"] = loci["Chromosome"].map(clinvar_chrom)
    loci["pos"] = pd.to_numeric(loci["Start"], errors="coerce").astype("Int64")
    loci = loci.dropna(subset=["pos"]).drop_duplicates(["chrom", "pos"])
    loci["ucsc_chrom"] = loci["chrom"].map(ucsc_chrom)
    return loci[["chrom", "pos", "ucsc_chrom"]].reset_index(drop=True)


def fetch_phylop(ucsc_chr: str, pos: int) -> float | None:
    """1-based pos → UCSC 0-based half-open interval. None if no score."""
    last_exc: Exception | None = None
    params = {
        "genome": "hg38",
        "track": "phyloP100way",
        "chrom": ucsc_chr,
        "start": int(pos) - 1,
        "end": int(pos),
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(UCSC_TRACK, params=params, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(min(2 ** attempt, 60))
                continue
            resp.raise_for_status()
            payload = resp.json()
            items = payload.get("phyloP100way") or []
            if not items:
                return None
            return float(items[0]["value"])
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            last_exc = exc
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"{ucsc_chr}:{pos}: {last_exc}")


def _one(row: dict) -> dict:
    score = fetch_phylop(row["ucsc_chrom"], int(row["pos"]))
    return {
        "chrom": row["chrom"],
        "pos": int(row["pos"]),
        "ucsc_chrom": row["ucsc_chrom"],
        "phylop_100way": score,
        "found": score is not None,
    }


def _save(existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    parts = []
    if not existing.empty:
        parts.append(existing)
    if new_rows:
        parts.append(pd.DataFrame(new_rows))
    out = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(["chrom", "pos"], keep="last")
    )
    data_processed.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PHYLOP_CLEAN, index=False)
    return out


def main() -> None:
    loci = clinvar_positions()
    existing = pd.DataFrame()
    if PHYLOP_CLEAN.exists():
        existing = pd.read_parquet(PHYLOP_CLEAN)
        done = set(zip(existing["chrom"].astype(str), existing["pos"].astype(int)))
        mask = [((c, int(p)) not in done) for c, p in zip(loci["chrom"], loci["pos"])]
        loci = loci.loc[mask].reset_index(drop=True)
        print(f"Resuming: {len(done)} already fetched, {len(loci)} remaining", flush=True)
    if loci.empty:
        print("Nothing to fetch:", PHYLOP_CLEAN, existing.shape, flush=True)
        return

    todo = loci.to_dict("records")
    print(f"Querying UCSC phyloP100way (hg38) for {len(todo)} positions", flush=True)
    new_rows: list[dict] = []
    found = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_one, row): i for i, row in enumerate(todo, start=1)}
        for i, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            new_rows.append(row)
            found += int(row["found"])
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} found {found}", flush=True)
            if i % CHECKPOINT_EVERY == 0:
                _save(existing, new_rows)

    out = _save(existing, new_rows)
    n_found = int(out["found"].sum())
    print("Saved:", PHYLOP_CLEAN, out.shape, f"found={n_found}/{len(out)}", flush=True)


if __name__ == "__main__":
    main()
