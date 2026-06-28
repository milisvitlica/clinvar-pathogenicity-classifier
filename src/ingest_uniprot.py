"""Download reviewed human UniProt entries and write data/raw/uniprot_human_reviewed.parquet.

Uses the UniProt REST *search* endpoint with cursor pagination instead of the
*stream* endpoint. The stream endpoint intermittently returns the body
"Error encountered when streaming data. Please try again later." for large
queries, which would otherwise be saved as a bogus 1-row parquet. Pagination +
retries + a sanity check make the download robust.
"""

import io
import time
from pathlib import Path

import pandas as pd
import requests

project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / "data/raw"
UNIPROT_PARQUET = data_raw / "uniprot_human_reviewed.parquet"

SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = (
    "accession,id,gene_names,protein_name,length,"
    "cc_function,cc_disease,cc_subcellular_location,cc_interaction,"
    "ft_domain,ft_region,ft_zn_fing,cc_domain,"
    "ft_act_site,ft_binding,ft_disulfid,ft_mod_res,"
    "ft_variant,cc_polymorphism,"
    "cc_tissue_specificity,cc_developmental_stage"
)
PARAMS = {
    "query": "(organism_id:9606) AND (reviewed:true)",
    "fields": FIELDS,
    "format": "tsv",
    "size": 500,
}
MAX_RETRIES = 5


def _get(url: str, params: dict | None = None) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=120)
            if resp.ok and "Error encountered when streaming" not in resp.text[:200]:
                return resp
            reason = f"HTTP {resp.status_code}" if not resp.ok else "server stream error"
        except requests.RequestException as exc:  # network hiccup
            last_exc = exc
            reason = repr(exc)
        wait = 2 ** attempt
        print(f"  retry {attempt}/{MAX_RETRIES} ({reason}) in {wait}s")
        time.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"UniProt request failed after {MAX_RETRIES} retries: {url}")


def fetch() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    resp = _get(SEARCH_URL, params=PARAMS)
    total = resp.headers.get("x-total-results")
    while True:
        frames.append(pd.read_csv(io.StringIO(resp.text), sep="\t", low_memory=False))
        got = sum(len(f) for f in frames)
        print(f"  fetched {got}" + (f"/{total}" if total else ""))
        next_url = resp.links.get("next", {}).get("url")
        if not next_url:
            break
        resp = _get(next_url)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = fetch()
    if df.empty or "Entry" not in df.columns:
        raise RuntimeError(
            f"Unexpected UniProt response (shape={df.shape}, columns={list(df.columns)})"
        )
    data_raw.mkdir(parents=True, exist_ok=True)
    df.to_parquet(UNIPROT_PARQUET, index=False)
    print("Saved:", UNIPROT_PARQUET, df.shape)


if __name__ == "__main__":
    main()
