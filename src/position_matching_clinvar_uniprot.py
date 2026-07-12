"""Add protein-position context to the gene-level ClinVar–UniProt join.

Reads data/processed/clinvar_uniprot_joined.parquet, extracts the amino-acid
position from ClinVar HGVS protein notation (p.), and annotates each variant
with UniProt feature overlap flags plus the closest annotated feature and its
distance (in amino-acid residues).

Writes data/processed/clinvar_uniprot_position_matched.parquet (leaves the
gene-level join unchanged).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
data_processed = project_root / "data/processed"
JOINED_IN = data_processed / "clinvar_uniprot_joined.parquet"
POSITION_MATCHED_OUT = data_processed / "clinvar_uniprot_position_matched.parquet"

FEATURE_COLUMNS: dict[str, str] = {
    "Domain [FT]": "DOMAIN",
    "Region": "REGION",
    "Zinc finger": "ZN_FING",
    "Active site": "ACT_SITE",
    "Binding site": "BINDING",
    "Disulfide bond": "DISULFID",
    "Modified residue": "MOD_RES",
}

# Prefer functional sites when several features share the same distance.
FEATURE_TYPE_PRIORITY: dict[str, int] = {
    "ACT_SITE": 0,
    "BINDING": 1,
    "ZN_FING": 2,
    "MOD_RES": 3,
    "DISULFID": 4,
    "DOMAIN": 5,
    "REGION": 6,
}

CONTEXT_COLUMNS = [
    "protein_position",
    "has_protein_position",
    "in_domain",
    "domain_names",
    "in_region",
    "region_names",
    "in_zinc_finger",
    "in_active_site",
    "in_binding_site",
    "in_disulfide",
    "in_mod_res",
    "in_functional_site",
    "in_any_feature",
    "closest_feature_type",
    "closest_feature_description",
    "distance_to_closest_feature",
]

PROTEIN_HGVS_RE = re.compile(r"p\.[A-Za-z]{3}(\d+)")


def extract_protein_position(name_field) -> int | None:
    """Return the first amino-acid index from ClinVar protein HGVS (p.) notation."""
    if pd.isna(name_field):
        return None
    match = PROTEIN_HGVS_RE.search(str(name_field))
    return int(match.group(1)) if match else None


def parse_uniprot_features(text, prefix: str) -> list[dict]:
    """Parse UniProt feature strings like ``DOMAIN 1642..1736; /note="BRCT 1"``."""
    if pd.isna(text) or not str(text).strip():
        return []
    features: list[dict] = []
    for match in re.finditer(rf"{re.escape(prefix)}\s+(\d+)(?:\.\.(\d+))?", str(text)):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        tail = str(text)[match.end() :]
        note_match = re.match(r';\s*/note="([^"]*)"', tail)
        features.append(
            {
                "start": start,
                "end": end,
                "description": note_match.group(1) if note_match else None,
            }
        )
    return features


def build_features_by_entry(proteins: pd.DataFrame) -> dict[str, list[dict]]:
    """Pre-parse UniProt feature intervals keyed by accession (Entry)."""
    features_by_entry: dict[str, list[dict]] = {}
    for _, row in proteins.drop_duplicates("Entry").iterrows():
        entry_feats: list[dict] = []
        for column, feature_type in FEATURE_COLUMNS.items():
            for feat in parse_uniprot_features(row[column], feature_type):
                entry_feats.append({**feat, "feature_type": feature_type})
        features_by_entry[row["Entry"]] = entry_feats
    return features_by_entry


def feature_distance(pos: int, feat: dict) -> int:
    """Amino-acid distance from ``pos`` to a UniProt feature.

    Interval features (domain, region, …): 0 if inside ``[start, end]``, else
    distance to the nearer boundary.

    Disulfide bonds annotate the two bonded cysteines, not the intervening
    span, so distance is to the nearer endpoint.
    """
    start, end = feat["start"], feat["end"]
    if feat["feature_type"] == "DISULFID" and start != end:
        return min(abs(pos - start), abs(pos - end))
    if start <= pos <= end:
        return 0
    if pos < start:
        return start - pos
    return pos - end


def position_overlaps_feature(pos: int, feat: dict) -> bool:
    """True when ``pos`` lies on the annotated feature (not merely near it)."""
    start, end = feat["start"], feat["end"]
    if feat["feature_type"] == "DISULFID" and start != end:
        return pos == start or pos == end
    return start <= pos <= end


def _empty_context() -> dict:
    return {
        "in_domain": False,
        "domain_names": None,
        "in_region": False,
        "region_names": None,
        "in_zinc_finger": False,
        "in_active_site": False,
        "in_binding_site": False,
        "in_disulfide": False,
        "in_mod_res": False,
        "in_functional_site": False,
        "in_any_feature": False,
        "closest_feature_type": None,
        "closest_feature_description": None,
        "distance_to_closest_feature": pd.NA,
    }


def get_protein_context(
    entry: str | None,
    protein_pos,
    features_by_entry: dict[str, list[dict]],
) -> dict:
    """Return overlap flags plus closest UniProt feature and distance."""
    if pd.isna(protein_pos) or entry not in features_by_entry:
        return _empty_context()

    pos = int(protein_pos)
    result = _empty_context()
    domain_names: list[str] = []
    region_names: list[str] = []

    features = features_by_entry[entry]
    if not features:
        return result

    closest = min(
        features,
        key=lambda feat: (
            feature_distance(pos, feat),
            FEATURE_TYPE_PRIORITY.get(feat["feature_type"], 99),
            feat["start"],
        ),
    )
    result["closest_feature_type"] = closest["feature_type"]
    result["closest_feature_description"] = closest["description"]
    result["distance_to_closest_feature"] = feature_distance(pos, closest)

    for feat in features:
        if not position_overlaps_feature(pos, feat):
            continue
        ftype = feat["feature_type"]
        if ftype == "DOMAIN":
            result["in_domain"] = True
            if feat["description"]:
                domain_names.append(feat["description"])
        elif ftype == "REGION":
            result["in_region"] = True
            if feat["description"]:
                region_names.append(feat["description"])
        elif ftype == "ZN_FING":
            result["in_zinc_finger"] = True
        elif ftype == "ACT_SITE":
            result["in_active_site"] = True
        elif ftype == "BINDING":
            result["in_binding_site"] = True
        elif ftype == "DISULFID":
            result["in_disulfide"] = True
        elif ftype == "MOD_RES":
            result["in_mod_res"] = True

    if domain_names:
        result["domain_names"] = "; ".join(dict.fromkeys(domain_names))
    if region_names:
        result["region_names"] = "; ".join(dict.fromkeys(region_names))

    result["in_functional_site"] = (
        result["in_active_site"] or result["in_binding_site"] or result["in_zinc_finger"]
    )
    result["in_any_feature"] = (
        result["in_domain"]
        or result["in_region"]
        or result["in_functional_site"]
        or result["in_disulfide"]
        or result["in_mod_res"]
    )
    return result


def enrich_joined_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add protein position, overlap flags, and closest-feature columns."""
    df = df.copy()

    proteins = df[df["Entry"].notna()].copy()
    features_by_entry = build_features_by_entry(proteins)

    df["protein_position"] = df["Name"].map(extract_protein_position).astype("Int64")
    df["has_protein_position"] = df["protein_position"].notna()

    context = df.apply(
        lambda row: get_protein_context(row["Entry"], row["protein_position"], features_by_entry),
        axis=1,
    )
    context_df = pd.DataFrame(context.tolist(), index=df.index)
    context_df["distance_to_closest_feature"] = context_df["distance_to_closest_feature"].astype(
        "Int64"
    )
    return pd.concat([df, context_df], axis=1)


def main() -> None:
    df = pd.read_parquet(JOINED_IN)
    enriched = enrich_joined_dataframe(df)

    data_processed.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(POSITION_MATCHED_OUT, index=False)

    both = enriched[enriched["match_type"] == "both"]
    with_pos = both[both["has_protein_position"]]
    in_domain = both["in_domain"].sum()
    in_functional = both["in_functional_site"].sum()
    has_closest = with_pos["closest_feature_type"].notna().sum()
    inside = (with_pos["distance_to_closest_feature"] == 0).sum()

    print("Read gene-level join:", JOINED_IN, df.shape)
    print("Saved position-matched parquet:", POSITION_MATCHED_OUT, enriched.shape)
    print(
        f"Matched variants with protein position: {len(with_pos):,} / {len(both):,} "
        f"({len(with_pos) / len(both):.1%})"
    )
    print(f"Matched variants in a UniProt domain: {in_domain:,} ({in_domain / len(both):.1%})")
    print(
        f"Matched variants in active/binding/zinc-finger site: "
        f"{in_functional:,} ({in_functional / len(both):.1%})"
    )
    print(
        f"With position and a closest feature: {has_closest:,} / {len(with_pos):,} "
        f"({has_closest / len(with_pos):.1%})"
    )
    print(
        f"Distance 0 (inside closest feature): {inside:,} / {len(with_pos):,} "
        f"({inside / len(with_pos):.1%})"
    )
    if has_closest:
        dist = with_pos["distance_to_closest_feature"].dropna()
        print(
            f"Distance to closest feature — median {dist.median():.0f}, "
            f"mean {dist.mean():.1f}"
        )


if __name__ == "__main__":
    main()
