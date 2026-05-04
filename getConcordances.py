"""
fetch_jskos.py
==============
Downloads mappings, vocabularies (concept schemes), and concordances
from the coli-conc JSKOS API and saves them locally for GNN/transformer training.

Outputs
-------
  data/
    mappings.ndjson       — all ~1.1M mappings, one JSON object per line
    vocabularies.json     — all ~113 concept schemes
    concordances.json     — all ~113 concordances
    concepts/<voc_uri_slug>.ndjson  — concepts per vocabulary (optional, can be large)
"""

import requests
import json
import os
import re
import time
from pathlib import Path

BASE_URL = "https://coli-conc.gbv.de/api"
DATA_DIR = Path("data")
CONCEPTS_DIR = DATA_DIR / "concepts"

DATA_DIR.mkdir(exist_ok=True)
CONCEPTS_DIR.mkdir(exist_ok=True)

session = requests.Session()
session.headers.update({"Accept": "application/json"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(uri: str) -> str:
    """Turn a URI into a safe filename slug."""
    return re.sub(r"[^\w\-]", "_", uri)[:80]


def fetch_json(url: str, params: dict = None, retries: int = 3) -> list | dict:
    """Fetch JSON from URL with simple retry logic."""
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  [attempt {attempt+1}/{retries}] Error: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}")


def fetch_paginated(endpoint: str, params: dict = None, page_size: int = 500) -> list:
    """Fetch all results from a paginated endpoint."""
    params = dict(params or {})
    params["limit"] = page_size
    params["offset"] = 0
    results = []

    while True:
        data = fetch_json(f"{BASE_URL}/{endpoint}", params)
        if not data:
            break
        results.extend(data)
        print(f"  Fetched {len(results)} records from /{endpoint}...")
        if len(data) < page_size:
            break
        params["offset"] += page_size

    return results


def stream_ndjson_download(endpoint: str, out_path: Path, params: dict = None):
    """
    Use the ?download=ndjson parameter to stream large datasets without
    holding everything in memory. Writes one JSON object per line.
    """
    params = dict(params or {})
    params["download"] = "ndjson"
    url = f"{BASE_URL}/{endpoint}"

    print(f"Streaming {url} → {out_path}")
    with session.get(url, params=params, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = r.headers.get("X-Total-Count")
        print(f"  Total records reported: {total or 'unknown (download mode)'}")

        with open(out_path, "w", encoding="utf-8") as f:
            count = 0
            for line in r.iter_lines(decode_unicode=True):
                if line.strip():
                    f.write(line + "\n")
                    count += 1
                    if count % 50_000 == 0:
                        print(f"  ...{count} records written")

        print(f"  Done. {count} records saved to {out_path}")
        return count


# ---------------------------------------------------------------------------
# Step 1: Vocabularies (concept schemes)
# ---------------------------------------------------------------------------

def fetch_vocabularies():
    out = DATA_DIR / "vocabularies.json"
    if out.exists() and out.stat().st_size > 0:
        print(f"[SKIP] {out} already exists.")
        return json.loads(out.read_text(encoding="utf-8"))

    print("\n=== Fetching vocabularies ===")
    vocs = fetch_paginated("voc", page_size=500)
    out.write_text(json.dumps(vocs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved {len(vocs)} vocabularies to {out}")
    return vocs


# ---------------------------------------------------------------------------
# Step 2: Concordances
# ---------------------------------------------------------------------------

def fetch_concordances():
    out = DATA_DIR / "concordances.json"
    if out.exists() and out.stat().st_size > 0:
        print(f"[SKIP] {out} already exists.")
        return json.loads(out.read_text(encoding="utf-8"))

    print("\n=== Fetching concordances ===")
    conc = fetch_paginated("concordances", page_size=200)
    out.write_text(json.dumps(conc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved {len(conc)} concordances to {out}")
    return conc


# ---------------------------------------------------------------------------
# Step 3: Mappings (large — streamed as NDJSON)
# ---------------------------------------------------------------------------

def fetch_mappings():
    out = DATA_DIR / "mappings.ndjson"
    if out.exists():
        print(f"[SKIP] {out} already exists.")
        return

    print("\n=== Fetching mappings (streaming NDJSON) ===")
    print("  This may take several minutes for ~1.1M records...")
    stream_ndjson_download("mappings", out)


# ---------------------------------------------------------------------------
# Step 4: Concepts per vocabulary (optional, can be large)
# ---------------------------------------------------------------------------

def fetch_concepts_for_voc(voc: dict):
    uri = voc.get("uri", "")
    if not uri:
        return
    slug = slugify(uri)
    out = CONCEPTS_DIR / f"{slug}.ndjson"
    if out.exists():
        return

    try:
        stream_ndjson_download("voc/concepts", out, params={"uri": uri})
    except Exception as e:
        print(f"  [WARN] Could not fetch concepts for {uri}: {e}")
        out.unlink(missing_ok=True)


def fetch_all_concepts(vocs: list):
    print(f"\n=== Fetching concepts for {len(vocs)} vocabularies ===")
    for i, voc in enumerate(vocs):
        label = (voc.get("prefLabel") or {}).get("en") or voc.get("uri", "?")
        print(f"[{i+1}/{len(vocs)}] {label}")
        fetch_concepts_for_voc(voc)


# ---------------------------------------------------------------------------
# Step 5: Build a simple edge list for graph construction
# ---------------------------------------------------------------------------

def build_edge_list():
    """
    Parse mappings.ndjson and produce a compact TSV edge list:
      from_uri  to_uri  mapping_type  concordance_uri

    Suitable for loading into torch_geometric, DGL, or networkx.
    """
    src = DATA_DIR / "mappings.ndjson"
    out = DATA_DIR / "edge_list.tsv"

    if out.exists():
        print(f"[SKIP] {out} already exists.")
        return

    if not src.exists():
        print("[SKIP] mappings.ndjson not found, run fetch_mappings() first.")
        return

    print("\n=== Building edge list from mappings ===")
    count = 0
    skipped = 0

    with open(src, encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
        f_out.write("from_uri\tto_uri\tmapping_type\tconcordance_uri\n")
        for line in f_in:
            try:
                m = json.loads(line)
                from_members = m.get("from", {}).get("memberSet", [])
                to_members = m.get("to", {}).get("memberSet", [])
                mtype = (m.get("type") or [""])[0].split("#")[-1]  # e.g. exactMatch
                concordance = (m.get("partOf") or [{}])[0].get("uri", "")

                # Expand many-to-many into pairs (most are 1-to-1 or 1-to-n)
                for fc in from_members:
                    for tc in to_members:
                        fu = fc.get("uri", "")
                        tu = tc.get("uri", "")
                        if fu and tu:
                            f_out.write(f"{fu}\t{tu}\t{mtype}\t{concordance}\n")
                            count += 1
            except Exception:
                skipped += 1

    print(f"  {count} edges written, {skipped} lines skipped.")
    print(f"  Saved to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    vocs = fetch_vocabularies()
    concordances = fetch_concordances()
    fetch_mappings()
    build_edge_list()

    # Uncomment to also fetch all concept labels/notations (takes longer):
    # fetch_all_concepts(vocs)