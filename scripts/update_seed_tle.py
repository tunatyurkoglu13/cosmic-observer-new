"""
scripts/update_seed_tle.py — Refresh the bundled last-known-good TLE seed
files (data/seed_tle/) from live CelesTrak, for whenever it's reachable.

See data/seed_tle/README.md for why these exist: a fallback of last
resort so the app still shows real satellites with zero network access
or during a CelesTrak outage with an empty cache.

Usage:
    python scripts/update_seed_tle.py [group ...]

With no arguments, refreshes the default set (stations, visual).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from core.tle_manager import CELESTRAK_BASE, GROUPS

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed_tle"
DEFAULT_GROUPS = ["stations", "visual"]


def main() -> None:
    groups = sys.argv[1:] or DEFAULT_GROUPS

    for group in groups:
        if group not in GROUPS:
            print(f"Skipping unknown group '{group}' (valid: {list(GROUPS)})")
            continue

        url = f"{CELESTRAK_BASE}?GROUP={GROUPS[group]}&FORMAT=tle"
        print(f"Fetching {group} ({GROUPS[group]})...")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  FAILED: {e} — leaving existing seed file (if any) untouched.")
            continue

        if resp.text.lstrip().startswith(("Invalid query", "No GP data found")):
            print(f"  FAILED: CelesTrak rejected the request: {resp.text.strip()}")
            continue

        object_count = resp.text.count("\n1 ")
        SEED_DIR.mkdir(parents=True, exist_ok=True)
        (SEED_DIR / f"{group}.tle").write_text(resp.text)
        print(f"  OK — wrote {object_count} objects to data/seed_tle/{group}.tle")


if __name__ == "__main__":
    main()
