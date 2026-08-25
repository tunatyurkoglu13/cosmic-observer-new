"""
scripts/generate_dashboard_snapshot.py — CLI: build a real snapshot.json
for the static dashboard from live CelesTrak TLE data.

Usage:
    python scripts/generate_dashboard_snapshot.py [--active-limit N] [--frames N] [--step-hours H]

Pulls "stations" (all) and "active" (first N, since propagating the full
~16,000-object active catalog for a multi-frame snapshot is a Phase 7
(WebSocket real-time propagation) concern, not this static-preview
snapshot's job) plus "debris" (first N//4), builds a DashboardSnapshot,
and writes it to static/snapshot.json alongside the retro theme CSS.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tle_manager import TLEManager
from viz.dashboard_3d import add_ground_tracks, build_snapshot, write_snapshot_json
from viz.retro_ui import generate_crt_flicker_css, generate_holo_panel_css, generate_scanline_css, generate_theme_css

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--active-group", default="visual",
        help="CelesTrak group to sample 'active'-like satellites from (default: 'visual', "
             "the ~100 brightest active satellites — the full 'active' group is ~16,000 objects "
             "and is intended for Phase 7's real-time WebSocket propagation, not this static preview)",
    )
    parser.add_argument("--active-limit", type=int, default=300, help="Max objects from --active-group to include")
    parser.add_argument("--debris-limit", type=int, default=75, help="Max 'debris' objects to include")
    parser.add_argument("--frames", type=int, default=24, help="Number of time frames")
    parser.add_argument("--step-hours", type=float, default=1.0, help="Hours between frames")
    args = parser.parse_args()

    mgr = TLEManager()
    stations = mgr.fetch_group("stations")
    active = mgr.fetch_group(args.active_group)[: args.active_limit]
    debris = mgr.fetch_group("debris")[: args.debris_limit]

    all_sats = stations + active + debris
    print(f"Building snapshot for {len(all_sats)} objects "
          f"({len(stations)} stations, {len(active)} active, {len(debris)} debris)...")

    epoch = datetime.now(timezone.utc)
    snapshot = build_snapshot(
        all_sats, epoch, n_frames=args.frames, frame_step=timedelta(hours=args.step_hours)
    )
    print(f"Propagated {len(snapshot.satellites)} objects successfully across {args.frames} frames.")

    add_ground_tracks(snapshot, stations, epoch, step=timedelta(minutes=1))
    print(f"Computed orbit ground tracks for {len(snapshot.ground_tracks)} station-class objects.")

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    write_snapshot_json(snapshot, STATIC_DIR / "snapshot.json")
    print(f"Wrote {STATIC_DIR / 'snapshot.json'}")

    css = "\n\n".join([
        generate_theme_css(),
        generate_scanline_css(),
        generate_crt_flicker_css(),
        generate_holo_panel_css(),
    ])
    (STATIC_DIR / "retro.css").write_text(css)
    print(f"Wrote {STATIC_DIR / 'retro.css'}")


if __name__ == "__main__":
    main()
