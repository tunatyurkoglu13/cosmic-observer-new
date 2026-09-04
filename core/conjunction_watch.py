"""
core/conjunction_watch.py — Continuous real close-approach screening for
the active alert layer (see app.py's background alert-scan loop).

Different use case from stm.cola (which screens a rocket's ascent
trajectory against the catalog for launch safety): this screens a
WATCHED object already in orbit (the ISS, or any other tracked object)
against every other tracked object over a forward time window, reusing
the exact same real SGP4 propagation machinery
(core.propagator.Propagator, via stm.cola.build_catalog_grid).

Honest scope: TLEs carry no covariance, so a rigorous collision
probability (Pc) is not something this — or any TLE-only tool — can
compute (core.collision/core.collision_monte_carlo's Pc functions all
require a real covariance as input; they don't fabricate one). What
real public/professional conjunction screening actually reports at this
data-availability level is MISS DISTANCE: USSPACECOM/space-track's own
close-approach notifications flag pairs inside ~5 km, and general
catalog screening volumes run a few to some tens of km — see
THRESHOLDS_KM, whose tiers are set directly from that real convention,
not invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from core.tle_manager import Satellite
from stm.cola import build_catalog_grid

# Real, disclosed screening-volume tiers — see module docstring.
THRESHOLDS_KM = {
    "critical": 5.0,
    "warning": 10.0,
    "info": 25.0,
}


@dataclass
class CloseApproach:
    watched: Satellite
    other: Satellite
    min_distance_km: float
    time_of_closest_approach: datetime
    severity: str  # "critical" | "warning" | "info"


def _severity_for_distance(distance_km: float) -> str | None:
    if distance_km < THRESHOLDS_KM["critical"]:
        return "critical"
    if distance_km < THRESHOLDS_KM["warning"]:
        return "warning"
    if distance_km < THRESHOLDS_KM["info"]:
        return "info"
    return None


def screen_close_approaches(
    watched_satellites: list[Satellite],
    catalog_satellites: list[Satellite],
    window_start: datetime,
    window_hours: float = 24.0,
    grid_step: timedelta = timedelta(seconds=30),
) -> list[CloseApproach]:
    """
    For each satellite in `watched_satellites`, find its closest real
    predicted approach to every satellite in `catalog_satellites` over
    [window_start, window_start + window_hours], and return every pair
    whose minimum distance falls under THRESHOLDS_KM["info"] (i.e. is
    worth surfacing at all) — the "nearest grid sample" approximation is
    the same one stm.cola uses for the same reason (see its own
    docstring): fine enough at a 30 s step for this screening use case,
    not meant to replace exact per-instant propagation.
    """
    if not watched_satellites or not catalog_satellites:
        return []

    window_end = window_start + timedelta(hours=window_hours)
    all_satellites = watched_satellites + catalog_satellites
    grid = build_catalog_grid(all_satellites, window_start, window_end, grid_step=grid_step)

    n_watched = len(watched_satellites)
    events: list[CloseApproach] = []

    for wi in range(n_watched):
        watched_positions = grid.positions[wi]  # (n_grid_times, 3)
        for oi in range(n_watched, len(all_satellites)):
            other_positions = grid.positions[oi]
            diffs = other_positions - watched_positions
            distances = np.linalg.norm(diffs, axis=1)
            finite = np.isfinite(distances)
            if not finite.any():
                continue

            safe_distances = np.where(finite, distances, np.inf)
            min_idx = int(np.argmin(safe_distances))
            min_distance = float(safe_distances[min_idx])
            if not np.isfinite(min_distance):
                continue

            severity = _severity_for_distance(min_distance)
            if severity is None:
                continue

            events.append(CloseApproach(
                watched=watched_satellites[wi],
                other=catalog_satellites[oi - n_watched],
                min_distance_km=min_distance,
                time_of_closest_approach=grid.grid_times[min_idx],
                severity=severity,
            ))

    return events
