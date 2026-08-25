"""
stm/conjunction.py — Catalog-wide conjunction screening (SOCRATES-style).

Mirrors the funnel used by operational tools like CelesTrak's SOCRATES
(Satellite Orbital Conjunction Reports Assessing Threatening Encounters in
Space) and NASA CARA, adapted to run on top of core/propagator.py and
core/tle_manager.py:

    Stage 1 (cheap, O(n)):      altitude-band overlap filter
    Stage 2 (moderate, O(k)):   batch-propagate survivors over the search
                                window once, vectorized coarse min-distance
                                scan over all candidate pairs
    Stage 3 (expensive, O(m)):  refine TCA to sub-second precision (reusing
                                core.collision.find_time_of_closest_approach)
                                only for the pairs that passed Stage 2

This keeps the expensive per-pair refinement step limited to genuinely
close approaches instead of running it n(n-1)/2 times, which is what makes
catalog-scale screening (thousands of objects) tractable.

Alert thresholds (project convention, not a single universal standard —
operational screening centers vary these; these bands are a reasonable
default for LEO screening):
    < 10 km   -> emergency
    < 25 km   -> high
    < 50 km   -> medium
    < 100 km  -> info
    >= 100 km -> not reported
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from core.collision import find_time_of_closest_approach
from core.propagator import Propagator
from core.tle_manager import Satellite

EMERGENCY_KM = 10.0
HIGH_KM = 25.0
MEDIUM_KM = 50.0
INFO_KM = 100.0

ALERT_LEVELS = ("emergency", "high", "medium", "info")


def classify_alert(miss_distance_km: float) -> str | None:
    """Map a miss distance to an alert level, or None if below the reporting threshold."""
    if miss_distance_km < EMERGENCY_KM:
        return "emergency"
    if miss_distance_km < HIGH_KM:
        return "high"
    if miss_distance_km < MEDIUM_KM:
        return "medium"
    if miss_distance_km < INFO_KM:
        return "info"
    return None


@dataclass
class ConjunctionSummary:
    """One flagged close approach between two catalog objects."""

    sat1: Satellite
    sat2: Satellite
    tca: datetime
    miss_distance_km: float
    relative_speed_km_s: float
    alert_level: str


def _perigee_apogee_km(sat: Satellite, r_earth: float = 6378.137) -> tuple[float, float]:
    """Geodetic-ish perigee/apogee altitude from (a, e) — used only for the coarse band filter."""
    perigee = sat.semi_major_axis_km * (1 - sat.eccentricity) - r_earth
    apogee = sat.semi_major_axis_km * (1 + sat.eccentricity) - r_earth
    return perigee, apogee


def screen_altitude_bands(satellites: list[Satellite], margin_km: float = 50.0) -> list[tuple[int, int]]:
    """
    Stage 1: return index pairs (into `satellites`) whose perigee/apogee
    altitude bands overlap (expanded by `margin_km`). Objects in disjoint
    altitude bands (e.g. a LEO object and a GEO object) can never have a
    close approach, so this prunes the O(n^2) pair search cheaply before
    any propagation happens.

    Bands are sorted by perigee first so the pairwise comparison is O(n
    log n + k) instead of O(n^2) — once band_i's lower edge exceeds
    band_j's upper edge for all subsequent j, we can stop scanning ahead
    for that i.
    """
    bands = [(_perigee_apogee_km(s), i) for i, s in enumerate(satellites)]
    bands = [((lo - margin_km, hi + margin_km), i) for (lo, hi), i in bands]
    bands.sort(key=lambda b: b[0][0])

    pairs = []
    n = len(bands)
    for a in range(n):
        (lo_a, hi_a), idx_a = bands[a]
        for b in range(a + 1, n):
            (lo_b, hi_b), idx_b = bands[b]
            if lo_b > hi_a:
                break  # sorted by lo -> no further b can overlap with a
            if lo_a <= hi_b and lo_b <= hi_a:
                pairs.append((idx_a, idx_b))
    return pairs


def _batch_propagate(satellites: list[Satellite], sample_times: list[datetime]) -> np.ndarray:
    """
    Propagate every satellite to every sample time once, returning an
    array of shape (n_satellites, n_times, 3) of TEME positions [km].

    Objects for which SGP4 errors out (e.g. decayed orbit) at any sample
    time are given NaN positions for that time so they simply never win a
    pairwise minimum, rather than crashing the whole batch.
    """
    n_sats = len(satellites)
    n_times = len(sample_times)
    positions = np.full((n_sats, n_times, 3), np.nan)

    for i, sat in enumerate(satellites):
        prop = Propagator(sat.line1, sat.line2, name=sat.name)
        for j, t in enumerate(sample_times):
            try:
                state = prop.propagate(t)
                positions[i, j] = state.r_teme
            except RuntimeError:
                continue
    return positions


def screen_catalog(
    satellites: list[Satellite],
    search_start: datetime,
    search_end: datetime,
    coarse_step: timedelta = timedelta(minutes=2),
    band_margin_km: float = 50.0,
    refine_window: timedelta = timedelta(minutes=10),
) -> list[ConjunctionSummary]:
    """
    Full three-stage conjunction screen over a satellite list and time window.

    Args:
        satellites: catalog objects to screen (pairwise, all-vs-all).
        search_start, search_end: UTC datetime window to screen over.
        coarse_step: Stage-2 sampling interval. Must be short enough not
            to miss a close approach between samples — as a rule of thumb,
            keep it well under the smallest relevant encounter geometry's
            timescale; a few minutes is adequate for LEO-LEO screening.
        band_margin_km: Stage-1 altitude-band overlap margin.
        refine_window: half-width of the window handed to
            find_time_of_closest_approach() around each Stage-2 coarse
            minimum, for Stage-3 refinement.

    Returns:
        List of ConjunctionSummary for every pair whose refined miss
        distance is under INFO_KM, sorted by ascending miss distance
        (most concerning first).
    """
    candidate_pairs = screen_altitude_bands(satellites, margin_km=band_margin_km)
    if not candidate_pairs:
        return []

    n_samples = max(2, int((search_end - search_start) / coarse_step) + 1)
    sample_times = [search_start + i * coarse_step for i in range(n_samples)]

    positions = _batch_propagate(satellites, sample_times)

    results: list[ConjunctionSummary] = []
    for idx_a, idx_b in candidate_pairs:
        diffs = positions[idx_a] - positions[idx_b]
        distances = np.linalg.norm(diffs, axis=1)
        if np.all(np.isnan(distances)):
            continue

        min_idx = int(np.nanargmin(distances))
        coarse_min_distance = distances[min_idx]
        if coarse_min_distance >= INFO_KM:
            continue

        approx_tca = sample_times[min_idx]
        window_start = max(search_start, approx_tca - refine_window)
        window_end = min(search_end, approx_tca + refine_window)

        sat_a, sat_b = satellites[idx_a], satellites[idx_b]
        prop_a = Propagator(sat_a.line1, sat_a.line2, name=sat_a.name)
        prop_b = Propagator(sat_b.line1, sat_b.line2, name=sat_b.name)

        conj = find_time_of_closest_approach(prop_a, prop_b, window_start, window_end)
        alert_level = classify_alert(conj.miss_distance_km)
        if alert_level is None:
            continue

        results.append(
            ConjunctionSummary(
                sat1=sat_a, sat2=sat_b, tca=conj.tca,
                miss_distance_km=conj.miss_distance_km,
                relative_speed_km_s=conj.relative_speed_km_s,
                alert_level=alert_level,
            )
        )

    results.sort(key=lambda c: c.miss_distance_km)
    return results
