"""
reports/statistics.py — Satellite population statistics.

Summarizes a catalog snapshot (from core.tle_manager) into the kind of
numbers a space-situational-awareness report opens with: how many
objects, in which altitude regime, at which inclinations, split by
classification. Reuses stm.debris_map's altitude/inclination histogram
machinery rather than re-implementing binning logic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from core.constants import R_EARTH
from core.tle_manager import Satellite
from stm.debris_map import build_density_histogram

# Standard orbital regime bands (altitude above Earth's surface, km).
REGIME_BANDS = {
    "LEO": (0.0, 2000.0),
    "MEO": (2000.0, 35586.0),
    "GEO": (35586.0, 35986.0),
    "HEO": (35986.0, float("inf")),
}


@dataclass
class PopulationStatistics:
    """Summary statistics for a satellite catalog snapshot."""

    total_count: int
    count_by_classification: dict[str, int]
    count_by_regime: dict[str, int]
    inclination_mean_deg: float
    inclination_std_deg: float
    eccentricity_mean: float
    altitude_min_km: float
    altitude_max_km: float
    altitude_median_km: float


def _mean_altitude_km(sat: Satellite) -> float:
    """Circular-equivalent altitude: (perigee + apogee) / 2."""
    perigee = sat.semi_major_axis_km * (1 - sat.eccentricity) - R_EARTH
    apogee = sat.semi_major_axis_km * (1 + sat.eccentricity) - R_EARTH
    return (perigee + apogee) / 2.0


def classify_regime(altitude_km: float) -> str:
    """Map an altitude to a named orbital regime band."""
    for name, (lo, hi) in REGIME_BANDS.items():
        if lo <= altitude_km < hi:
            return name
    return "unknown"


def compute_population_statistics(satellites: list[Satellite]) -> PopulationStatistics:
    """
    Compute summary statistics over a satellite list.

    Args:
        satellites: catalog objects (any mix of classifications).

    Returns:
        PopulationStatistics. Returns all-zero/NaN-safe defaults if
        `satellites` is empty rather than raising, since "the catalog
        subset was empty" is a valid (if uninteresting) report outcome.
    """
    if not satellites:
        return PopulationStatistics(
            total_count=0, count_by_classification={}, count_by_regime={},
            inclination_mean_deg=0.0, inclination_std_deg=0.0, eccentricity_mean=0.0,
            altitude_min_km=0.0, altitude_max_km=0.0, altitude_median_km=0.0,
        )

    altitudes = np.array([_mean_altitude_km(s) for s in satellites])
    inclinations = np.array([s.inclination_deg for s in satellites])
    eccentricities = np.array([s.eccentricity for s in satellites])

    count_by_classification = dict(Counter(s.classification for s in satellites))
    count_by_regime = dict(Counter(classify_regime(a) for a in altitudes))

    return PopulationStatistics(
        total_count=len(satellites),
        count_by_classification=count_by_classification,
        count_by_regime=count_by_regime,
        inclination_mean_deg=float(inclinations.mean()),
        inclination_std_deg=float(inclinations.std()),
        eccentricity_mean=float(eccentricities.mean()),
        altitude_min_km=float(altitudes.min()),
        altitude_max_km=float(altitudes.max()),
        altitude_median_km=float(np.median(altitudes)),
    )


def altitude_inclination_histogram_summary(satellites: list[Satellite]) -> dict:
    """
    Convenience wrapper around stm.debris_map.build_density_histogram(),
    returning a JSON-serializable summary (bin edges + counts) suitable
    for embedding in an API response or report.
    """
    hist = build_density_histogram(satellites)
    return {
        "counts": hist.counts.tolist(),
        "altitude_edges_km": hist.altitude_edges_km.tolist(),
        "inclination_edges_deg": hist.inclination_edges_deg.tolist(),
    }
