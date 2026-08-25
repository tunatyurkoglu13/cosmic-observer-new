"""
stm/debris_map.py — Orbital debris density mapping and Kessler-risk indicators.

Builds a 2D (altitude x inclination) density histogram over a catalog of
objects — the standard way to visualize where debris clusters (LEO
sun-synchronous bands ~700-900 km, the "graveyard" bands, GEO, Molniya-type
highly-elliptical orbits) — and flags a few well-known critical shells by
name.

"Kessler syndrome" (Kessler & Cour-Palais, 1978) is the runaway-collision
scenario where debris density in a band becomes high enough that
collisions generate debris faster than natural decay removes it. We don't
attempt to model the full collisional cascade here (that requires a
population/flux model well beyond catalog data); instead we expose a
simple, transparent proxy — object count per km of altitude shell — that
a caller can track over time (by calling this on repeated catalog
snapshots) to watch for the trend that matters operationally: is density
in a critical band increasing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.tle_manager import Satellite

R_EARTH_KM = 6378.137

# Well-known critical altitude/regime bands, per project spec.
CRITICAL_BANDS = {
    "LEO_dense": (800.0, 1000.0),      # heavily used sun-synchronous / Earth-observation band
    "GEO": (35586.0, 35986.0),         # +/-200 km around the 35,786 km geostationary altitude
}

MOLNIYA_INCLINATION_DEG = 63.4
MOLNIYA_INCLINATION_TOLERANCE_DEG = 2.0
MOLNIYA_MIN_APOGEE_KM = 30000.0


@dataclass
class DebrisHistogram:
    """A 2D altitude x inclination density histogram."""

    counts: np.ndarray            # shape (n_alt_bins, n_inc_bins)
    altitude_edges_km: np.ndarray  # length n_alt_bins + 1
    inclination_edges_deg: np.ndarray  # length n_inc_bins + 1


def _mean_altitude_km(sat: Satellite) -> float:
    """Approximate circular-equivalent altitude: (perigee + apogee) / 2, geodetic-ish."""
    perigee = sat.semi_major_axis_km * (1 - sat.eccentricity) - R_EARTH_KM
    apogee = sat.semi_major_axis_km * (1 + sat.eccentricity) - R_EARTH_KM
    return (perigee + apogee) / 2.0


def build_density_histogram(
    satellites: list[Satellite],
    altitude_bins_km: np.ndarray | None = None,
    inclination_bins_deg: np.ndarray | None = None,
) -> DebrisHistogram:
    """
    Build a 2D (altitude x inclination) object-count histogram.

    Args:
        satellites: catalog objects (typically the "debris" TLE group,
            but works for any classification).
        altitude_bins_km: bin edges; defaults to 0-40,000 km in 500 km
            steps (covers LEO through GEO).
        inclination_bins_deg: bin edges; defaults to 0-180 deg in 5 deg steps.

    Returns:
        DebrisHistogram with counts and the bin edges used.
    """
    if altitude_bins_km is None:
        altitude_bins_km = np.arange(0, 40_001, 500)
    if inclination_bins_deg is None:
        inclination_bins_deg = np.arange(0, 181, 5)

    altitudes = np.array([_mean_altitude_km(s) for s in satellites])
    inclinations = np.array([s.inclination_deg for s in satellites])

    counts, alt_edges, inc_edges = np.histogram2d(
        altitudes, inclinations, bins=[altitude_bins_km, inclination_bins_deg]
    )

    return DebrisHistogram(counts=counts, altitude_edges_km=alt_edges, inclination_edges_deg=inc_edges)


def identify_critical_band_populations(satellites: list[Satellite]) -> dict[str, int]:
    """
    Count objects falling in each well-known critical regime:
      - LEO_dense: mean altitude in [800, 1000] km
      - GEO: mean altitude in [35586, 35986] km
      - Molniya: apogee > 30,000 km and inclination within 2 deg of the
        63.4 deg critical inclination (where the argument-of-perigee
        drift from J2 vanishes — see core.propagator.j2_secular_rates —
        which is exactly why Molniya-type orbits are designed at this
        inclination in the first place).
    """
    counts = {"LEO_dense": 0, "GEO": 0, "Molniya": 0}

    for sat in satellites:
        alt = _mean_altitude_km(sat)
        lo, hi = CRITICAL_BANDS["LEO_dense"]
        if lo <= alt <= hi:
            counts["LEO_dense"] += 1

        lo, hi = CRITICAL_BANDS["GEO"]
        if lo <= alt <= hi:
            counts["GEO"] += 1

        apogee = sat.semi_major_axis_km * (1 + sat.eccentricity) - R_EARTH_KM
        if apogee > MOLNIYA_MIN_APOGEE_KM and abs(sat.inclination_deg - MOLNIYA_INCLINATION_DEG) <= MOLNIYA_INCLINATION_TOLERANCE_DEG:
            counts["Molniya"] += 1

    return counts


def kessler_risk_index(satellites: list[Satellite], altitude_band_km: tuple[float, float] = (800.0, 1000.0)) -> float:
    """
    Simple debris-density proxy for a given altitude band: object count
    per km of shell thickness. This is NOT a physically calibrated
    collision-cascade risk model — it's a transparent, trackable number
    intended to be compared across repeated catalog snapshots over time
    (rising count/km in a critical band is the actionable signal, not the
    absolute value in isolation).

    Args:
        satellites: catalog objects to count (e.g. "debris" + "active" combined).
        altitude_band_km: (low, high) altitude bounds of the shell.

    Returns:
        Objects per km of altitude within the band.
    """
    lo, hi = altitude_band_km
    thickness = hi - lo
    if thickness <= 0:
        raise ValueError("altitude_band_km upper bound must exceed lower bound")

    count = sum(1 for s in satellites if lo <= _mean_altitude_km(s) <= hi)
    return count / thickness
