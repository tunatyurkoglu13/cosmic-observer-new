"""
core/rocket_trajectory.py — Idealized launch-ascent trajectory model.

Real launch providers (SpaceX, NASA/ULA, etc.) run COLA (Collision On
Launch Assessment) against a *reference* ascent trajectory — a planned,
idealized flight path with safety margins added around it — not the
literal real-time guided path, since the actual GNC (guidance,
navigation & control) trajectory depends on real-time wind, vehicle
performance, and other factors not known days in advance when a launch
window is first cleared. This module builds that same kind of reference
trajectory: a smooth, physically-reasonable ascent profile from a launch
site to orbital insertion, good enough to screen against the tracked
catalog for close approaches, not a substitute for a real GNC/6-DOF
trajectory simulation.

Ascent profile (idealized, per this module):
  - Downrange ground distance grows ~linearly with time (roughly matches
    a typical powered ascent's horizontal progress).
  - Altitude follows a smoothstep S-curve from 0 to the target insertion
    altitude: slow initial climb (still mostly fighting gravity/drag
    near the pad), a faster middle climb, and a leveling-off approaching
    insertion (matching the qualitative shape of a real ascent-altitude
    profile) — not a rigorous gravity-turn integration.
  - The ground track follows a great-circle path from the launch site at
    the given launch azimuth (core.launch_window.launch_azimuth), using
    the standard "destination point given start point, bearing and
    distance" spherical-Earth formula (Vallado / standard geodesy).

References: Vallado, "Fundamentals of Astrodynamics and Applications"
(4th Ed), Ch. 11 (launch site operations); the destination-point formula
is the standard great-circle "direct geodetic problem" on a sphere.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from core.constants import R_EARTH
from core.launch_window import LaunchSite
from core.propagator import ecef_to_teme, geodetic_to_ecef, gmst_from_jd, jd_fr_array_from_datetimes


@dataclass
class TrajectorySample:
    """One instant along an ascent trajectory."""

    t_offset_s: float          # seconds since launch
    absolute_time: datetime    # UTC epoch of this sample
    lat_deg: float
    lon_deg: float
    alt_km: float
    r_teme: np.ndarray         # position in TEME (inertial) frame [km] — comparable to satellite positions


def _destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    """
    Great-circle "destination point" formula: given a start point, an
    initial bearing, and a distance along the sphere's surface, find the
    resulting (lat, lon).

        lat2 = asin(sin(lat1)*cos(d/R) + cos(lat1)*sin(d/R)*cos(brng))
        lon2 = lon1 + atan2(sin(brng)*sin(d/R)*cos(lat1),
                             cos(d/R) - sin(lat1)*sin(lat2))

    where d/R is the angular distance traveled (distance / Earth radius).
    """
    lat1 = np.radians(lat_deg)
    lon1 = np.radians(lon_deg)
    brng = np.radians(bearing_deg)
    delta = distance_km / R_EARTH

    lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) + np.cos(lat1) * np.sin(delta) * np.cos(brng))
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * np.sin(delta) * np.cos(lat1),
        np.cos(delta) - np.sin(lat1) * np.sin(lat2),
    )

    lon2_deg = (np.degrees(lon2) + 180.0) % 360.0 - 180.0
    return float(np.degrees(lat2)), lon2_deg


def _smoothstep(frac: np.ndarray | float) -> np.ndarray | float:
    """Standard smoothstep S-curve: 3x^2 - 2x^3, 0 at x=0, 1 at x=1, zero slope at both ends."""
    return 3 * frac**2 - 2 * frac**3


def compute_ascent_trajectory(
    site: LaunchSite,
    azimuth_deg: float,
    launch_time: datetime,
    target_altitude_km: float,
    ascent_duration_s: float = 600.0,
    max_downrange_km: float = 2000.0,
    n_samples: int = 41,
) -> list[TrajectorySample]:
    """
    Build an idealized ascent trajectory from a launch site to orbital
    insertion altitude.

    Args:
        site: launch site (gives the starting lat/lon).
        azimuth_deg: launch azimuth (from core.launch_window.launch_azimuth()).
        launch_time: UTC liftoff time.
        target_altitude_km: insertion altitude to climb to.
        ascent_duration_s: total powered-ascent time to insertion, in
            seconds. 600 s (10 min) is a reasonable default for a LEO
            insertion (real vehicles vary roughly 8-12 minutes).
        max_downrange_km: total great-circle ground distance covered by
            insertion. ~2000 km is a reasonable default order of
            magnitude for a LEO ascent; adjust for the mission profile
            being screened.
        n_samples: number of trajectory samples (including t=0 and t=ascent_duration_s).

    Returns:
        List of TrajectorySample, ascending in time, each with position
        already converted into the TEME inertial frame so it can be
        directly distance-compared against SGP4-propagated satellite
        positions (see stm/cola.py).
    """
    t_offsets = np.linspace(0.0, ascent_duration_s, n_samples)
    frac = t_offsets / ascent_duration_s

    altitudes = target_altitude_km * _smoothstep(frac)
    downranges = max_downrange_km * frac

    absolute_times = [launch_time + timedelta(seconds=float(t)) for t in t_offsets]
    jd_array, fr_array = jd_fr_array_from_datetimes(absolute_times)

    samples = []
    for i in range(n_samples):
        lat, lon = _destination_point(site.lat_deg, site.lon_deg, azimuth_deg, downranges[i])
        alt = float(altitudes[i])

        r_ecef = geodetic_to_ecef(lat, lon, alt)
        gmst = gmst_from_jd(jd_array[i] + fr_array[i])
        r_teme = ecef_to_teme(r_ecef, gmst)

        samples.append(
            TrajectorySample(
                t_offset_s=float(t_offsets[i]),
                absolute_time=absolute_times[i],
                lat_deg=lat, lon_deg=lon, alt_km=alt,
                r_teme=r_teme,
            )
        )

    return samples
