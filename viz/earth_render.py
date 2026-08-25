"""
viz/earth_render.py — Subsolar point and day/night terminator computation.

Feeds the 3D dashboard's Earth shader with the information it needs to
shade the day and night hemispheres correctly: the subsolar point (where
the Sun is directly overhead) and the terminator (the great circle
separating day from night).

Uses a low-precision analytic solar position formula (good to ~0.01 deg
in ecliptic longitude — ample for a visualization, not for precision
navigation), rather than pulling in a full ephemeris (JPL Horizons) for
what is ultimately a rendering detail.

Reference: Astronomical Almanac low-precision Sun formulas (as given in
Meeus, "Astronomical Algorithms", Ch. 25, "low accuracy" variant), and
Vallado Ch. 5 for the Earth-fixed frame convention used to place the
result at an ECEF longitude.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from sgp4.api import jday

from core.constants import JD_J2000
from core.propagator import gmst_from_jd


def _datetime_to_jd(dt: datetime) -> float:
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)
    return jd + fr


def subsolar_point(dt: datetime) -> tuple[float, float]:
    """
    Compute the subsolar point (lat, lon) at a given UTC datetime: the
    point on Earth's surface directly under the Sun.

    Method (low-precision solar position, Meeus Ch. 25):
        n = JD - JD_J2000
        L = 280.460 + 0.9856474*n           (mean longitude, deg)
        g = 357.528 + 0.9856003*n           (mean anomaly, deg)
        lambda = L + 1.915*sin(g) + 0.020*sin(2g)   (ecliptic longitude, deg)
        epsilon = 23.439 - 0.0000004*n      (obliquity of the ecliptic, deg)
        RA = atan2(cos(epsilon)*sin(lambda), cos(lambda))   (right ascension)
        Dec = asin(sin(epsilon)*sin(lambda))                (declination)

    The subsolar latitude equals the Sun's declination directly. The
    subsolar longitude is where the Sun's hour angle is zero, i.e. where
    local sidereal time equals the Sun's RA:

        lon_subsolar = RA - GMST   (mod 360, wrapped to [-180, 180])

    Returns:
        (lat_deg, lon_deg) of the subsolar point.
    """
    jd = _datetime_to_jd(dt)
    n = jd - JD_J2000

    L = np.radians((280.460 + 0.9856474 * n) % 360.0)
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    epsilon = np.radians(23.439 - 0.0000004 * n)

    ra = np.arctan2(np.cos(epsilon) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(epsilon) * np.sin(lam))

    gmst_deg = np.degrees(gmst_from_jd(jd))
    lon = (np.degrees(ra) - gmst_deg + 180.0) % 360.0 - 180.0

    return float(np.degrees(dec)), float(lon)


def _latlon_to_unit_vector(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _unit_vector_to_latlon(v: np.ndarray) -> tuple[float, float]:
    lat = np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(v[1], v[0]))
    return float(lat), float(lon)


def terminator_points(dt: datetime, n_points: int = 180) -> list[tuple[float, float]]:
    """
    Compute points along the day/night terminator: the great circle 90
    degrees away from the subsolar point (where the Sun sits exactly on
    the horizon).

    Method: build an orthonormal basis {u, v} spanning the plane
    perpendicular to the subsolar direction s, then sample
    p(t) = cos(t)*u + sin(t)*v for t in [0, 2*pi) — every such p is, by
    construction, perpendicular to s (on the terminator).

    Returns:
        List of (lat_deg, lon_deg) tuples tracing the terminator.
    """
    lat_s, lon_s = subsolar_point(dt)
    s = _latlon_to_unit_vector(lat_s, lon_s)

    helper = np.array([0.0, 0.0, 1.0]) if abs(s[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = helper - np.dot(helper, s) * s
    u /= np.linalg.norm(u)
    v = np.cross(s, u)

    points = []
    for t in np.linspace(0, 2 * np.pi, n_points, endpoint=False):
        p = np.cos(t) * u + np.sin(t) * v
        points.append(_unit_vector_to_latlon(p))
    return points


def is_daylight(lat_deg: float, lon_deg: float, dt: datetime) -> bool:
    """True if the given surface point currently faces the Sun (subsolar point within 90 deg)."""
    lat_s, lon_s = subsolar_point(dt)
    s = _latlon_to_unit_vector(lat_s, lon_s)
    p = _latlon_to_unit_vector(lat_deg, lon_deg)
    return float(np.dot(s, p)) > 0.0
