"""
data/space_weather.py — Space weather feeds and an atmospheric-density
model driven by them.

Space weather matters to this platform for two reasons:
  1. Geomagnetic/solar activity (Kp index, F10.7 flux) inflates Earth's
     upper atmosphere, which increases drag on LEO objects and therefore
     the uncertainty in their propagated position — directly feeding the
     covariance used in collision.py's Pc calculation.
  2. Geomagnetic storms are operationally relevant to launch/reentry
     planning and can trigger NASA DONKI notifications.

Data sources:
  - NOAA SWPC (no key required):
      Planetary Kp index (3-hourly):
        https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json
      Planetary Kp index (1-minute estimate):
        https://services.swpc.noaa.gov/json/planetary_k_index_1m.json
      F10.7 cm solar radio flux (observation history):
        https://services.swpc.noaa.gov/json/f107_cm_flux.json
      F10.7 cm flux (latest single value):
        https://services.swpc.noaa.gov/products/summary/10cm-flux.json
  - NASA DONKI (requires NASA_API_KEY, from api.nasa.gov):
      https://api.nasa.gov/DONKI/{endpoint}

Atmospheric density model:
  We use the simple exponential (single-scale-height) model rather than
  the full NRLMSISE-00 empirical atmosphere. NRLMSISE-00 is far more
  accurate (it's a function of altitude, latitude, local time, F10.7, and
  Kp simultaneously) but requires a dedicated implementation/binding; the
  exponential model captures the first-order effect we need for
  educational drag estimates and coarse orbit-decay/covariance-inflation
  purposes, using tabulated scale heights that already vary by altitude
  band (Vallado, Table 8-4).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import requests

SWPC_KP_3H_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
SWPC_KP_1M_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
SWPC_F107_HISTORY_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
SWPC_F107_LATEST_URL = "https://services.swpc.noaa.gov/products/summary/10cm-flux.json"
DONKI_BASE_URL = "https://api.nasa.gov/DONKI"

# Exponential atmosphere reference table: (base altitude km, nominal density kg/m^3, scale height km)
# Source: Vallado, "Fundamentals of Astrodynamics and Applications" (4th Ed), Table 8-4.
_EXPONENTIAL_ATMOSPHERE_TABLE = [
    (0, 1.225, 7.249),
    (25, 3.899e-2, 6.349),
    (30, 1.774e-2, 6.682),
    (40, 3.972e-3, 7.554),
    (50, 1.057e-3, 8.382),
    (60, 3.206e-4, 7.714),
    (70, 8.770e-5, 6.549),
    (80, 1.905e-5, 5.799),
    (90, 3.396e-6, 5.382),
    (100, 5.297e-7, 5.877),
    (110, 9.661e-8, 7.263),
    (120, 2.438e-8, 9.473),
    (130, 8.484e-9, 12.636),
    (140, 3.845e-9, 16.149),
    (150, 2.070e-9, 22.523),
    (180, 5.464e-10, 29.740),
    (200, 2.789e-10, 37.105),
    (250, 7.248e-11, 45.546),
    (300, 2.418e-11, 53.628),
    (350, 9.518e-12, 53.298),
    (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828),
    (500, 6.967e-13, 63.822),
    (600, 1.454e-13, 71.835),
    (700, 3.614e-14, 88.667),
    (800, 1.170e-14, 124.64),
    (900, 5.245e-15, 181.05),
    (1000, 3.019e-15, 268.00),
]


@dataclass
class SpaceWeatherSnapshot:
    """A single point-in-time space weather reading used for drag/covariance modeling."""

    kp_index: float
    f107_flux: float


def fetch_latest_kp(use_1min: bool = True) -> float:
    """
    Fetch the most recent planetary Kp index.

    Kp ranges 0-9 and measures global geomagnetic disturbance; Kp >= 5
    corresponds to a geomagnetic storm (G1+ on NOAA's scale), which
    measurably inflates thermospheric density within hours.

    Args:
        use_1min: if True, use the 1-minute estimated-Kp feed (freshest,
            typically < 5 min old); if False, use the official 3-hourly
            planetary Kp product (authoritative but coarser cadence).
    """
    url = SWPC_KP_1M_URL if use_1min else SWPC_KP_3H_URL
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    records = resp.json()
    if not records:
        raise RuntimeError(f"Empty Kp response from {url}")

    latest = records[-1]
    return float(latest.get("estimated_kp") or latest.get("Kp"))


def fetch_latest_f107() -> float:
    """
    Fetch the latest observed F10.7 cm solar radio flux [solar flux
    units, sfu]. F10.7 is the standard proxy for solar EUV output driving
    thermospheric heating (and thus density) on day-to-week timescales,
    used because EUV itself is hard to measure continuously from the
    ground.
    """
    resp = requests.get(SWPC_F107_LATEST_URL, timeout=15)
    resp.raise_for_status()
    records = resp.json()
    if not records:
        raise RuntimeError("Empty F10.7 response from SWPC")
    return float(records[-1]["flux"])


def fetch_current_snapshot() -> SpaceWeatherSnapshot:
    """Convenience: fetch both Kp and F10.7 in one call for feeding the density model."""
    return SpaceWeatherSnapshot(kp_index=fetch_latest_kp(), f107_flux=fetch_latest_f107())


def fetch_donki_notifications(start_date: str, end_date: str, api_key: str | None = None) -> list[dict]:
    """
    Fetch NASA DONKI space weather notifications (solar flares, CMEs,
    geomagnetic storms, etc.) over a date range.

    Args:
        start_date, end_date: "YYYY-MM-DD" strings.
        api_key: NASA API key; falls back to the NASA_API_KEY environment
            variable, then to "DEMO_KEY" (heavily rate-limited).
    """
    key = api_key or os.environ.get("NASA_API_KEY", "DEMO_KEY")
    params = {"startDate": start_date, "endDate": end_date, "api_key": key}
    resp = requests.get(f"{DONKI_BASE_URL}/notifications", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_donki_geomagnetic_storms(start_date: str, end_date: str, api_key: str | None = None) -> list[dict]:
    """Fetch NASA DONKI Geomagnetic Storm (GST) events over a date range."""
    key = api_key or os.environ.get("NASA_API_KEY", "DEMO_KEY")
    params = {"startDate": start_date, "endDate": end_date, "api_key": key}
    resp = requests.get(f"{DONKI_BASE_URL}/GST", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def atmospheric_density(altitude_km: float, kp_index: float | None = None) -> float:
    """
    Exponential-atmosphere density model:

        rho = rho_0 * exp(-(h - h_0) / H)

    using the altitude band's tabulated (rho_0, H) from
    _EXPONENTIAL_ATMOSPHERE_TABLE (Vallado Table 8-4), which already bakes
    in the altitude-dependence of scale height H (H itself grows with
    altitude because the thermosphere isn't isothermal).

    A simple empirical Kp correction is applied on top: storm-time
    heating measurably inflates density above ~200 km, roughly
    proportional to (1 + Kp/9) in this simplified treatment (a crude
    stand-in for the exospheric-temperature dependence NRLMSISE-00 models
    properly — good enough to flag "expect more drag than nominal" but
    not for precision orbit determination).

    Args:
        altitude_km: geodetic altitude above the ellipsoid [km].
        kp_index: optional current Kp (0-9); if given and altitude > 200
            km, density is scaled up by (1 + kp_index/9).

    Returns:
        Atmospheric density [kg/m^3].
    """
    if altitude_km < 0:
        raise ValueError("altitude_km must be non-negative")

    # Find the highest table entry with base altitude <= altitude_km.
    band = _EXPONENTIAL_ATMOSPHERE_TABLE[0]
    for entry in _EXPONENTIAL_ATMOSPHERE_TABLE:
        if entry[0] <= altitude_km:
            band = entry
        else:
            break

    h0, rho0, H = band
    rho = rho0 * np.exp(-(altitude_km - h0) / H)

    if kp_index is not None and altitude_km > 200:
        rho *= 1.0 + kp_index / 9.0

    return float(rho)


def drag_acceleration(
    velocity_km_s: np.ndarray,
    altitude_km: float,
    drag_coefficient: float,
    area_m2: float,
    mass_kg: float,
    kp_index: float | None = None,
) -> np.ndarray:
    """
    Atmospheric drag acceleration:

        a_drag = -(1/2) * rho * v^2 * (Cd * A / m) * v_hat

    Args:
        velocity_km_s: relative-to-atmosphere velocity vector [km/s]
            (strictly this should be inertial velocity minus the
            atmosphere's corotation velocity omega_E x r, not raw
            inertial velocity — callers doing precision work should pass
            that corrected vector; for typical LEO objects the
            corotation term is a ~5-10% correction to |v|).
        altitude_km: current altitude, feeds atmospheric_density().
        drag_coefficient: dimensionless Cd, typically 2.0-2.2 for
            tumbling satellites/debris.
        area_m2: cross-sectional area exposed to the flow [m^2].
        mass_kg: object mass [kg].
        kp_index: optional, passed through to atmospheric_density().

    Returns:
        Drag acceleration vector [km/s^2], same direction convention as
        velocity_km_s (i.e. opposing it).
    """
    rho_kg_m3 = atmospheric_density(altitude_km, kp_index=kp_index)
    # Convert density to kg/km^3 so the whole expression stays in km/s units.
    rho_kg_km3 = rho_kg_m3 * 1e9

    v_mag = np.linalg.norm(velocity_km_s)
    if v_mag == 0:
        return np.zeros(3)
    v_hat = velocity_km_s / v_mag

    ballistic_coefficient = drag_coefficient * (area_m2 * 1e-6) / mass_kg  # km^2/kg
    a_mag = 0.5 * rho_kg_km3 * v_mag**2 * ballistic_coefficient

    return -a_mag * v_hat
