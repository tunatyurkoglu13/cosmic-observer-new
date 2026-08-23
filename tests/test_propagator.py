from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import Satrec, jday

from core.propagator import Propagator, ecef_to_geodetic, gmst_from_jd, teme_to_ecef

# Classic Spacetrack-Report-3 / Vallado SGP4 test TLE (satellite 00005,
# "Vanguard 1"): a well-known eccentric orbit (e ~ 0.163, perigee ~650 km,
# apogee ~4000 km) commonly used as an SGP4 regression fixture.
LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
LINE2 = "2 00005  34.2682 348.7242 1627203 289.2321  55.4550 10.82419157413667"


def _tle_epoch() -> datetime:
    epoch_year = 2000
    epoch_day = 179.78495062
    return datetime(epoch_year, 1, 1, tzinfo=timezone.utc) + timedelta(days=epoch_day - 1)


def test_propagator_wrapper_matches_raw_sgp4_at_epoch():
    """
    Cross-check the Propagator.propagate() datetime->JD conversion path
    against calling the underlying sgp4 library directly with its own
    jdsatepoch/jdsatepochF (i.e. tsince = 0). This validates our wrapper's
    plumbing (jday conversion, error handling, frame bookkeeping) rather
    than SGP4 itself, which is exercised via the `sgp4` package's own tests.
    """
    raw_sat = Satrec.twoline2rv(LINE1, LINE2)
    error_code, r_raw, v_raw = raw_sat.sgp4(raw_sat.jdsatepoch, raw_sat.jdsatepochF)
    assert error_code == 0

    prop = Propagator(LINE1, LINE2, name="test-sat")
    state = prop.propagate(_tle_epoch())

    assert np.allclose(state.r_teme, r_raw, atol=1e-6)
    assert np.allclose(state.v_teme, v_raw, atol=1e-6)


def test_geodetic_altitude_within_perigee_apogee_band():
    prop = Propagator(LINE1, LINE2, name="test-sat")
    state = prop.propagate(_tle_epoch())

    # Vanguard 1: e ~ 0.1627, a ~ 8617 km -> perigee ~ 7213 km, apogee ~ 10022 km
    # (geocentric radii); altitude = geocentric radius - R_EARTH (~6378 km).
    a = 8617.0
    e = 0.1627203
    r_earth = 6378.137
    perigee_alt = a * (1 - e) - r_earth
    apogee_alt = a * (1 + e) - r_earth

    assert perigee_alt - 5 < state.alt_km < apogee_alt + 5
    assert -90.0 <= state.lat_deg <= 90.0
    assert -180.0 <= state.lon_deg <= 180.0


def test_gmst_known_epoch():
    # At J2000.0 (JD 2451545.0), GMST should equal the constant term, 280.46061837 deg.
    gmst_rad = gmst_from_jd(2451545.0)
    assert np.isclose(np.degrees(gmst_rad), 280.46061837, atol=1e-6)


def test_teme_to_ecef_then_geodetic_roundtrip():
    r_teme = np.array([7022.580127, -1400.087066, 0.036577])
    gmst = gmst_from_jd(2451545.0)
    r_ecef = teme_to_ecef(r_teme, gmst)

    # Rotation must preserve vector magnitude.
    assert np.isclose(np.linalg.norm(r_ecef), np.linalg.norm(r_teme), rtol=1e-9)

    lat, lon, alt = ecef_to_geodetic(r_ecef)
    assert -90.0 <= lat <= 90.0
    assert -180.0 <= lon <= 180.0
    assert alt > 0
