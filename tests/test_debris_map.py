from datetime import datetime, timezone

from core.tle_manager import Satellite
from stm.debris_map import (
    build_density_histogram,
    identify_critical_band_populations,
    kessler_risk_index,
)


def _make_sat(norad_id, incl, ecc, a):
    return Satellite(
        norad_id=norad_id, name=f"SAT-{norad_id}", line1="l1", line2="l2",
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        inclination_deg=incl, eccentricity=ecc,
        mean_motion_rev_per_day=15.0, semi_major_axis_km=a,
    )


def test_build_density_histogram_counts_all_objects():
    sats = [
        _make_sat(1, 51.6, 0.0001, 6793.0),   # ~415 km alt
        _make_sat(2, 98.0, 0.0001, 7178.0),   # ~800 km alt
        _make_sat(3, 0.1, 0.0001, 42164.0),   # GEO
    ]
    hist = build_density_histogram(sats)
    assert hist.counts.sum() == len(sats)


def test_identify_critical_band_populations_leo_dense():
    # semi_major_axis_km for ~900 km circular altitude: a = 900 + 6378.137
    leo_dense_sat = _make_sat(1, 98.0, 0.0001, 900.0 + 6378.137)
    other_sat = _make_sat(2, 51.6, 0.0001, 6793.0)  # ~415 km, not in LEO_dense band

    counts = identify_critical_band_populations([leo_dense_sat, other_sat])
    assert counts["LEO_dense"] == 1
    assert counts["GEO"] == 0


def test_identify_critical_band_populations_geo():
    geo_sat = _make_sat(1, 0.05, 0.0001, 42164.0)  # ~35786 km alt
    counts = identify_critical_band_populations([geo_sat])
    assert counts["GEO"] == 1


def test_identify_critical_band_populations_molniya():
    # Molniya: e ~0.74, a ~26562 km -> apogee ~ a(1+e) - R_earth ~ 39785 - 6378 ~ 33400 km
    molniya_sat = _make_sat(1, 63.4, 0.74, 26562.0)
    counts = identify_critical_band_populations([molniya_sat])
    assert counts["Molniya"] == 1


def test_identify_critical_band_populations_wrong_inclination_not_molniya():
    # Same eccentricity/apogee as a Molniya orbit but sun-synchronous-like inclination.
    not_molniya = _make_sat(1, 98.0, 0.74, 26562.0)
    counts = identify_critical_band_populations([not_molniya])
    assert counts["Molniya"] == 0


def test_kessler_risk_index_counts_per_km():
    sats = [_make_sat(i, 98.0, 0.0001, 900.0 + 6378.137) for i in range(4)]
    index = kessler_risk_index(sats, altitude_band_km=(800.0, 1000.0))
    assert index == 4 / 200.0
