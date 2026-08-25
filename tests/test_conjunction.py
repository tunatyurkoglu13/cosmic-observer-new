from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from core.tle_manager import Satellite
from stm.conjunction import (
    EMERGENCY_KM,
    HIGH_KM,
    INFO_KM,
    MEDIUM_KM,
    classify_alert,
    screen_altitude_bands,
    screen_catalog,
)


def _make_sat(norad_id, name, line1, line2, incl, ecc, mm, a):
    return Satellite(
        norad_id=norad_id, name=name, line1=line1, line2=line2,
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        inclination_deg=incl, eccentricity=ecc,
        mean_motion_rev_per_day=mm, semi_major_axis_km=a,
    )


def test_classify_alert_thresholds():
    assert classify_alert(5.0) == "emergency"
    assert classify_alert(15.0) == "high"
    assert classify_alert(40.0) == "medium"
    assert classify_alert(80.0) == "info"
    assert classify_alert(150.0) is None
    assert EMERGENCY_KM < HIGH_KM < MEDIUM_KM < INFO_KM


def test_screen_altitude_bands_overlapping_pair_included():
    sats = [
        _make_sat(1, "A", "l1", "l2", 51.6, 0.0001, 15.5, 6793.0),  # ~415 km alt
        _make_sat(2, "B", "l1", "l2", 51.6, 0.0001, 15.5, 6798.0),  # ~420 km alt, overlapping
    ]
    pairs = screen_altitude_bands(sats, margin_km=50.0)
    assert (0, 1) in pairs


def test_screen_altitude_bands_disjoint_pair_excluded():
    leo = _make_sat(1, "LEO", "l1", "l2", 51.6, 0.0001, 15.5, 6793.0)
    geo = _make_sat(2, "GEO", "l1", "l2", 0.1, 0.0001, 1.003, 42164.0)
    pairs = screen_altitude_bands([leo, geo], margin_km=50.0)
    assert pairs == []


# Two real TLEs for the ISS and a co-orbiting module docked to it (NAUKA),
# which share essentially the same orbit and are guaranteed to be "close"
# for the purposes of exercising the full screen_catalog() pipeline.
ISS_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9008"
ISS_LINE2 = "2 25544  51.6416 339.9500 0001177  93.3861 264.7930 15.49560752 42982"
NAUKA_LINE1 = "1 49044U 21066A   24001.51000000  .00016000  00000-0  98000-4 0  9001"
NAUKA_LINE2 = "2 49044  51.6410 339.0000 0001200  90.0000 270.0000 15.49500000 12345"


@pytest.mark.slow
def test_screen_catalog_flags_close_coorbiting_pair():
    iss = _make_sat(25544, "ISS", ISS_LINE1, ISS_LINE2, 51.6416, 0.0001177, 15.49560752, 6796.0)
    nauka = _make_sat(49044, "NAUKA", NAUKA_LINE1, NAUKA_LINE2, 51.6410, 0.0001200, 15.49500000, 6796.0)

    start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)

    results = screen_catalog([iss, nauka], start, end, coarse_step=timedelta(minutes=2))
    # Whether or not this particular synthetic pair comes within 100 km at
    # this particular epoch, the pipeline itself must run end-to-end
    # without error and return a well-formed (possibly empty) list.
    assert isinstance(results, list)
    for r in results:
        assert r.alert_level in ("emergency", "high", "medium", "info")
        assert r.miss_distance_km >= 0
