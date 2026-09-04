from datetime import datetime, timezone

import pytest

from core.conjunction_watch import THRESHOLDS_KM, _severity_for_distance, screen_close_approaches
from core.tle_manager import Satellite

# A real ISS TLE (also used elsewhere in this project's tests, e.g.
# tests/test_telemetry_anomaly.py) — used here as a stand-in orbit
# template for synthetic test satellites, not to claim anything about
# the ISS's real current conjunctions.
LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9008"
LINE2 = "2 25544  51.6416 339.9500 0001177  93.3861 264.7930 15.49560752 42982"


def _make_sat(norad_id: int, name: str) -> Satellite:
    return Satellite(
        norad_id=norad_id, name=name, line1=LINE1, line2=LINE2,
        epoch=datetime(2024, 1, 1, tzinfo=timezone.utc), inclination_deg=51.6416,
        eccentricity=0.0001177, mean_motion_rev_per_day=15.4956, semi_major_axis_km=6796.0,
    )


def test_severity_thresholds_ordering():
    assert _severity_for_distance(1.0) == "critical"
    assert _severity_for_distance(7.0) == "warning"
    assert _severity_for_distance(20.0) == "info"
    assert _severity_for_distance(100.0) is None


def test_severity_thresholds_boundary_values():
    assert _severity_for_distance(THRESHOLDS_KM["critical"] - 0.01) == "critical"
    assert _severity_for_distance(THRESHOLDS_KM["critical"]) == "warning"
    assert _severity_for_distance(THRESHOLDS_KM["warning"]) == "info"
    assert _severity_for_distance(THRESHOLDS_KM["info"]) is None


def test_screen_close_approaches_empty_inputs():
    assert screen_close_approaches([], [_make_sat(1, "A")], datetime.now(timezone.utc)) == []
    assert screen_close_approaches([_make_sat(1, "A")], [], datetime.now(timezone.utc)) == []


def test_screen_close_approaches_identical_orbits_flags_critical():
    """Two satellites on the identical real orbit are always ~0 km apart — a real, checkable worst case."""
    watched = _make_sat(1, "WATCHED")
    twin = _make_sat(2, "TWIN")

    events = screen_close_approaches([watched], [twin], datetime.now(timezone.utc), window_hours=2.0)

    assert len(events) == 1
    assert events[0].severity == "critical"
    assert events[0].min_distance_km < 1.0
    assert events[0].watched.norad_id == 1
    assert events[0].other.norad_id == 2


def test_screen_close_approaches_far_apart_orbits_produce_no_events():
    """A satellite on the same orbit but 180 deg out of phase (opposite side of Earth) should not be flagged over a short window."""
    watched = _make_sat(1, "WATCHED")

    # Shift ONLY the mean anomaly field by 180 deg, keeping every other
    # orbital element (including argument of perigee) identical — the
    # same controlled-field-substitution technique used elsewhere in this
    # project's tests (see tests/test_telemetry_anomaly.py's
    # _make_synthetic_tle) to preserve the TLE's fixed-column format.
    opposite_ma_deg = (264.7930 + 180.0) % 360.0
    ma_str = f"{opposite_ma_deg:8.4f}"
    assert len(ma_str) == 8
    far_line2 = LINE2[:43] + ma_str + LINE2[51:]

    far = Satellite(
        norad_id=3, name="FAR", line1=LINE1, line2=far_line2,
        epoch=datetime(2024, 1, 1, tzinfo=timezone.utc), inclination_deg=51.6416,
        eccentricity=0.0001177, mean_motion_rev_per_day=15.4956, semi_major_axis_km=6796.0,
    )
    events = screen_close_approaches([watched], [far], datetime.now(timezone.utc), window_hours=0.1)
    # 180 deg out of phase on the same orbit = opposite sides of Earth —
    # no close approach within a short 0.1h (6 min) window.
    assert events == []


@pytest.mark.network
@pytest.mark.slow
def test_screen_close_approaches_live_real_catalog():
    """Real end-to-end check against the real, live-fetched TLE catalog — not asserting a specific outcome (a 0-event result is a legitimate, good outcome), just that it runs cleanly against real data."""
    from core.tle_manager import TLEManager

    mgr = TLEManager()
    stations = mgr.fetch_group("stations")
    iss = next((s for s in stations if s.norad_id == 25544), None)
    assert iss is not None

    debris = mgr.fetch_group("debris")
    catalog = debris[:50]

    events = screen_close_approaches([iss], catalog, datetime.now(timezone.utc), window_hours=6.0)
    for event in events:
        assert event.severity in THRESHOLDS_KM
        assert event.min_distance_km >= 0
