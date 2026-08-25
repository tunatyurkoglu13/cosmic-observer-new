from datetime import datetime, timedelta, timezone

from core.propagator import Propagator
from viz.orbit_trails import compute_ground_track, compute_orbit_path_teme

# Reuse the same Vanguard-1-derived TLE fixture as test_propagator.py.
LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
LINE2 = "2 00005  34.2682 348.7242 1627203 289.2321  55.4550 10.82419157413667"


def _epoch():
    return datetime(2000, 6, 27, 18, 50, 19, tzinfo=timezone.utc)


def test_compute_orbit_path_teme_returns_closed_loop_of_requested_length():
    prop = Propagator(LINE1, LINE2, name="test-sat")
    a = 8617.0  # km, matches the Vanguard-1 fixture used in test_propagator.py
    path = compute_orbit_path_teme(prop, _epoch(), semi_major_axis_km=a, num_points=50)
    assert len(path) == 50
    for x, y, z in path:
        r = (x**2 + y**2 + z**2) ** 0.5
        assert 6000 < r < 12000  # sane geocentric radius range for this orbit


def test_compute_ground_track_produces_valid_latlon_segments():
    prop = Propagator(LINE1, LINE2, name="test-sat")
    segments = compute_ground_track(
        prop, _epoch(), duration=timedelta(hours=3), step=timedelta(minutes=5)
    )
    assert len(segments) >= 1
    total_points = sum(len(seg) for seg in segments)
    assert total_points > 0

    for seg in segments:
        for lat, lon in seg:
            assert -90.0 <= lat <= 90.0
            assert -180.0 <= lon <= 180.0


def test_compute_ground_track_splits_at_antimeridian_crossings():
    prop = Propagator(LINE1, LINE2, name="test-sat")
    segments = compute_ground_track(
        prop, _epoch(), duration=timedelta(hours=6), step=timedelta(minutes=2)
    )
    # Within each individual segment, consecutive longitude samples should
    # never jump by more than 180 deg (that's exactly what triggers a new segment).
    for seg in segments:
        for i in range(1, len(seg)):
            assert abs(seg[i][1] - seg[i - 1][1]) <= 180.0
