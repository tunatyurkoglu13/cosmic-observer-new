from datetime import datetime, timezone

import numpy as np

from viz.earth_render import is_daylight, subsolar_point, terminator_points


def test_subsolar_latitude_within_declination_bounds():
    # Earth's axial tilt bounds the subsolar latitude to roughly +/-23.5 deg year-round.
    for month in range(1, 13):
        dt = datetime(2026, month, 15, 12, 0, tzinfo=timezone.utc)
        lat, lon = subsolar_point(dt)
        assert -23.5 <= lat <= 23.5
        assert -180.0 <= lon <= 180.0


def test_subsolar_point_near_summer_solstice_max_north_declination():
    # Around June 21, the subsolar latitude should be near its northern maximum (~23.4 deg).
    dt = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    lat, _ = subsolar_point(dt)
    assert 22.0 < lat < 23.6


def test_subsolar_point_near_winter_solstice_max_south_declination():
    dt = datetime(2026, 12, 21, 12, 0, tzinfo=timezone.utc)
    lat, _ = subsolar_point(dt)
    assert -23.6 < lat < -22.0


def test_terminator_points_are_90_degrees_from_subsolar_point():
    dt = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    lat_s, lon_s = subsolar_point(dt)
    s = np.array([
        np.cos(np.radians(lat_s)) * np.cos(np.radians(lon_s)),
        np.cos(np.radians(lat_s)) * np.sin(np.radians(lon_s)),
        np.sin(np.radians(lat_s)),
    ])

    points = terminator_points(dt, n_points=36)
    assert len(points) == 36
    for lat, lon in points:
        p = np.array([
            np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
            np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
            np.sin(np.radians(lat)),
        ])
        assert np.isclose(np.dot(s, p), 0.0, atol=1e-9)


def test_is_daylight_true_at_subsolar_point():
    dt = datetime(2026, 5, 1, 6, 0, tzinfo=timezone.utc)
    lat_s, lon_s = subsolar_point(dt)
    assert is_daylight(lat_s, lon_s, dt)


def test_is_daylight_false_at_antisolar_point():
    dt = datetime(2026, 5, 1, 6, 0, tzinfo=timezone.utc)
    lat_s, lon_s = subsolar_point(dt)
    antisolar_lat = -lat_s
    antisolar_lon = ((lon_s + 180.0) + 180.0) % 360.0 - 180.0
    assert not is_daylight(antisolar_lat, antisolar_lon, dt)
