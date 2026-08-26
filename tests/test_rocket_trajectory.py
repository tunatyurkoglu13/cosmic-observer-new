from datetime import datetime, timezone

import numpy as np

from core.constants import R_EARTH
from core.launch_window import LAUNCH_SITES
from core.rocket_trajectory import _destination_point, compute_ascent_trajectory


def test_destination_point_due_north_moves_latitude_only():
    lat2, lon2 = _destination_point(0.0, 0.0, bearing_deg=0.0, distance_km=111.0)  # ~1 deg of latitude
    assert lat2 > 0.0
    assert abs(lon2) < 1e-6


def test_destination_point_due_east_at_equator_moves_longitude_only():
    lat2, lon2 = _destination_point(0.0, 0.0, bearing_deg=90.0, distance_km=111.0)
    assert abs(lat2) < 1e-6
    assert lon2 > 0.0


def test_destination_point_zero_distance_is_identity():
    lat2, lon2 = _destination_point(28.5, -80.5, bearing_deg=45.0, distance_km=0.0)
    assert np.isclose(lat2, 28.5, atol=1e-9)
    assert np.isclose(lon2, -80.5, atol=1e-9)


def test_compute_ascent_trajectory_starts_at_launch_site():
    site = LAUNCH_SITES["cape_canaveral"]
    launch_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trajectory = compute_ascent_trajectory(site, azimuth_deg=45.0, launch_time=launch_time, target_altitude_km=400.0)

    first = trajectory[0]
    assert np.isclose(first.lat_deg, site.lat_deg, atol=1e-6)
    assert np.isclose(first.lon_deg, site.lon_deg, atol=1e-6)
    assert np.isclose(first.alt_km, 0.0, atol=1e-9)
    assert first.t_offset_s == 0.0


def test_compute_ascent_trajectory_ends_near_target_altitude():
    site = LAUNCH_SITES["cape_canaveral"]
    launch_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trajectory = compute_ascent_trajectory(site, azimuth_deg=45.0, launch_time=launch_time, target_altitude_km=400.0)

    last = trajectory[-1]
    assert np.isclose(last.alt_km, 400.0, atol=1e-6)
    assert last.t_offset_s == 600.0  # default ascent_duration_s


def test_compute_ascent_trajectory_altitude_is_monotonically_nondecreasing():
    site = LAUNCH_SITES["cape_canaveral"]
    launch_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trajectory = compute_ascent_trajectory(site, azimuth_deg=45.0, launch_time=launch_time, target_altitude_km=400.0)

    altitudes = [s.alt_km for s in trajectory]
    assert all(altitudes[i] <= altitudes[i + 1] + 1e-9 for i in range(len(altitudes) - 1))


def test_compute_ascent_trajectory_r_teme_magnitude_grows_with_altitude():
    site = LAUNCH_SITES["cape_canaveral"]
    launch_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trajectory = compute_ascent_trajectory(site, azimuth_deg=45.0, launch_time=launch_time, target_altitude_km=400.0)

    first_radius = np.linalg.norm(trajectory[0].r_teme)
    last_radius = np.linalg.norm(trajectory[-1].r_teme)

    # Note: geocentric radius at 28.56 deg N sea level is a few km less than
    # the equatorial R_EARTH, due to WGS ellipsoid flattening (~4.9 km here,
    # matching a*f*sin^2(lat)) — hence the looser tolerance on first_radius.
    assert np.isclose(first_radius, R_EARTH, atol=10.0)
    # The final point has also moved ~2000 km downrange (default
    # max_downrange_km), to a different latitude with its own slightly
    # different ellipsoid-flattening correction, so allow a wider margin.
    assert np.isclose(last_radius, first_radius + 400.0, atol=10.0)
    assert last_radius > first_radius


def test_compute_ascent_trajectory_sample_count():
    site = LAUNCH_SITES["cape_canaveral"]
    launch_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trajectory = compute_ascent_trajectory(
        site, azimuth_deg=45.0, launch_time=launch_time, target_altitude_km=400.0, n_samples=21
    )
    assert len(trajectory) == 21
