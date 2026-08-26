from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from core.launch_window import LAUNCH_SITES
from core.propagator import ecef_to_teme, geodetic_to_ecef, gmst_from_jd
from core.rocket_trajectory import compute_ascent_trajectory
from core.tle_manager import Satellite
from stm.cola import build_catalog_grid, find_cola_launch_windows, screen_launch_time

# Real TLEs for two objects at very different altitude bands (used only
# as *propagatable* satellites — the point of these tests is to control
# where a synthetic "test object" is relative to the rocket, not to use
# realistic orbital elements).
ISS_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9008"
ISS_LINE2 = "2 25544  51.6416 339.9500 0001177  93.3861 264.7930 15.49560752 42982"


def _iss_like_sat(norad_id=25544, name="ISS"):
    return Satellite(
        norad_id=norad_id, name=name, line1=ISS_LINE1, line2=ISS_LINE2,
        epoch=datetime(2024, 1, 1, tzinfo=timezone.utc),
        inclination_deg=51.64, eccentricity=0.0001177, mean_motion_rev_per_day=15.4956,
        semi_major_axis_km=6796.0, classification="stations",
    )


def test_build_catalog_grid_shape_and_values():
    sat = _iss_like_sat()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=5)
    grid = build_catalog_grid([sat], start, end, grid_step=timedelta(seconds=30))

    assert grid.positions.shape[0] == 1
    assert grid.positions.shape[2] == 3
    assert np.all(np.isfinite(grid.positions))  # ISS-like TLE should propagate cleanly throughout


def test_screen_launch_time_clear_when_catalog_object_is_far_away():
    # A geostationary-like object (42164 km radius) is always far from a
    # LEO-insertion ascent trajectory (max radius ~ R_EARTH + 400 km).
    geo_sat = Satellite(
        norad_id=90001, name="FAR-AWAY-GEO", line1=ISS_LINE1, line2=ISS_LINE2,
        epoch=datetime(2024, 1, 1, tzinfo=timezone.utc),
        inclination_deg=0.05, eccentricity=0.0001, mean_motion_rev_per_day=1.0027,
        semi_major_axis_km=42164.0, classification="active",
    )
    # Give it a TLE that actually reflects a GEO orbit (mean_motion ~1 rev/day),
    # so propagation places it near GEO radius, not the ISS's actual real orbit.
    geo_sat.line1 = "1 90001U 24001A   24001.50000000  .00000000  00000-0  00000-0 0  9000"
    geo_sat.line2 = "2 90001   0.0500 100.0000 0001000  90.0000 270.0000  1.00270000000009"

    site = LAUNCH_SITES["cape_canaveral"]
    launch_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    catalog = build_catalog_grid(
        [geo_sat], launch_time, launch_time + timedelta(seconds=700), grid_step=timedelta(seconds=15)
    )

    result = screen_launch_time(
        site, target_inclination_deg=51.6, launch_time=launch_time, catalog=catalog,
        target_altitude_km=400.0, ascent_duration_s=600.0, bubble_radius_km=50.0,
    )
    assert result.clear is True
    assert result.violations == []
    assert result.closest_approach_km > 50.0


def test_screen_launch_time_flags_synthetic_object_placed_on_the_ascent_path():
    """
    Construct a synthetic 'catalog object' by directly injecting a
    precomputed grid position that sits exactly on the rocket's ascent
    path at a known instant, and verify it gets flagged as a violation
    at (approximately) that instant and not flagged as clear.
    """
    site = LAUNCH_SITES["cape_canaveral"]
    launch_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    azimuth = 45.0  # matches launch_azimuth(51.6, 28.5621) approx, but we don't need exact match

    trajectory = compute_ascent_trajectory(
        site, azimuth, launch_time, target_altitude_km=400.0, ascent_duration_s=600.0
    )
    midpoint = trajectory[len(trajectory) // 2]

    # Build a fake catalog grid whose sole object sits exactly at the
    # rocket's own position at the midpoint sample time, at every grid time
    # (a "stationary" synthetic hazard for test purposes).
    grid_times = [midpoint.absolute_time + timedelta(seconds=s) for s in range(-30, 31, 15)]
    fake_positions = np.tile(midpoint.r_teme, (1, len(grid_times), 1))

    from stm.cola import CatalogGrid
    fake_sat = Satellite(
        norad_id=1, name="SYNTHETIC-HAZARD", line1="x", line2="y",
        epoch=launch_time, inclination_deg=0, eccentricity=0, mean_motion_rev_per_day=15,
        semi_major_axis_km=7000, classification="debris",
    )
    catalog = CatalogGrid(
        grid_times=grid_times,
        jd_array=np.zeros(len(grid_times)), fr_array=np.zeros(len(grid_times)),
        satellites=[fake_sat], positions=fake_positions,
    )

    result = screen_launch_time(
        site, target_inclination_deg=51.6, launch_time=launch_time, catalog=catalog,
        target_altitude_km=400.0, ascent_duration_s=600.0, bubble_radius_km=50.0, direction="northeast",
    )

    assert result.clear is False
    assert len(result.violations) >= 1
    assert result.violations[0].satellite.name == "SYNTHETIC-HAZARD"
    assert result.violations[0].distance_km < 50.0
    assert result.closest_approach_km < 1.0  # the hazard sits essentially exactly on the path


@pytest.mark.slow
def test_find_cola_launch_windows_returns_one_result_per_candidate():
    sat = _iss_like_sat()
    site = LAUNCH_SITES["cape_canaveral"]
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=10)

    results = find_cola_launch_windows(
        site, target_inclination_deg=51.6, satellites=[sat],
        search_start=start, search_end=end, candidate_step=timedelta(minutes=2),
    )
    assert len(results) == 6  # 0,2,4,6,8,10 minutes
    for r in results:
        assert isinstance(r.clear, bool)
        assert r.azimuth_deg > 0
