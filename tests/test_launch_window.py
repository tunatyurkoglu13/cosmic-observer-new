from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from core.constants import R_EARTH
from core.launch_window import (
    LAUNCH_SITES,
    LaunchSite,
    LaunchTarget,
    LaunchWindowCalculator,
    achievable_inclination_range,
    ascending_node_longitude_deg,
    hohmann_transfer,
    launch_azimuth,
    next_launch_window_time,
    phasing_wait_time,
    sun_synchronous_inclination,
)


def test_achievable_inclination_range_cape_canaveral():
    i_min, i_max = achievable_inclination_range(28.5621)
    assert np.isclose(i_min, 28.5621)
    assert np.isclose(i_max, 180.0 - 28.5621)


def test_launch_azimuth_iss_from_cape_canaveral_matches_known_geometry():
    # Cape Canaveral (28.5 deg N) -> ISS inclination (51.6 deg): known
    # real-world azimuth is roughly in the low-to-mid 40s degrees
    # (actual Shuttle-era published azimuth ~40 deg; our simplified
    # formula ignores Earth-rotation velocity addition, so expect it in
    # the same ballpark rather than exact agreement).
    az = launch_azimuth(51.6, 28.5621, direction="northeast")
    assert 35.0 < az < 55.0


def test_launch_azimuth_equatorial_orbit_from_cape_canaveral_unreachable():
    with pytest.raises(ValueError):
        launch_azimuth(0.0, 28.5621)


def test_launch_azimuth_at_site_latitude_gives_due_east():
    # When target inclination exactly equals the site latitude, sin(Az) = 1 -> Az = 90 deg (due east).
    az = launch_azimuth(28.5621, 28.5621, direction="northeast")
    assert np.isclose(az, 90.0, atol=1e-6)


def test_launch_azimuth_kourou_near_equatorial_reachable():
    # Kourou (5.236 deg N) can reach near-equatorial orbits directly.
    az = launch_azimuth(6.0, 5.2360, direction="northeast")
    assert 0.0 <= az <= 360.0


def test_launch_azimuth_northeast_southeast_symmetry():
    az_ne = launch_azimuth(51.6, 28.5621, direction="northeast")
    az_se = launch_azimuth(51.6, 28.5621, direction="southeast")
    assert np.isclose(az_se, 180.0 - az_ne)


def test_ascending_node_longitude_wraps_to_valid_range():
    epoch = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    lon = ascending_node_longitude_deg(raan_deg=123.4, epoch=epoch)
    assert 0.0 <= lon < 360.0


def test_next_launch_window_time_is_in_future_and_within_one_day():
    epoch = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    window = next_launch_window_time(site_lon_deg=-80.6, raan_deg=45.0, reference_epoch=epoch, orbit_count=0)
    assert window > epoch
    assert window - epoch <= timedelta(days=1, minutes=1)


def test_next_launch_window_time_orbit_count_advances_by_one_sidereal_day():
    # The formula advances by 2*pi/omega_E per orbit_count, i.e. one
    # sidereal day (~86164.09 s) — not one solar day (86400 s) — since
    # omega_E is Earth's sidereal rotation rate.
    epoch = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    w0 = next_launch_window_time(-80.6, 45.0, epoch, orbit_count=0)
    w1 = next_launch_window_time(-80.6, 45.0, epoch, orbit_count=1)
    delta = (w1 - w0).total_seconds()
    sidereal_day_s = 2 * np.pi / 7.2921159e-5
    assert np.isclose(delta, sidereal_day_s, atol=1.0)


def test_next_launch_window_time_matches_node_longitude_directly():
    # Construct an epoch/RAAN pair such that the node is already exactly
    # over the site's longitude -> the wait time should be ~0 (mod a full
    # rotation), i.e. orbit_count=0 gives a time within a few seconds of epoch.
    epoch = datetime(2026, 3, 15, 6, 0, tzinfo=timezone.utc)
    node_lon = ascending_node_longitude_deg(raan_deg=200.0, epoch=epoch)
    window = next_launch_window_time(site_lon_deg=node_lon, raan_deg=200.0, reference_epoch=epoch, orbit_count=0)
    assert (window - epoch).total_seconds() < 5.0


def test_sun_synchronous_inclination_is_retrograde_and_plausible():
    # Real SSO missions at ~700-800 km altitude use inclinations around 98 deg.
    i = sun_synchronous_inclination(altitude_km=700.0)
    assert 95.0 < i < 102.0


def test_sun_synchronous_inclination_increases_with_altitude():
    i_low = sun_synchronous_inclination(altitude_km=400.0)
    i_high = sun_synchronous_inclination(altitude_km=800.0)
    assert i_high > i_low


def test_phasing_wait_time_zero_relative_rate_raises():
    with pytest.raises(ValueError):
        phasing_wait_time(0.0011, 0.0011, k=1)


def test_phasing_wait_time_scales_with_k():
    t1 = phasing_wait_time(0.0012, 0.0011, k=1)
    t2 = phasing_wait_time(0.0012, 0.0011, k=2)
    assert np.isclose(t2, 2 * t1)


def test_hohmann_transfer_leo_to_geo_matches_known_ballpark():
    r1 = R_EARTH + 200.0   # LEO parking orbit
    r2 = R_EARTH + 35786.0  # GEO
    result = hohmann_transfer(r1, r2)
    # Textbook LEO->GEO Hohmann total delta-v is well known to be ~3.9 km/s.
    assert 3.7 < result.delta_v_total_km_s < 4.1
    assert result.transfer_time_s > 0


def test_hohmann_transfer_same_radius_is_zero_dv():
    r = R_EARTH + 500.0
    result = hohmann_transfer(r, r)
    assert np.isclose(result.delta_v_total_km_s, 0.0, atol=1e-9)


def test_launch_sites_preset_table_has_expected_entries():
    for key in ("cape_canaveral", "vandenberg", "baikonur", "kourou"):
        assert key in LAUNCH_SITES
        site = LAUNCH_SITES[key]
        assert isinstance(site, LaunchSite)
        assert -90.0 <= site.lat_deg <= 90.0
        assert -180.0 <= site.lon_deg <= 180.0


def test_launch_window_calculator_no_raan_returns_single_window():
    calc = LaunchWindowCalculator(LAUNCH_SITES["cape_canaveral"])
    target = LaunchTarget(altitude_km=400.0, inclination_deg=51.6)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=3)

    windows = calc.find_windows(target, start, end)
    assert len(windows) == 1
    assert windows[0].feasible


def test_launch_window_calculator_with_raan_returns_multiple_windows():
    calc = LaunchWindowCalculator(LAUNCH_SITES["cape_canaveral"])
    target = LaunchTarget(altitude_km=400.0, inclination_deg=51.6, raan_deg=200.0)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=3)

    windows = calc.find_windows(target, start, end)
    assert len(windows) >= 2
    for w in windows:
        assert start <= w.launch_time <= end
        assert w.feasible


def test_launch_window_calculator_infeasible_azimuth_reports_not_feasible():
    # Vandenberg's corridor (147-201 deg) does not include the azimuth
    # needed for a low/mid inclination like 30 deg from its ~34.6 deg
    # latitude (that inclination is below the site's own latitude and
    # thus unreachable outright).
    calc = LaunchWindowCalculator(LAUNCH_SITES["vandenberg"])
    target = LaunchTarget(altitude_km=500.0, inclination_deg=30.0)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    windows = calc.find_windows(target, start, end)
    assert len(windows) == 1
    assert not windows[0].feasible


def test_launch_window_calculator_sun_synchronous_overrides_inclination():
    # Sun-synchronous inclinations are retrograde (>90 deg); the
    # "northeast" azimuth solution for a retrograde inclination lands
    # near due-north/northwest, which is why real Vandenberg SSO launches
    # use the complementary "southeast" solution (azimuth ~180-190 deg,
    # within its 147-201 deg corridor) instead.
    calc = LaunchWindowCalculator(LAUNCH_SITES["vandenberg"])
    target = LaunchTarget(altitude_km=700.0, inclination_deg=0.0, sun_synchronous=True)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    windows = calc.find_windows(target, start, end, direction="southeast")
    assert len(windows) == 1
    assert windows[0].feasible
