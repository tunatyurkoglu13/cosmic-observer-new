from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from data.space_weather import (
    SWPC_KP_1M_URL,
    SWPC_KP_3H_URL,
    atmospheric_density,
    drag_acceleration,
    fetch_current_snapshot,
    fetch_f107_history,
    fetch_kp_history,
)


def test_atmospheric_density_decreases_with_altitude():
    rho_200 = atmospheric_density(200)
    rho_400 = atmospheric_density(400)
    rho_800 = atmospheric_density(800)
    assert rho_200 > rho_400 > rho_800 > 0


def test_atmospheric_density_sea_level_matches_table():
    # Table entry at h0=0 is rho0=1.225 kg/m^3 (standard sea-level density).
    assert np.isclose(atmospheric_density(0), 1.225, rtol=1e-6)


def test_atmospheric_density_kp_inflation_above_200km():
    baseline = atmospheric_density(400, kp_index=0)
    storm = atmospheric_density(400, kp_index=9)
    assert storm > baseline
    assert np.isclose(storm, baseline * 2.0, rtol=1e-6)


def test_atmospheric_density_kp_ignored_below_200km():
    baseline = atmospheric_density(100, kp_index=0)
    storm = atmospheric_density(100, kp_index=9)
    assert np.isclose(baseline, storm)


def test_atmospheric_density_rejects_negative_altitude():
    with pytest.raises(ValueError):
        atmospheric_density(-10)


def test_drag_acceleration_opposes_velocity():
    v = np.array([7.5, 0.0, 0.0])  # km/s
    a = drag_acceleration(v, altitude_km=400, drag_coefficient=2.2, area_m2=10.0, mass_kg=500.0)
    assert a[0] < 0
    assert np.isclose(a[1], 0.0)
    assert np.isclose(a[2], 0.0)


def test_drag_acceleration_zero_velocity_is_zero():
    v = np.zeros(3)
    a = drag_acceleration(v, altitude_km=400, drag_coefficient=2.2, area_m2=10.0, mass_kg=500.0)
    assert np.allclose(a, 0.0)


def test_drag_acceleration_increases_at_lower_altitude():
    v = np.array([7.5, 0.0, 0.0])
    a_low = drag_acceleration(v, altitude_km=300, drag_coefficient=2.2, area_m2=10.0, mass_kg=500.0)
    a_high = drag_acceleration(v, altitude_km=800, drag_coefficient=2.2, area_m2=10.0, mass_kg=500.0)
    assert np.linalg.norm(a_low) > np.linalg.norm(a_high)


@pytest.mark.network
def test_fetch_current_snapshot_live():
    snapshot = fetch_current_snapshot()
    assert 0.0 <= snapshot.kp_index <= 9.0
    assert snapshot.f107_flux > 0


def _json_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


def test_fetch_kp_history_merges_3h_product_with_thinned_1m_feed():
    one_minute = [{"time_tag": f"2026-11-04T12:{m:02d}:00", "estimated_kp": 2.0 + m / 100} for m in range(31)]
    three_hourly = [
        {"time_tag": "2026-11-04T06:00:00", "Kp": 1.0},
        {"time_tag": "2026-11-04T09:00:00", "Kp": 1.67},
        {"time_tag": "2026-11-04T12:00:00", "Kp": 2.33},  # overlaps the 1-minute feed — dropped
    ]
    responses = {SWPC_KP_1M_URL: one_minute, SWPC_KP_3H_URL: three_hourly}
    with patch("data.space_weather.requests.get", side_effect=lambda url, timeout: _json_response(responses[url])):
        history = fetch_kp_history(resample_minutes=15)

    times = [t.strftime("%H:%M") for t, _ in history]
    assert times == ["06:00", "09:00", "12:00", "12:15", "12:30"]
    assert history[2][1] == 2.0  # 12:00 comes from the 1-minute feed, not the 3-hourly product
    assert all(t.tzinfo is not None for t, _ in history)


def test_fetch_f107_history_sorts_oldest_first():
    payload = [
        {"time_tag": "2026-11-03T20:00:00", "flux": 100.0},
        {"time_tag": "2026-11-01T20:00:00", "flux": 98.0},
        {"time_tag": "2026-11-02T20:00:00", "flux": None},
    ]
    with patch("data.space_weather.requests.get", return_value=_json_response(payload)):
        history = fetch_f107_history()
    assert [v for _, v in history] == [98.0, 100.0]
