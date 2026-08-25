from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from core.collision import ConjunctionResult
from stm.collision_avoid import (
    along_track_dv_for_target_miss_distance,
    clohessy_wiltshire_state,
    combined_maneuver_dv,
    plan_along_track_maneuver,
    plane_change_dv,
)


def test_clohessy_wiltshire_zero_dv_stays_at_origin():
    dv = np.zeros(3)
    n = 0.0011  # rad/s, roughly LEO mean motion
    pos = clohessy_wiltshire_state(dv, n, t=600.0)
    assert np.allclose(pos, 0.0)


def test_clohessy_wiltshire_along_track_secular_drift_exact_at_full_periods():
    # Pure along-track burn: y(t) = (4*dvy/n)*sin(nt) - 3*dvy*t. At t equal
    # to an exact multiple of the orbital period, sin(nt) = 0 and cos(nt) = 1,
    # so the oscillatory terms vanish exactly and y(t) = -3*dvy*t exactly.
    dvy = 0.001  # km/s
    n = 0.0011
    dv = np.array([0.0, dvy, 0.0])

    period = 2 * np.pi / n
    t_full_periods = 3 * period
    pos = clohessy_wiltshire_state(dv, n, t_full_periods)
    secular_exact = -3 * dvy * t_full_periods
    assert np.isclose(pos[1], secular_exact, atol=1e-9)


def test_clohessy_wiltshire_along_track_secular_drift_dominates_at_large_t():
    # Away from an exact period multiple, the oscillatory term is bounded
    # by 4*|dvy|/n regardless of t, so it becomes negligible relative to
    # the secular term as t grows.
    dvy = 0.001  # km/s
    n = 0.0011
    dv = np.array([0.0, dvy, 0.0])
    oscillatory_bound = 4 * abs(dvy) / n

    t_large = 5_000_000.0  # many orbits later
    pos = clohessy_wiltshire_state(dv, n, t_large)
    secular_approx = -3 * dvy * t_large
    assert abs(pos[1] - secular_approx) <= oscillatory_bound


def test_along_track_dv_increases_miss_distance():
    dv = along_track_dv_for_target_miss_distance(
        current_miss_distance_km=1.0, target_miss_distance_km=10.0, time_to_tca_s=3600.0
    )
    assert dv != 0.0
    # Larger lead time should require proportionally smaller dv for the same target.
    dv_more_lead = along_track_dv_for_target_miss_distance(
        current_miss_distance_km=1.0, target_miss_distance_km=10.0, time_to_tca_s=7200.0
    )
    assert abs(dv_more_lead) < abs(dv)


def test_along_track_dv_rejects_nonpositive_lead_time():
    with pytest.raises(ValueError):
        along_track_dv_for_target_miss_distance(1.0, 10.0, time_to_tca_s=0.0)


def test_plan_along_track_maneuver_achieves_target_via_cw():
    # Choose a lead time that is an exact multiple of the orbital period
    # so the CW oscillatory terms vanish and the secular-term prediction
    # can be checked exactly, rather than approximately.
    n = 0.0011
    period_s = 2 * np.pi / n
    lead_time = timedelta(seconds=4 * period_s)

    tca = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    burn_time = tca - lead_time
    conjunction = ConjunctionResult(tca=tca, miss_distance_km=1.0, relative_speed_km_s=7.5)

    maneuver = plan_along_track_maneuver(conjunction, burn_time, mean_motion_rad_s=n, target_miss_distance_km=5.0)

    assert maneuver.maneuver_type == "along-track"
    assert maneuver.delta_v_km_s > 0
    assert maneuver.predicted_miss_distance_km == 5.0

    # The CW y(t) here is the *additional* along-track offset the burn
    # introduces relative to the unperturbed trajectory; the resulting
    # miss distance is the pre-burn miss distance plus this offset.
    dv_signed = -(5.0 - 1.0) / (3 * maneuver.time_to_tca_s)
    pos = clohessy_wiltshire_state(np.array([0.0, dv_signed, 0.0]), n, maneuver.time_to_tca_s)
    resulting_miss = conjunction.miss_distance_km + pos[1]
    assert np.isclose(resulting_miss, 5.0, atol=1e-6)


def test_plan_along_track_maneuver_rejects_target_beyond_cw_validity():
    tca = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    burn_time = tca - timedelta(hours=2)
    conjunction = ConjunctionResult(tca=tca, miss_distance_km=1.0, relative_speed_km_s=7.5)
    with pytest.raises(ValueError):
        plan_along_track_maneuver(conjunction, burn_time, mean_motion_rad_s=0.0011, target_miss_distance_km=1000.0)


def test_plane_change_dv_matches_known_formula():
    v = 7.5  # km/s
    delta_i = np.radians(1.0)
    dv = plane_change_dv(v, delta_i)
    expected = 2 * v * np.sin(delta_i / 2)
    assert np.isclose(dv, expected)


def test_combined_maneuver_reduces_to_plane_change_when_speeds_equal():
    v = 7.5
    delta_i = np.radians(2.0)
    combined = combined_maneuver_dv(v, v, delta_i)
    pure = plane_change_dv(v, delta_i)
    assert np.isclose(combined, pure, rtol=1e-6)
