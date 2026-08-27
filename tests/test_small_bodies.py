from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pytest

from data.sbdb import OrbitalElements, SmallBody
from data.small_bodies import (
    CURATED_BODIES,
    SmallBodyTracker,
    _julian_date_now,
    _propagate_heliocentric_position,
)


def _fake_body():
    import math

    from data.small_bodies import AU_KM, GM_SUN_KM3_S2

    a_au = 1.2
    a_km = a_au * AU_KM
    real_period_days = 2 * math.pi * math.sqrt(a_km**3 / GM_SUN_KM3_S2) / 86400.0

    return SmallBody(
        designation="TEST1", full_name="Test Body 1", orbit_class_name="Amor", orbit_class_code="AMO",
        is_neo=True, is_potentially_hazardous=False,
        elements=OrbitalElements(
            eccentricity=0.05, semi_major_axis_au=a_au, perihelion_distance_au=1.14,
            inclination_deg=5.0, longitude_ascending_node_deg=80.0, argument_perihelion_deg=30.0,
            mean_anomaly_deg=0.0, period_days=real_period_days, epoch_jd=2451545.0,  # J2000.0
        ),
    )


def test_fetch_elements_rejects_unknown_body(tmp_path):
    tracker = SmallBodyTracker(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown small body"):
        tracker.fetch_elements("pluto")


def test_propagate_heliocentric_position_moves_with_time():
    body = _fake_body()
    jd_epoch = body.elements.epoch_jd
    r_at_epoch = _propagate_heliocentric_position(body, jd_epoch)
    r_much_later = _propagate_heliocentric_position(body, jd_epoch + body.elements.period_days / 2)

    # Half a period later, the body should be roughly on the opposite
    # side of its orbit (near-antipodal direction from the Sun).
    cos_angle = np.dot(r_at_epoch, r_much_later) / (np.linalg.norm(r_at_epoch) * np.linalg.norm(r_much_later))
    assert cos_angle < -0.9


def test_propagate_heliocentric_position_full_period_returns_to_start():
    body = _fake_body()
    jd_epoch = body.elements.epoch_jd
    r_at_epoch = _propagate_heliocentric_position(body, jd_epoch)
    r_one_period_later = _propagate_heliocentric_position(body, jd_epoch + body.elements.period_days)
    assert np.allclose(r_at_epoch, r_one_period_later, rtol=1e-3)


def test_current_position_caches_elements_to_disk(tmp_path):
    tracker = SmallBodyTracker(cache_dir=tmp_path)
    with patch.object(SmallBodyTracker, "_fetch_live", return_value=_fake_body()):
        state = tracker.current_position("eros", earth_helio_r_km=(1.496e8, 0.0, 0.0))
    assert state.key == "eros"
    assert state.designation == "TEST1"
    assert (tmp_path / "eros.json").exists()

    with patch.object(SmallBodyTracker, "_fetch_live", side_effect=AssertionError("should not refetch")):
        state2 = tracker.current_position("eros", earth_helio_r_km=(1.496e8, 0.0, 0.0))
    # Not exactly equal: current_position re-propagates to "now" each
    # call (cheap, local math), so real wall-clock time elapsed between
    # the two calls causes a tiny but genuine difference. What's cached
    # (and must NOT be refetched, enforced by the AssertionError above)
    # is the orbital ELEMENTS, not the derived position.
    assert state2.distance_km == pytest.approx(state.distance_km, rel=1e-6)


def test_current_position_direction_is_unit_vector(tmp_path):
    tracker = SmallBodyTracker(cache_dir=tmp_path)
    with patch.object(SmallBodyTracker, "_fetch_live", return_value=_fake_body()):
        state = tracker.current_position("bennu", earth_helio_r_km=(1.496e8, 0.0, 0.0))
    mag = sum(d * d for d in state.direction) ** 0.5
    assert mag == pytest.approx(1.0, abs=1e-6)


@pytest.mark.network
def test_current_position_live_all_curated_bodies():
    from data.solar_system import SolarSystemClient

    ssc = SolarSystemClient(cache_dir="/tmp/cosmic_observer_test_ss_cache2")
    sun_pos = ssc.fetch_position("sun", force=True)
    earth_helio = tuple(-x for x in sun_pos.r_km)

    tracker = SmallBodyTracker(cache_dir="/tmp/cosmic_observer_test_sb_cache")
    for key in CURATED_BODIES:
        state = tracker.current_position(key, earth_helio)
        assert state.distance_km > 0
        mag = sum(d * d for d in state.direction) ** 0.5
        assert mag == pytest.approx(1.0, abs=1e-6)
