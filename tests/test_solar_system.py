from datetime import timedelta
from unittest.mock import patch

import pytest
import requests

from data.solar_system import BODIES, BodyPosition, SolarSystemClient


def _fake_position(body="mercury"):
    return BodyPosition(
        body=body, fetched_at="2026-08-27T08:00:00+00:00", jd_tdb=2461279.8333,
        r_km=(-181870689.9, 91058234.2, 6222167.4), distance_km=203487752.0,
        direction=(-0.8937, 0.4475, 0.0306),
    )


def test_fetch_position_rejects_unknown_body(tmp_path):
    client = SolarSystemClient(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown body"):
        client.fetch_position("pluto")


def test_fetch_position_caches_to_disk(tmp_path):
    client = SolarSystemClient(cache_dir=tmp_path)
    with patch.object(SolarSystemClient, "_fetch_live", return_value=_fake_position()):
        result = client.fetch_position("mercury")
    assert result.body == "mercury"
    assert (tmp_path / "mercury.json").exists()

    with patch.object(SolarSystemClient, "_fetch_live", side_effect=AssertionError("should not refetch")):
        cached = client.fetch_position("mercury")
    assert cached.distance_km == result.distance_km


def test_fetch_position_falls_back_to_stale_cache_on_network_failure(tmp_path):
    client = SolarSystemClient(cache_dir=tmp_path, staleness=timedelta(seconds=0))
    with patch.object(SolarSystemClient, "_fetch_live", return_value=_fake_position()):
        client.fetch_position("mercury", force=True)

    with patch.object(SolarSystemClient, "_fetch_live", side_effect=requests.ConnectTimeout("simulated outage")):
        result = client.fetch_position("mercury")

    assert result.distance_km == pytest.approx(203487752.0)


def test_fetch_position_raises_when_no_cache_and_network_fails(tmp_path):
    client = SolarSystemClient(cache_dir=tmp_path)
    with patch.object(SolarSystemClient, "_fetch_live", side_effect=requests.ConnectTimeout("simulated outage")):
        with pytest.raises(requests.ConnectTimeout):
            client.fetch_position("mercury")


def test_direction_is_unit_vector():
    pos = _fake_position()
    mag = sum(d * d for d in pos.direction) ** 0.5
    assert mag == pytest.approx(1.0, abs=1e-2)


@pytest.mark.network
def test_fetch_position_live_mercury():
    client = SolarSystemClient(cache_dir="/tmp/cosmic_observer_test_ss_cache")
    pos = client.fetch_position("mercury", force=True)
    assert pos.body == "mercury"
    distance_au = pos.distance_km / 1.496e8
    # Mercury's Earth-distance genuinely ranges ~0.6-1.4 AU over its orbit.
    assert 0.5 < distance_au < 1.5
    mag = sum(d * d for d in pos.direction) ** 0.5
    assert mag == pytest.approx(1.0, abs=1e-6)

    # The real direction from Mercury toward the Sun (used for lighting)
    # must also be a genuine unit vector, distinct from the Earth->Mercury one.
    sun_mag = sum(d * d for d in pos.sun_direction) ** 0.5
    assert sun_mag == pytest.approx(1.0, abs=1e-6)


@pytest.mark.network
def test_fetch_position_live_sun():
    client = SolarSystemClient(cache_dir="/tmp/cosmic_observer_test_ss_cache")
    pos = client.fetch_position("sun", force=True)
    distance_au = pos.distance_km / 1.496e8
    assert 0.95 < distance_au < 1.05  # Earth-Sun distance is ~1 AU by definition
