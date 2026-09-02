from datetime import timedelta
from unittest.mock import patch

import pytest
import requests

from data.solar_system import BODIES, MOONS, BodyPosition, SolarSystemClient


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


@pytest.mark.network
def test_fetch_position_live_venus_and_mars():
    client = SolarSystemClient(cache_dir="/tmp/cosmic_observer_test_ss_cache")

    venus = client.fetch_position("venus", force=True)
    venus_au = venus.distance_km / 1.496e8
    assert 0.25 < venus_au < 1.75  # Venus's real Earth-distance range

    mars = client.fetch_position("mars", force=True)
    mars_au = mars.distance_km / 1.496e8
    assert 0.35 < mars_au < 2.7  # Mars's real Earth-distance range

    for pos in (venus, mars):
        assert sum(d * d for d in pos.direction) ** 0.5 == pytest.approx(1.0, abs=1e-6)
        assert sum(d * d for d in pos.sun_direction) ** 0.5 == pytest.approx(1.0, abs=1e-6)
        assert pos.relative_to == "earth"


def test_fetch_moon_position_rejects_unknown_moon(tmp_path):
    client = SolarSystemClient(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown moon"):
        client.fetch_moon_position("triton")  # a real moon (Neptune's) but not one this project tracks


@pytest.mark.network
def test_fetch_moon_position_live_mars_moons():
    client = SolarSystemClient(cache_dir="/tmp/cosmic_observer_test_ss_cache")

    phobos = client.fetch_moon_position("phobos", force=True)
    assert phobos.relative_to == "mars"
    # Phobos orbits Mars at a real semi-major axis of ~9376 km.
    assert 9000 < phobos.distance_km < 9800

    deimos = client.fetch_moon_position("deimos", force=True)
    assert deimos.relative_to == "mars"
    # Deimos orbits Mars at a real semi-major axis of ~23463 km.
    assert 22800 < deimos.distance_km < 24200

    for pos in (phobos, deimos):
        assert sum(d * d for d in pos.direction) ** 0.5 == pytest.approx(1.0, abs=1e-6)
        assert sum(d * d for d in pos.sun_direction) ** 0.5 == pytest.approx(1.0, abs=1e-6)


@pytest.mark.network
def test_fetch_moon_position_live_earth_moon():
    client = SolarSystemClient(cache_dir="/tmp/cosmic_observer_test_ss_cache")
    moon = client.fetch_moon_position("moon", force=True)
    assert moon.relative_to == "earth"
    # The Moon's real distance from Earth varies ~356,500-406,700 km.
    assert 350000 < moon.distance_km < 410000
    assert sum(d * d for d in moon.sun_direction) ** 0.5 == pytest.approx(1.0, abs=1e-6)


@pytest.mark.network
def test_fetch_position_live_jupiter():
    client = SolarSystemClient(cache_dir="/tmp/cosmic_observer_test_ss_cache")
    jupiter = client.fetch_position("jupiter", force=True)
    jupiter_au = jupiter.distance_km / 1.496e8
    assert 3.9 < jupiter_au < 6.5  # Jupiter's real Earth-distance range
    assert sum(d * d for d in jupiter.direction) ** 0.5 == pytest.approx(1.0, abs=1e-6)
    assert sum(d * d for d in jupiter.sun_direction) ** 0.5 == pytest.approx(1.0, abs=1e-6)
    assert jupiter.relative_to == "earth"


@pytest.mark.network
def test_fetch_moon_position_live_galilean_moons():
    client = SolarSystemClient(cache_dir="/tmp/cosmic_observer_test_ss_cache")

    # Real semi-major axes (km): Io 421700, Europa 670900, Ganymede
    # 1070400, Callisto 1882700 — checked with generous tolerance since
    # actual distance varies with orbital position.
    expected_ranges = {
        "io": (390000, 450000),
        "europa": (620000, 720000),
        "ganymede": (1000000, 1140000),
        "callisto": (1750000, 2000000),
    }
    for key, (lo, hi) in expected_ranges.items():
        pos = client.fetch_moon_position(key, force=True)
        assert pos.relative_to == "jupiter"
        assert lo < pos.distance_km < hi, f"{key}: {pos.distance_km} not in [{lo}, {hi}]"
        assert sum(d * d for d in pos.direction) ** 0.5 == pytest.approx(1.0, abs=1e-6)
        assert sum(d * d for d in pos.sun_direction) ** 0.5 == pytest.approx(1.0, abs=1e-6)

    # Real ordering: Io closest, Callisto farthest.
    io_dist = client.fetch_moon_position("io").distance_km
    callisto_dist = client.fetch_moon_position("callisto").distance_km
    assert io_dist < callisto_dist


@pytest.mark.network
def test_fetch_position_live_saturn():
    client = SolarSystemClient(cache_dir="/tmp/cosmic_observer_test_ss_cache")
    saturn = client.fetch_position("saturn", force=True)
    saturn_au = saturn.distance_km / 1.496e8
    assert 8.0 < saturn_au < 11.0  # Saturn's real Earth-distance range
    assert sum(d * d for d in saturn.direction) ** 0.5 == pytest.approx(1.0, abs=1e-6)
    assert sum(d * d for d in saturn.sun_direction) ** 0.5 == pytest.approx(1.0, abs=1e-6)
    assert saturn.relative_to == "earth"


@pytest.mark.network
def test_fetch_moon_position_live_saturn_moons():
    client = SolarSystemClient(cache_dir="/tmp/cosmic_observer_test_ss_cache")

    # Real semi-major axes (km): Mimas 185540, Enceladus 238037, Rhea
    # 527108, Titan 1221870, Helene ~377400 — generous tolerance for
    # actual orbital position.
    expected_ranges = {
        "mimas": (170000, 200000),
        "enceladus": (220000, 255000),
        "rhea": (500000, 555000),
        "titan": (1150000, 1300000),
        "helene": (350000, 400000),
    }
    for key, (lo, hi) in expected_ranges.items():
        pos = client.fetch_moon_position(key, force=True)
        assert pos.relative_to == "saturn"
        assert lo < pos.distance_km < hi, f"{key}: {pos.distance_km} not in [{lo}, {hi}]"
        assert sum(d * d for d in pos.direction) ** 0.5 == pytest.approx(1.0, abs=1e-6)
        assert sum(d * d for d in pos.sun_direction) ** 0.5 == pytest.approx(1.0, abs=1e-6)

    # Real ordering: Mimas closest, Titan farthest.
    mimas_dist = client.fetch_moon_position("mimas").distance_km
    titan_dist = client.fetch_moon_position("titan").distance_km
    assert mimas_dist < titan_dist
