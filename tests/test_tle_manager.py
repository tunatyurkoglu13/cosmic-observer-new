from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
import requests

from core.tle_manager import Satellite, TLEManager

LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
LINE2 = "2 00005  34.2682 348.7242 1627203 289.2321  55.4550 10.82419157413667"


def _make_sat(norad_id, classification):
    return Satellite(
        norad_id=norad_id, name=f"SAT-{norad_id}", line1=LINE1, line2=LINE2,
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        inclination_deg=34.27, eccentricity=0.1627, mean_motion_rev_per_day=10.82,
        semi_major_axis_km=8617.0, classification=classification,
    )


def test_satellite_belonging_to_two_groups_is_returned_for_both(tmp_path):
    """
    Regression test: a satellite fetched under group A and later also
    fetched under group B (e.g. the ISS appearing in both "stations" and
    "visual") must remain visible to load_cached("A") even after the "B"
    fetch — group membership is per-group, not a single overwritable field.
    """
    mgr = TLEManager(db_path=tmp_path / "test_cache.sqlite3")

    sat_as_stations = _make_sat(25544, "stations")
    mgr._upsert([sat_as_stations])

    sat_as_visual = _make_sat(25544, "visual")
    mgr._upsert([sat_as_visual])

    stations_result = mgr.load_cached(classification="stations")
    visual_result = mgr.load_cached(classification="visual")

    assert any(s.norad_id == 25544 for s in stations_result)
    assert any(s.norad_id == 25544 for s in visual_result)


def test_load_cached_classification_field_matches_queried_group(tmp_path):
    mgr = TLEManager(db_path=tmp_path / "test_cache.sqlite3")
    mgr._upsert([_make_sat(25544, "stations")])
    mgr._upsert([_make_sat(25544, "visual")])

    stations_result = mgr.load_cached(classification="stations")
    assert stations_result[0].classification == "stations"

    visual_result = mgr.load_cached(classification="visual")
    assert visual_result[0].classification == "visual"


def test_load_cached_no_filter_returns_first_seen_classification(tmp_path):
    mgr = TLEManager(db_path=tmp_path / "test_cache.sqlite3")
    mgr._upsert([_make_sat(25544, "stations")])
    mgr._upsert([_make_sat(25544, "visual")])  # should NOT overwrite the stored classification

    unfiltered = mgr.load_cached()
    assert len(unfiltered) == 1
    assert unfiltered[0].classification == "stations"


def test_satellite_from_tle_parses_norad_id_and_epoch():
    sat = Satellite.from_tle("TEST", LINE1, LINE2, classification="active")
    assert sat.norad_id == 5
    assert sat.inclination_deg == 34.2682
    assert sat.semi_major_axis_km > 0


def test_fetch_group_falls_back_to_stale_cache_on_network_failure(tmp_path):
    """
    Regression test for the real-world failure mode this project hit
    repeatedly: CelesTrak becomes temporarily unreachable. If we already
    have *any* cached data for the group, fetch_group should degrade to
    serving that stale data rather than raising — a slightly outdated
    catalog is far more useful to an interactive tool than a hard failure.
    """
    mgr = TLEManager(db_path=tmp_path / "test_cache.sqlite3", staleness=timedelta(hours=6))
    mgr._upsert([_make_sat(25544, "stations")])
    # Mark the group as having been fetched a long time ago, so
    # fetch_group() will actually attempt a fresh network call rather
    # than short-circuiting to the cache via the staleness check.
    import sqlite3
    with sqlite3.connect(mgr.db_path) as conn:
        conn.execute(
            "INSERT INTO fetch_log (group_name, fetched_at) VALUES (?, ?)",
            ("stations", (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()),
        )

    with patch("core.tle_manager.requests.get", side_effect=requests.ConnectTimeout("simulated outage")):
        result = mgr.fetch_group("stations")

    assert len(result) == 1
    assert result[0].norad_id == 25544


def test_fetch_group_raises_when_no_cache_and_network_fails(tmp_path):
    mgr = TLEManager(db_path=tmp_path / "test_cache.sqlite3")
    with patch("core.tle_manager.SEED_TLE_DIR", tmp_path / "no_seed_here"):
        with patch("core.tle_manager.requests.get", side_effect=requests.ConnectTimeout("simulated outage")):
            with pytest.raises(requests.ConnectTimeout):
                mgr.fetch_group("stations")


def test_fetch_group_falls_back_to_bundled_seed_when_no_cache_and_network_fails(tmp_path):
    """
    The deepest fallback: brand-new cache (e.g. a fresh clone of the repo)
    AND the network is down. fetch_group() must still return the bundled
    seed data rather than raising, so the app works out of the box.
    """
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (seed_dir / "stations.tle").write_text(f"ISS (ZARYA)\n{LINE1}\n{LINE2}\n")

    mgr = TLEManager(db_path=tmp_path / "test_cache.sqlite3")
    with patch("core.tle_manager.SEED_TLE_DIR", seed_dir):
        with patch("core.tle_manager.requests.get", side_effect=requests.ConnectTimeout("simulated outage")):
            result = mgr.fetch_group("stations")

    assert len(result) == 1
    assert result[0].name == "ISS (ZARYA)"

    # A subsequent call within the failure-retry cooldown must not eat
    # another network attempt (which could take a long time to time out
    # again) — it should go straight to the now-cached seed data. We
    # still patch requests.get here (raising if called) to make that
    # guarantee explicit rather than relying on timing.
    with patch("core.tle_manager.SEED_TLE_DIR", tmp_path / "gone"):
        with patch("core.tle_manager.requests.get", side_effect=AssertionError("should not hit network again")):
            cached_result = mgr.fetch_group("stations")
    assert len(cached_result) == 1
    assert cached_result[0].name == "ISS (ZARYA)"




def test_fetch_group_skips_network_during_failure_cooldown_even_with_no_cache(tmp_path):
    """
    During a sustained outage, repeated fetch_group() calls must not each
    eat a fresh connect-timeout — after one failure, subsequent calls
    within failure_retry_cooldown should return the seed fallback
    directly without attempting the network again.
    """
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (seed_dir / "stations.tle").write_text(f"ISS (ZARYA)\n{LINE1}\n{LINE2}\n")

    mgr = TLEManager(db_path=tmp_path / "test_cache.sqlite3", failure_retry_cooldown=timedelta(minutes=5))

    call_count = {"n": 0}

    def flaky_get(*args, **kwargs):
        call_count["n"] += 1
        raise requests.ConnectTimeout("simulated outage")

    with patch("core.tle_manager.SEED_TLE_DIR", seed_dir):
        with patch("core.tle_manager.requests.get", side_effect=flaky_get):
            mgr.fetch_group("stations")  # first call: fails, falls back to seed, records failure
            mgr.fetch_group("stations")  # second call: should skip network entirely (cooldown active)

    assert call_count["n"] == 1


def test_fetch_group_allow_stale_fallback_false_raises_even_with_cache(tmp_path):
    mgr = TLEManager(db_path=tmp_path / "test_cache.sqlite3", staleness=timedelta(hours=6))
    mgr._upsert([_make_sat(25544, "stations")])
    import sqlite3
    with sqlite3.connect(mgr.db_path) as conn:
        conn.execute(
            "INSERT INTO fetch_log (group_name, fetched_at) VALUES (?, ?)",
            ("stations", (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()),
        )

    with patch("core.tle_manager.requests.get", side_effect=requests.ConnectTimeout("simulated outage")):
        with pytest.raises(requests.ConnectTimeout):
            mgr.fetch_group("stations", allow_stale_fallback=False)
