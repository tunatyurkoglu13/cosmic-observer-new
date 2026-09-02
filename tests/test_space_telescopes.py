from datetime import timedelta
from unittest.mock import patch

import pytest
import requests

from data.space_telescopes import SpaceTelescopeClient, TelescopeObservation, _datetime_to_mjd, _mjd_to_datetime
from datetime import datetime, timezone


def _fake_observation(telescope="jwst"):
    return TelescopeObservation(
        telescope=telescope, display_name="James Webb Space Telescope", target_name="MCG-07-04-032",
        ra_deg=26.2657, dec_deg=-43.596, instrument="NIRCAM/IMAGE", proposal_id="9734",
        obs_title="Test proposal", observed_at_utc="2026-09-02T12:09:10+00:00",
    )


def test_mjd_datetime_roundtrip():
    now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
    mjd = _datetime_to_mjd(now)
    back = _mjd_to_datetime(mjd)
    assert abs((back - now).total_seconds()) < 1.0


def test_fetch_latest_observation_rejects_unknown_telescope(tmp_path):
    client = SpaceTelescopeClient(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown telescope"):
        client.fetch_latest_observation("spitzer")


def test_fetch_latest_observation_caches_to_disk(tmp_path):
    client = SpaceTelescopeClient(cache_dir=tmp_path)
    with patch.object(SpaceTelescopeClient, "_fetch_live", return_value=_fake_observation()):
        result = client.fetch_latest_observation("jwst")
    assert result.target_name == "MCG-07-04-032"
    assert (tmp_path / "jwst.json").exists()

    with patch.object(SpaceTelescopeClient, "_fetch_live", side_effect=AssertionError("should not refetch")):
        cached = client.fetch_latest_observation("jwst")
    assert cached.target_name == "MCG-07-04-032"


def test_fetch_latest_observation_falls_back_to_stale_cache_on_network_failure(tmp_path):
    client = SpaceTelescopeClient(cache_dir=tmp_path, staleness=timedelta(seconds=0))
    with patch.object(SpaceTelescopeClient, "_fetch_live", return_value=_fake_observation()):
        client.fetch_latest_observation("jwst", force=True)

    with patch.object(SpaceTelescopeClient, "_fetch_live", side_effect=requests.ConnectTimeout("simulated outage")):
        result = client.fetch_latest_observation("jwst")

    assert result.target_name == "MCG-07-04-032"


def test_fetch_latest_observation_raises_when_no_cache_and_network_fails(tmp_path):
    client = SpaceTelescopeClient(cache_dir=tmp_path)
    with patch.object(SpaceTelescopeClient, "_fetch_live", side_effect=requests.ConnectTimeout("simulated outage")):
        with pytest.raises(requests.ConnectTimeout):
            client.fetch_latest_observation("jwst")


@pytest.mark.network
def test_fetch_latest_observation_live_both_telescopes():
    client = SpaceTelescopeClient(cache_dir="/tmp/cosmic_observer_test_st_cache")
    for telescope in ("hubble", "jwst"):
        obs = client.fetch_latest_observation(telescope, force=True)
        assert obs.target_name
        assert obs.instrument
        assert -90 <= obs.dec_deg <= 90
        assert 0 <= obs.ra_deg < 360
