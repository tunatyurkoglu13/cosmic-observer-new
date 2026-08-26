from unittest.mock import MagicMock, patch

import pytest
import requests

from core.space_track_client import SpaceTrackClient
from core.tle_manager import TLEManager


def test_not_configured_without_credentials():
    client = SpaceTrackClient(username="", password="")
    assert client.is_configured is False


def test_not_configured_with_placeholder_values():
    """The literal .env.example placeholders must not be mistaken for real credentials."""
    client = SpaceTrackClient(username="your_username", password="your_password")
    assert client.is_configured is False


def test_configured_with_real_looking_credentials():
    client = SpaceTrackClient(username="a_real_user", password="hunter2")
    assert client.is_configured is True


def test_login_raises_when_not_configured():
    client = SpaceTrackClient(username="", password="")
    with pytest.raises(RuntimeError, match="not configured"):
        client.login()


def test_login_success_sets_logged_in_flag():
    client = SpaceTrackClient(username="user", password="pass")
    fake_resp = MagicMock(text="", status_code=200)
    fake_resp.raise_for_status.return_value = None
    with patch.object(client._session, "request", return_value=fake_resp) as mock_request:
        client.login()
    assert client._logged_in is True
    called_url = mock_request.call_args.args[1]
    assert called_url.endswith("ajaxauth/login")
    assert mock_request.call_args.kwargs["data"] == {"identity": "user", "password": "pass"}


def test_login_raises_on_rejected_credentials():
    client = SpaceTrackClient(username="user", password="wrong")
    fake_resp = MagicMock(text='{"Login": "Failed"}', status_code=200)
    fake_resp.raise_for_status.return_value = None
    with patch.object(client._session, "request", return_value=fake_resp):
        with pytest.raises(RuntimeError, match="rejected"):
            client.login()
    assert client._logged_in is False


def test_fetch_group_tle_raises_valueerror_for_unmapped_group():
    client = SpaceTrackClient(username="user", password="pass")
    client._logged_in = True  # skip login for this test
    with pytest.raises(ValueError, match="No Space-Track OBJECT_NAME mapping"):
        client.fetch_group_tle("active")


def test_fetch_group_tle_builds_expected_query_url():
    client = SpaceTrackClient(username="user", password="pass")
    client._logged_in = True
    fake_resp = MagicMock(text="TLE TEXT HERE", status_code=200)
    fake_resp.raise_for_status.return_value = None
    with patch.object(client._session, "request", return_value=fake_resp) as mock_request:
        result = client.fetch_group_tle("starlink")
    assert result == "TLE TEXT HERE"
    called_url = mock_request.call_args.args[1]
    assert "class/gp/" in called_url
    assert "OBJECT_NAME/~~STARLINK" in called_url
    assert "format/tle" in called_url


def test_rate_limiter_sleeps_between_requests():
    client = SpaceTrackClient(username="user", password="pass", min_request_interval=0.05)
    client._logged_in = True
    fake_resp = MagicMock(text="ok", status_code=200)
    fake_resp.raise_for_status.return_value = None
    with patch.object(client._session, "request", return_value=fake_resp):
        with patch("core.space_track_client.time.sleep") as mock_sleep:
            client.fetch_gp(norad_ids=[25544])
            client.fetch_gp(norad_ids=[25544])
    assert mock_sleep.called


def test_tle_manager_falls_back_to_celestrak_when_space_track_unconfigured(tmp_path):
    """
    A SpaceTrackClient passed in without credentials must behave
    identically to not passing one at all — the whole point of the
    is_configured guard in TLEManager._fetch_live.
    """
    unconfigured = SpaceTrackClient(username="", password="")
    mgr = TLEManager(db_path=tmp_path / "cache.sqlite3", space_track_client=unconfigured)

    fake_resp = MagicMock(text="ISS (ZARYA)\n1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9008\n2 25544  51.6416 339.9500 0001177  93.3861 264.7930 15.49560752 42982\n", status_code=200)
    fake_resp.raise_for_status.return_value = None
    with patch("core.tle_manager.requests.get", return_value=fake_resp) as mock_get:
        result = mgr.fetch_group("stations")

    assert mock_get.called  # CelesTrak was actually used
    assert len(result) == 1
    assert result[0].norad_id == 25544


def test_tle_manager_uses_space_track_when_configured_and_falls_back_on_failure(tmp_path):
    """When Space-Track is configured but its request fails, TLEManager must still fall through to CelesTrak."""
    configured = SpaceTrackClient(username="user", password="pass")
    mgr = TLEManager(db_path=tmp_path / "cache.sqlite3", space_track_client=configured)

    fake_celestrak_resp = MagicMock(text="ISS (ZARYA)\n1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9008\n2 25544  51.6416 339.9500 0001177  93.3861 264.7930 15.49560752 42982\n", status_code=200)
    fake_celestrak_resp.raise_for_status.return_value = None

    with patch.object(configured, "fetch_group_tle", side_effect=requests.ConnectTimeout("space-track down")):
        with patch("core.tle_manager.requests.get", return_value=fake_celestrak_resp) as mock_get:
            result = mgr.fetch_group("stations")

    assert mock_get.called
    assert len(result) == 1
    assert result[0].norad_id == 25544
