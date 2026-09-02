from datetime import timedelta
from unittest.mock import patch

import pytest
import requests

from data.mars_rover_photos import MarsRoverPhotosClient, RoverPhoto


def _fake_photos():
    return [
        RoverPhoto(
            photo_id=1, sol=1000, camera_name="MAST", camera_full_name="Mast Camera",
            img_src="https://mars.nasa.gov/msl-raw-images/test.jpg", earth_date="2026-08-30",
            rover_name="Curiosity", rover_status="active",
        )
    ]


def test_fetch_latest_rejects_unknown_rover(tmp_path):
    client = MarsRoverPhotosClient(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown rover"):
        client.fetch_latest("opportunity")


def test_fetch_latest_caches_to_disk(tmp_path):
    client = MarsRoverPhotosClient(cache_dir=tmp_path)
    with patch.object(MarsRoverPhotosClient, "_fetch_live", return_value=_fake_photos()):
        result = client.fetch_latest("curiosity")
    assert result[0].rover_name == "Curiosity"
    assert (tmp_path / "curiosity.json").exists()

    with patch.object(MarsRoverPhotosClient, "_fetch_live", side_effect=AssertionError("should not refetch")):
        cached = client.fetch_latest("curiosity")
    assert cached[0].rover_name == "Curiosity"


def test_fetch_latest_falls_back_to_stale_cache_on_network_failure(tmp_path):
    client = MarsRoverPhotosClient(cache_dir=tmp_path, staleness=timedelta(seconds=0))
    with patch.object(MarsRoverPhotosClient, "_fetch_live", return_value=_fake_photos()):
        client.fetch_latest("curiosity", force=True)

    with patch.object(MarsRoverPhotosClient, "_fetch_live", side_effect=requests.ConnectTimeout("simulated outage")):
        result = client.fetch_latest("curiosity")

    assert result[0].rover_name == "Curiosity"


def test_fetch_latest_raises_when_no_cache_and_network_fails(tmp_path):
    client = MarsRoverPhotosClient(cache_dir=tmp_path)
    with patch.object(MarsRoverPhotosClient, "_fetch_live", side_effect=requests.ConnectTimeout("simulated outage")):
        with pytest.raises(requests.ConnectTimeout):
            client.fetch_latest("curiosity")


def test_fetch_live_parses_real_response_shape():
    """Exercises the real, documented api.nasa.gov response shape (see module docstring's honest status note)."""
    client = MarsRoverPhotosClient(cache_dir="/tmp/mars_rover_shape_test")
    fake_payload = {
        "latest_photos": [
            {
                "id": 42, "sol": 1000,
                "camera": {"name": "MAST", "full_name": "Mast Camera"},
                "img_src": "https://mars.nasa.gov/msl-raw-images/test.jpg",
                "earth_date": "2026-08-30",
                "rover": {"name": "Curiosity", "status": "active"},
            }
        ]
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return fake_payload

    with patch("data.mars_rover_photos.requests.get", return_value=FakeResponse()):
        result = client._fetch_live("curiosity")

    assert len(result) == 1
    assert result[0].photo_id == 42
    assert result[0].camera_name == "MAST"


@pytest.mark.network
def test_fetch_latest_live_or_honest_failure():
    """
    As of development, this NASA API's backend was unreachable (see
    module docstring) — this test accepts either real success (the
    service may have recovered) or a clean network-layer failure, but
    NOT silently-fabricated data either way.
    """
    client = MarsRoverPhotosClient(cache_dir="/tmp/cosmic_observer_test_mars_rover_cache")
    try:
        photos = client.fetch_latest("curiosity", force=True)
        assert len(photos) > 0
        assert photos[0].img_src.startswith("http")
    except requests.RequestException:
        pass  # confirmed-current real outage — an honest failure, not fabricated data
