from datetime import timedelta
from unittest.mock import patch

import pytest
import requests

from data.dscovr_epic import EPICClient, EPICImage


def _fake_image():
    return EPICImage(
        identifier="20260831224420", caption="Test caption",
        image_url="https://epic.gsfc.nasa.gov/archive/natural/2026/08/31/png/epic_1b_20260831224420.png",
        date="2026-08-31 22:39:32", centroid_lat=7.95, centroid_lon=-155.30,
    )


def test_fetch_latest_caches_to_disk(tmp_path):
    client = EPICClient(cache_path=tmp_path / "epic.json")
    with patch.object(EPICClient, "_fetch_live", return_value=_fake_image()):
        result = client.fetch_latest()
    assert result.identifier == "20260831224420"
    assert (tmp_path / "epic.json").exists()

    with patch.object(EPICClient, "_fetch_live", side_effect=AssertionError("should not refetch")):
        cached = client.fetch_latest()
    assert cached.identifier == "20260831224420"


def test_fetch_latest_falls_back_to_stale_cache_on_network_failure(tmp_path):
    client = EPICClient(cache_path=tmp_path / "epic.json", staleness=timedelta(seconds=0))
    with patch.object(EPICClient, "_fetch_live", return_value=_fake_image()):
        client.fetch_latest(force=True)

    with patch.object(EPICClient, "_fetch_live", side_effect=requests.ConnectTimeout("simulated outage")):
        result = client.fetch_latest()

    assert result.identifier == "20260831224420"


def test_fetch_latest_raises_when_no_cache_and_network_fails(tmp_path):
    client = EPICClient(cache_path=tmp_path / "epic.json")
    with patch.object(EPICClient, "_fetch_live", side_effect=requests.ConnectTimeout("simulated outage")):
        with pytest.raises(requests.ConnectTimeout):
            client.fetch_latest()


def test_fetch_live_builds_correct_archive_url(tmp_path):
    client = EPICClient(cache_path=tmp_path / "epic.json")
    fake_response_json = [
        {
            "identifier": "20260831005515", "caption": "real caption",
            "image": "epic_1b_20260831005515", "date": "2026-08-31 00:50:27",
            "centroid_coordinates": {"lat": 7.95, "lon": 172.29},
        }
    ]

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return fake_response_json

    with patch("data.dscovr_epic.requests.get", return_value=FakeResponse()):
        result = client._fetch_live("latest")

    assert result.image_url == "https://epic.gsfc.nasa.gov/archive/natural/2026/08/31/png/epic_1b_20260831005515.png"
    assert result.centroid_lat == 7.95


@pytest.mark.network
def test_fetch_latest_live():
    client = EPICClient(cache_path="/tmp/cosmic_observer_test_epic_cache.json")
    image = client.fetch_latest(force=True)
    assert image.identifier
    assert image.image_url.startswith("https://epic.gsfc.nasa.gov/archive/natural/")
    assert -90 <= image.centroid_lat <= 90
