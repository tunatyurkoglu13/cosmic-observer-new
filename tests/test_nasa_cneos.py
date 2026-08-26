from datetime import timedelta
from unittest.mock import patch

import pytest
import requests

from data.nasa_cneos import CNEOSClient, SentryObject, TORINO_DESCRIPTIONS


def _fake_risk_list():
    return [
        SentryObject(
            designation="99942", full_name="99942 Apophis (2004 MN4)", diameter_km=0.34,
            torino_scale_max=0, palermo_scale_cum=-5.23, palermo_scale_max=-5.23,
            impact_probability_cum=1.2e-05, potential_impact_count=3,
            years_observed="2029-2116", last_observed="2023-03-01",
        )
    ]


def test_sentry_object_from_api_record_parses_fields():
    rec = {
        "des": "99942",
        "fullname": "99942 Apophis (2004 MN4)",
        "diameter": "0.34",
        "ts_max": "0",
        "ps_cum": "-5.23",
        "ps_max": "-5.23",
        "ip": "1.2e-05",
        "n_imp": "3",
        "range": "2029-2116",
        "last_obs": "2023-03-01",
    }
    obj = SentryObject.from_api_record(rec)
    assert obj.designation == "99942"
    assert obj.diameter_km == 0.34
    assert obj.torino_scale_max == 0
    assert obj.risk_flag == "green"
    assert obj.torino_description == TORINO_DESCRIPTIONS[0]


def test_sentry_object_handles_missing_diameter():
    rec = {"des": "2024 AB", "fullname": "2024 AB", "ts_max": "2", "ps_cum": "-1.0",
           "ps_max": "-1.0", "ip": "0.0001", "n_imp": "1", "range": "2050-2050"}
    obj = SentryObject.from_api_record(rec)
    assert obj.diameter_km is None
    assert obj.risk_flag == "yellow"


@pytest.mark.network
def test_fetch_risk_list_live():
    client = CNEOSClient()
    objects = client.fetch_risk_list()
    assert len(objects) > 0
    assert all(isinstance(o, SentryObject) for o in objects)
    # Sorted descending by Torino scale (then Palermo).
    torinos = [o.torino_scale_max for o in objects]
    assert torinos == sorted(torinos, reverse=True)


def test_fetch_risk_list_caches_to_disk(tmp_path):
    client = CNEOSClient(cache_path=tmp_path / "cneos.json")
    with patch.object(CNEOSClient, "_fetch_live", return_value=_fake_risk_list()):
        result = client.fetch_risk_list()
    assert result[0].designation == "99942"
    assert (tmp_path / "cneos.json").exists()

    # A second call within the staleness window must not hit the network again.
    with patch.object(CNEOSClient, "_fetch_live", side_effect=AssertionError("should not refetch")):
        cached_result = client.fetch_risk_list()
    assert cached_result[0].designation == "99942"


def test_fetch_risk_list_falls_back_to_stale_cache_on_network_failure(tmp_path):
    client = CNEOSClient(cache_path=tmp_path / "cneos.json", staleness=timedelta(seconds=0))
    with patch.object(CNEOSClient, "_fetch_live", return_value=_fake_risk_list()):
        client.fetch_risk_list(force=True)

    with patch.object(CNEOSClient, "_fetch_live", side_effect=requests.ConnectTimeout("simulated outage")):
        result = client.fetch_risk_list()

    assert len(result) == 1
    assert result[0].designation == "99942"


def test_fetch_risk_list_skips_network_during_failure_cooldown(tmp_path):
    client = CNEOSClient(
        cache_path=tmp_path / "cneos.json",
        staleness=timedelta(seconds=0),
        failure_retry_cooldown=timedelta(minutes=5),
    )
    call_count = {"n": 0}

    def flaky_fetch(self, key):
        call_count["n"] += 1
        raise requests.ConnectTimeout("simulated outage")

    with patch.object(CNEOSClient, "_fetch_live", flaky_fetch):
        with pytest.raises(requests.ConnectTimeout):
            client.fetch_risk_list()  # first call: no cache, no seed -> raises, records failure
        with pytest.raises(requests.ConnectTimeout):
            client.fetch_risk_list()  # would normally retry network, but cooldown should still gate it

    assert call_count["n"] == 2  # cooldown only skips the network attempt when a fallback exists to serve instead


def test_fetch_risk_list_raises_when_no_cache_and_network_fails(tmp_path):
    client = CNEOSClient(cache_path=tmp_path / "cneos.json")
    with patch.object(CNEOSClient, "_fetch_live", side_effect=requests.ConnectTimeout("simulated outage")):
        with pytest.raises(requests.ConnectTimeout):
            client.fetch_risk_list()


def test_fetch_risk_list_allow_stale_fallback_false_raises_even_with_cache(tmp_path):
    client = CNEOSClient(cache_path=tmp_path / "cneos.json", staleness=timedelta(seconds=0))
    with patch.object(CNEOSClient, "_fetch_live", return_value=_fake_risk_list()):
        client.fetch_risk_list(force=True)

    with patch.object(CNEOSClient, "_fetch_live", side_effect=requests.ConnectTimeout("simulated outage")):
        with pytest.raises(requests.ConnectTimeout):
            client.fetch_risk_list(allow_stale_fallback=False)
