import pytest

from data.nasa_cneos import CNEOSClient, SentryObject, TORINO_DESCRIPTIONS


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
