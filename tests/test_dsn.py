from datetime import timedelta
from unittest.mock import patch

import pytest
import requests

from data.dsn import DSNClient, DSNStatus, _parse_dsn_xml

SAMPLE_XML = """<dsn>
\t<station name="gdscc" friendlyName="Goldstone" timeUTC="0" timeZoneOffset="0"/>
\t<dish name="DSS14" azimuthAngle="0" elevationAngle="90" windSpeed="" isMSPA="false" isArray="false" isDDOR="false" activity="Engineering Upgrades">
\t\t<target name="DSN" id="99" uplegRange="-1" downlegRange="-1" rtlt="-1"/>
\t</dish>
\t<dish name="DSS24" azimuthAngle="130" elevationAngle="68" windSpeed="12" isMSPA="false" isArray="false" isDDOR="false" activity="Spacecraft Telemetry, Tracking, and Command">
\t\t<upSignal active="false" signalType="none" dataRate="0" frequency="0" band="X" power="0" spacecraft="PSYC" spacecraftID="-255"/>
\t\t<downSignal active="true" signalType="data" dataRate="358700" frequency="0" band="X" power="-120" spacecraft="PSYC" spacecraftID="-255"/>
\t\t<target name="PSYC" id="255" uplegRange="-1" downlegRange="-1" rtlt="-1"/>
\t</dish>
\t<station name="mdscc" friendlyName="Madrid" timeUTC="0" timeZoneOffset="0"/>
\t<dish name="DSS63" azimuthAngle="10" elevationAngle="20" windSpeed="5" isMSPA="false" isArray="false" isDDOR="false" activity="Spacecraft Telemetry, Tracking, and Command">
\t\t<upSignal active="false" signalType="none" dataRate="0" frequency="0" band="X" power="0" spacecraft="MSL" spacecraftID="-76"/>
\t\t<downSignal active="false" signalType="none" dataRate="0" frequency="0" band="X" power="-480" spacecraft="MSL" spacecraftID="-76"/>
\t\t<target name="MSL" id="76" uplegRange="275000000" downlegRange="275000000" rtlt="-1"/>
\t</dish>
</dsn>
"""


def test_parse_dsn_xml_tracks_station_across_dishes():
    status = _parse_dsn_xml(SAMPLE_XML)
    assert status.station_names == ["Goldstone", "Madrid"]
    assert len(status.dishes) == 3
    assert status.dishes[0].station == "Goldstone"
    assert status.dishes[1].station == "Goldstone"
    assert status.dishes[2].station == "Madrid"


def test_parse_dsn_xml_parses_signals():
    status = _parse_dsn_xml(SAMPLE_XML)
    psyc_dish = status.dishes[1]
    assert psyc_dish.target_name == "PSYC"
    assert len(psyc_dish.signals) == 2
    down = next(s for s in psyc_dish.signals if s.direction == "down")
    assert down.active is True
    assert down.data_rate_bps == 358700.0
    assert down.spacecraft == "PSYC"


def test_parse_dsn_xml_negative_range_becomes_none():
    status = _parse_dsn_xml(SAMPLE_XML)
    dsn_dish = status.dishes[0]
    assert dsn_dish.uplink_range_km is None
    assert dsn_dish.downlink_range_km is None


def test_parse_dsn_xml_positive_range_preserved():
    status = _parse_dsn_xml(SAMPLE_XML)
    msl_dish = status.dishes[2]
    assert msl_dish.downlink_range_km == 275000000.0


def test_active_spacecraft_excludes_idle_targets():
    status = _parse_dsn_xml(SAMPLE_XML)
    assert status.active_spacecraft == ["MSL", "PSYC"]  # "DSN" (idle/maintenance target) excluded


def test_fetch_status_caches_to_disk(tmp_path):
    client = DSNClient(cache_path=tmp_path / "dsn.json")
    fake = _parse_dsn_xml(SAMPLE_XML)
    with patch.object(DSNClient, "_fetch_live", return_value=fake):
        result = client.fetch_status()
    assert result.active_spacecraft == ["MSL", "PSYC"]
    assert (tmp_path / "dsn.json").exists()

    with patch.object(DSNClient, "_fetch_live", side_effect=AssertionError("should not refetch")):
        cached = client.fetch_status()
    assert cached.active_spacecraft == ["MSL", "PSYC"]


def test_fetch_status_falls_back_to_stale_cache_on_network_failure(tmp_path):
    client = DSNClient(cache_path=tmp_path / "dsn.json", staleness=timedelta(seconds=0))
    fake = _parse_dsn_xml(SAMPLE_XML)
    with patch.object(DSNClient, "_fetch_live", return_value=fake):
        client.fetch_status(force=True)

    with patch.object(DSNClient, "_fetch_live", side_effect=requests.ConnectTimeout("simulated outage")):
        result = client.fetch_status()

    assert result.active_spacecraft == ["MSL", "PSYC"]


def test_fetch_status_raises_when_no_cache_and_network_fails(tmp_path):
    client = DSNClient(cache_path=tmp_path / "dsn.json")
    with patch.object(DSNClient, "_fetch_live", side_effect=requests.ConnectTimeout("simulated outage")):
        with pytest.raises(requests.ConnectTimeout):
            client.fetch_status()


@pytest.mark.network
def test_fetch_status_live():
    client = DSNClient(cache_path="/tmp/cosmic_observer_test_dsn_cache.json")
    status = client.fetch_status(force=True)
    assert len(status.dishes) > 0
    assert len(status.station_names) > 0
    # Real DSN stations are always one of these three.
    assert set(status.station_names) <= {"Goldstone", "Madrid", "Canberra"}
