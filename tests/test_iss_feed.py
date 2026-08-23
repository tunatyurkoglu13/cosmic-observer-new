import pytest

from data.iss_feed import get_iss_status


@pytest.mark.network
def test_get_iss_status_live():
    status = get_iss_status()
    assert status.satellite.norad_id == 25544
    assert -90.0 <= status.state.lat_deg <= 90.0
    assert -180.0 <= status.state.lon_deg <= 180.0
    assert 300.0 < status.state.alt_km < 500.0  # ISS operates ~400-420 km nominally
    assert "nasa_live" in status.stream_sources
