import pytest
from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


def test_index_serves_dashboard_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "COSMIC OBSERVER" in resp.text


def test_static_assets_resolve_at_root_relative_paths():
    # index.html references these with relative (no-leading-slash) paths,
    # so they must resolve at the root, not under /static/.
    for asset in ("retro.css", "dashboard.js"):
        resp = client.get(f"/{asset}")
        assert resp.status_code == 200


@pytest.mark.network
def test_list_satellites_stations_group():
    resp = client.get("/api/satellites", params={"group": "stations"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["group"] == "stations"
    assert data["count"] > 0
    assert any(s["name"].startswith("ISS") for s in data["satellites"])


def test_list_satellites_rejects_unknown_group():
    resp = client.get("/api/satellites", params={"group": "not-a-real-group"})
    assert resp.status_code == 400


@pytest.mark.network
def test_satellite_position_returns_valid_geodetic():
    resp = client.get("/api/satellites/25544/position", params={"group": "stations"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["norad_id"] == 25544
    assert -90.0 <= data["lat_deg"] <= 90.0
    assert -180.0 <= data["lon_deg"] <= 180.0
    assert 300.0 < data["alt_km"] < 500.0


def test_satellite_position_404_for_unknown_norad_id():
    resp = client.get("/api/satellites/99999999/position", params={"group": "stations"})
    assert resp.status_code == 404


@pytest.mark.network
def test_conjunctions_endpoint_returns_report_shape():
    resp = client.get("/api/conjunctions", params={"group": "stations", "hours": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "entries" in data
    assert "objects_screened" in data
    assert data["objects_screened"] > 0


@pytest.mark.network
def test_statistics_endpoint_returns_report_shape():
    resp = client.get("/api/statistics", params={"group": "stations"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_count"] > 0
    assert "count_by_regime" in data


def test_launch_windows_endpoint():
    resp = client.post("/api/launch-windows", json={
        "site": "cape_canaveral",
        "altitude_km": 400.0,
        "inclination_deg": 51.6,
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-02T00:00:00Z",
        "check_space_weather": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["site"]["name"] == "Cape Canaveral SLC-40"
    assert len(data["assessments"]) >= 1
    assert data["assessments"][0]["window"]["feasible"] is True


def test_launch_windows_rejects_unknown_site():
    resp = client.post("/api/launch-windows", json={
        "site": "not-a-real-site",
        "altitude_km": 400.0,
        "inclination_deg": 51.6,
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-02T00:00:00Z",
    })
    assert resp.status_code == 400


def test_launch_sites_endpoint():
    resp = client.get("/api/launch-sites")
    assert resp.status_code == 200
    data = resp.json()
    assert "cape_canaveral" in data
    assert "vandenberg" in data


@pytest.mark.network
def test_neo_risk_list_endpoint():
    resp = client.get("/api/neo/risk-list")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] > 0


@pytest.mark.network
def test_space_weather_endpoint():
    resp = client.get("/api/space-weather")
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["kp_index"] <= 9.0
    assert data["f107_flux"] > 0


@pytest.mark.network
def test_dashboard_snapshot_endpoint():
    resp = client.get("/api/dashboard-snapshot", params={"frames": 2, "active_limit": 10})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["frame_times_iso"]) == 2
    assert len(data["satellites"]) > 0


@pytest.mark.network
def test_ws_positions_streams_at_least_one_frame():
    with client.websocket_connect("/ws/positions?group=stations&interval_seconds=0.1") as ws:
        message = ws.receive_json()
        assert "epoch" in message
        assert "positions" in message
        assert len(message["positions"]) > 0
        pos = message["positions"][0]
        assert "lat_deg" in pos and "lon_deg" in pos and "alt_km" in pos


def test_ws_positions_rejects_unknown_group():
    with client.websocket_connect("/ws/positions?group=not-a-real-group") as ws:
        message = ws.receive_json()
        assert "error" in message
