from unittest.mock import patch

import pytest
import requests
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


def test_cola_scan_returns_clean_503_when_celestrak_unreachable():
    """
    Regression test for the exact bug the user hit: CelesTrak being
    unreachable used to bubble up as an opaque, unhandled 500. It must
    now come back as a 503 with an explanatory `detail` message the
    frontend can display directly.
    """
    with patch("app.tle_manager.fetch_group", side_effect=requests.ConnectTimeout("simulated outage")):
        resp = client.post("/api/cola/scan", json={
            "site": "cape_canaveral",
            "target_inclination_deg": 51.6,
            "search_start": "2026-01-01T00:00:00Z",
            "search_end": "2026-01-01T00:10:00Z",
        })
    assert resp.status_code == 503
    assert "unreachable" in resp.json()["detail"].lower()


def test_cola_scan_degrades_gracefully_when_one_group_fails_but_another_succeeds():
    """
    Regression test: a COLA scan requesting ['stations', 'visual'] should
    still return real results using whatever groups succeeded, rather
    than failing the whole request just because one group (e.g. 'visual',
    which has no bundled seed) is unavailable during an outage.
    """
    from datetime import datetime, timezone

    from core.tle_manager import Satellite

    fake_sat = Satellite(
        norad_id=25544, name="ISS (ZARYA)", line1=(
            "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9008"
        ), line2=(
            "2 25544  51.6416 339.9500 0001177  93.3861 264.7930 15.49560752 42982"
        ), epoch=datetime(2024, 1, 1, tzinfo=timezone.utc),
        inclination_deg=51.6416, eccentricity=0.0001177, mean_motion_rev_per_day=15.4956,
        semi_major_axis_km=6796.0, classification="stations",
    )

    def flaky_fetch(group, *args, **kwargs):
        if group == "visual":
            raise requests.ConnectTimeout("simulated outage for visual only")
        return [fake_sat]

    with patch("app.tle_manager.fetch_group", side_effect=flaky_fetch):
        resp = client.post("/api/cola/scan", json={
            "site": "cape_canaveral",
            "target_inclination_deg": 51.6,
            "search_start": "2026-01-01T00:00:00Z",
            "search_end": "2026-01-01T00:10:00Z",
            "catalog_groups": ["stations", "visual"],
        })
    assert resp.status_code == 200
    assert resp.json()["objects_screened"] == 1


def test_cola_scan_fails_only_when_every_group_fails():
    with patch("app.tle_manager.fetch_group", side_effect=requests.ConnectTimeout("simulated total outage")):
        resp = client.post("/api/cola/scan", json={
            "site": "cape_canaveral",
            "target_inclination_deg": 51.6,
            "search_start": "2026-01-01T00:00:00Z",
            "search_end": "2026-01-01T00:10:00Z",
            "catalog_groups": ["stations", "visual"],
        })
    assert resp.status_code == 503


def test_list_satellites_returns_clean_503_when_celestrak_unreachable():
    with patch("app.tle_manager.fetch_group", side_effect=requests.ConnectTimeout("simulated outage")):
        resp = client.get("/api/satellites", params={"group": "stations"})
    assert resp.status_code == 503
    assert "detail" in resp.json()


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


@pytest.mark.network
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


@pytest.mark.network
def test_cola_scan_endpoint_returns_candidates():
    resp = client.post("/api/cola/scan", json={
        "site": "cape_canaveral",
        "target_inclination_deg": 51.6,
        "target_altitude_km": 400.0,
        "search_start": "2026-01-01T00:00:00Z",
        "search_end": "2026-01-01T00:20:00Z",
        "candidate_step_minutes": 5,
        "bubble_radius_km": 50.0,
        "catalog_groups": ["stations"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["objects_screened"] > 0
    assert len(data["candidates"]) == 5  # 0,5,10,15,20 min
    for c in data["candidates"]:
        assert isinstance(c["clear"], bool)
        assert c["azimuth_deg"] > 0


def test_cola_scan_rejects_unknown_site():
    resp = client.post("/api/cola/scan", json={
        "site": "not-a-real-site",
        "target_inclination_deg": 51.6,
        "search_start": "2026-01-01T00:00:00Z",
        "search_end": "2026-01-01T00:10:00Z",
    })
    assert resp.status_code == 400


@pytest.mark.network
def test_cola_trajectory_endpoint_returns_full_path():
    resp = client.post("/api/cola/trajectory", json={
        "site": "cape_canaveral",
        "target_inclination_deg": 51.6,
        "target_altitude_km": 400.0,
        "launch_time": "2026-01-01T00:00:00Z",
        "bubble_radius_km": 50.0,
        "catalog_groups": ["stations"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["trajectory"]) > 0
    first, last = data["trajectory"][0], data["trajectory"][-1]
    assert first["alt_km"] < last["alt_km"]
    assert "closest_distance_km" in first


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


# ---------------------------------------------------------------------------
# CV module
# ---------------------------------------------------------------------------

def test_cv_page_serves_html():
    resp = client.get("/cv")
    assert resp.status_code == 200
    assert "ISS CV MODULE" in resp.text


def test_cv_static_assets_resolve():
    for asset in ("cv.js",):
        resp = client.get(f"/{asset}")
        assert resp.status_code == 200


def test_cv_upload_rejects_non_video_content_type():
    resp = client.post(
        "/api/cv/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400


def test_cv_upload_accepts_video_file(tmp_path):
    fake_video_bytes = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100  # not a real mp4, just exercises the upload path
    resp = client.post(
        "/api/cv/upload",
        files={"file": ("test_clip.mp4", fake_video_bytes, "video/mp4")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "test_clip.mp4"
    assert data["size_bytes"] == len(fake_video_bytes)


@pytest.mark.network
@pytest.mark.slow
def test_ws_cv_iss_live_source_streams_real_frame_when_available():
    """
    Real end-to-end check against NASA's actual live feed. If it happens
    to be unreachable at test time, the endpoint must still work by
    disclosing a fallback notice and continuing with the sample clip —
    both outcomes are asserted as acceptable, since NASA's feed uptime is
    outside this project's control (see cv/iss_live.py's docstring).
    """
    with client.websocket_connect("/ws/cv?source=iss_live&target_fps=1000") as ws:
        first = ws.receive()
        if "text" in first:
            # A fallback notice arrived before the first frame.
            import json
            payload = json.loads(first["text"])
            assert "notice" in payload
            frame_bytes = ws.receive_bytes()
        else:
            frame_bytes = first["bytes"]

        assert frame_bytes[:2] == b"\xff\xd8"  # JPEG magic bytes
        meta = ws.receive_json()
        assert "detections" in meta or "notice" in meta


def test_ws_cv_iss_live_falls_back_to_sample_when_unreachable():
    """Regression test: an unreachable ISS live feed must degrade to the sample clip with a disclosed notice, not fail silently or crash."""
    with patch("app.resolve_iss_stream_url", side_effect=RuntimeError("simulated: stream not live")):
        with client.websocket_connect("/ws/cv?source=iss_live&target_fps=1000") as ws:
            notice_msg = ws.receive_json()
            assert "notice" in notice_msg
            assert "sample" in notice_msg["notice"].lower()

            frame_bytes = ws.receive_bytes()
            assert frame_bytes[:2] == b"\xff\xd8"


@pytest.mark.slow
def test_ws_cv_sample_source_streams_real_annotated_frame():
    """
    Full pipeline over the WebSocket, real components: builds the sample
    clip from real Ultralytics imagery on first use, runs real YOLOv8
    detection + HUD annotation, and streams back a real JPEG frame
    followed by a JSON detections/metrics message.
    """
    with client.websocket_connect("/ws/cv?source=sample&target_fps=1000") as ws:
        frame_bytes = ws.receive_bytes()
        assert len(frame_bytes) > 0
        assert frame_bytes[:2] == b"\xff\xd8"  # JPEG magic bytes

        meta = ws.receive_json()
        assert "detections" in meta
        assert "fps" in meta
        assert "frame_index" in meta


def test_ws_cv_upload_source_without_upload_returns_error():
    with client.websocket_connect("/ws/cv?source=upload") as ws:
        message = ws.receive_json()
        assert "error" in message
