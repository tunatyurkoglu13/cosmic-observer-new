"""
app.py — COSMIC OBSERVER main application: FastAPI REST API + WebSocket
live-position streaming + static dashboard hosting.

Ties together every prior phase:
  - core/       orbital mechanics (TLE fetch/cache, SGP4 propagation, collision Pc)
  - data/       external feeds (NASA Sentry NEOs, NOAA/DONKI space weather)
  - stm/        conjunction screening, keep-out zones, debris density
  - core.launch_window  launch site/window/Hohmann-transfer calculations
  - viz/        3D dashboard data layer (served as static/ + this API)
  - reports/    risk, population-statistics, and launch-analysis reports

Run with:
    uvicorn app:app --reload
"""

from __future__ import annotations

import asyncio
import base64
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import cv2
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.launch_window import LAUNCH_SITES, LaunchTarget
from core.propagator import Propagator
from core.tle_manager import GROUPS, TLEManager
from cv.fits_ingest import is_fits_path, load_fits_image
from cv.hud import draw_label, draw_reticle
from cv.iss_live import resolve_iss_stream_url
from cv.open_vocabulary import DEFAULT_SPACE_QUERIES, OpenVocabularyDetector
from cv.streak_detection import draw_streaks, detect_streaks
from cv.streamer import FrameProcessor, build_sample_video, encode_jpeg
from data.dscovr_epic import EPICClient
from data.dsn import DSNClient
from data.mars_rover_photos import ROVERS as MARS_ROVERS, MarsRoverPhotosClient
from data.nasa_cneos import CNEOSClient
from data.neows import NeoWsClient
from data.sbdb import SBDBClient
from data.small_bodies import CURATED_BODIES as SMALL_BODIES, SmallBodyTracker
from data.solar_system import BODIES as SOLAR_SYSTEM_BODIES, MOONS as SOLAR_SYSTEM_MOONS, SolarSystemClient
from data.space_telescopes import TELESCOPES, SpaceTelescopeClient
from data.space_weather import fetch_current_snapshot
from reports.launch_analysis import generate_launch_analysis
from reports.risk_report import generate_risk_report
from reports.statistics import compute_population_statistics
from stm.cola import build_catalog_grid, find_cola_launch_windows, screen_launch_time
from viz.dashboard_3d import add_ground_tracks, build_snapshot, snapshot_to_dict

app = FastAPI(title="COSMIC OBSERVER", version="1.0.0")

tle_manager = TLEManager()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = str(BASE_DIR / "static")
CV_UPLOAD_DIR = BASE_DIR / "data" / "cv_uploads"


def _safe_fetch_group(group: str, limit: int | None = None) -> list:
    """
    tle_manager.fetch_group(), but converted into clean HTTP errors instead
    of letting a network exception surface as an opaque 500.

    CelesTrak being temporarily unreachable/rate-limited is a real,
    recurring condition in practice (not a hypothetical) — this happened
    repeatedly during this project's own development. TLEManager already
    falls back to stale cached data when possible; this only raises when
    there's truly nothing usable (first-ever request for a group, with
    the source down).
    """
    if group not in GROUPS:
        raise HTTPException(400, f"Unknown group '{group}'. Valid groups: {list(GROUPS)}")
    try:
        satellites = tle_manager.fetch_group(group)
    except requests.RequestException:
        raise HTTPException(
            503,
            f"CelesTrak (TLE data source) is currently unreachable and no cached data is "
            f"available yet for group '{group}'. This is an external service outage, not an "
            f"application bug — please try again in a moment.",
        )
    except RuntimeError as e:
        raise HTTPException(502, f"CelesTrak returned an unexpected response for group '{group}': {e}")
    return satellites[:limit] if limit is not None else satellites


# ---------------------------------------------------------------------------
# Satellites
# ---------------------------------------------------------------------------

@app.get("/api/satellites")
def list_satellites(group: str = "stations", limit: int = 200):
    """List satellites in a CelesTrak group (cached; see core.tle_manager.GROUPS for valid names)."""
    satellites = _safe_fetch_group(group, limit)
    return {"group": group, "count": len(satellites), "satellites": satellites}


@app.get("/api/satellites/{norad_id}/position")
def satellite_position(norad_id: int, group: str = "stations"):
    """Current propagated position (lat/lon/alt) of a single satellite."""
    satellites = _safe_fetch_group(group)
    sat = next((s for s in satellites if s.norad_id == norad_id), None)
    if sat is None:
        raise HTTPException(404, f"NORAD ID {norad_id} not found in group '{group}'")

    prop = Propagator(sat.line1, sat.line2, name=sat.name)
    try:
        state = prop.propagate(datetime.now(timezone.utc))
    except RuntimeError as e:
        raise HTTPException(422, f"SGP4 propagation failed: {e}")

    return {
        "norad_id": sat.norad_id, "name": sat.name,
        "lat_deg": state.lat_deg, "lon_deg": state.lon_deg, "alt_km": state.alt_km,
        "epoch_jd": state.epoch_jd,
    }


# ---------------------------------------------------------------------------
# Conjunctions / risk
# ---------------------------------------------------------------------------

@app.get("/api/conjunctions")
def conjunctions(group: str = "stations", hours: float = 6.0):
    """Run the collision-risk report over a group for the next `hours`."""
    satellites = _safe_fetch_group(group)
    start = datetime.now(timezone.utc)
    end = start + timedelta(hours=hours)
    report = generate_risk_report(satellites, start, end)
    return report


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@app.get("/api/statistics")
def statistics(group: str = "active", limit: int = 2000):
    """Population statistics (altitude/inclination distribution, regime counts) for a group."""
    satellites = _safe_fetch_group(group, limit)
    return compute_population_statistics(satellites)


# ---------------------------------------------------------------------------
# Launch windows
# ---------------------------------------------------------------------------

class LaunchWindowRequest(BaseModel):
    site: str
    altitude_km: float
    inclination_deg: float = 0.0
    raan_deg: float | None = None
    sun_synchronous: bool = False
    start: datetime
    end: datetime
    direction: str = "northeast"
    check_space_weather: bool = True


@app.post("/api/launch-windows")
def launch_windows(req: LaunchWindowRequest):
    if req.site not in LAUNCH_SITES:
        raise HTTPException(400, f"Unknown site '{req.site}'. Valid sites: {list(LAUNCH_SITES)}")

    site = LAUNCH_SITES[req.site]
    target = LaunchTarget(
        altitude_km=req.altitude_km, inclination_deg=req.inclination_deg,
        raan_deg=req.raan_deg, sun_synchronous=req.sun_synchronous,
    )
    report = generate_launch_analysis(
        site, target, req.start, req.end,
        direction=req.direction, check_space_weather=req.check_space_weather,
    )
    return report


@app.get("/api/launch-sites")
def launch_sites():
    return LAUNCH_SITES


# ---------------------------------------------------------------------------
# COLA (Collision On Launch Assessment)
# ---------------------------------------------------------------------------

class COLAScanRequest(BaseModel):
    site: str
    target_inclination_deg: float
    target_altitude_km: float = 400.0
    search_start: datetime
    search_end: datetime
    candidate_step_minutes: float = 1.0
    ascent_duration_s: float = 600.0
    max_downrange_km: float = 2000.0
    bubble_radius_km: float = 50.0
    direction: str = "northeast"
    catalog_groups: list[str] = ["stations", "visual"]
    catalog_limit_per_group: int = 150


def _fetch_cola_catalog(groups: list[str], limit_per_group: int):
    """
    Fetch and merge several catalog groups, tolerating a partial failure:
    if one group has neither live data, cache, nor a bundled seed (e.g.
    CelesTrak is down AND that particular group was never seeded), skip
    it and continue with whatever groups did succeed, rather than failing
    the whole COLA scan over one missing group. Only raise if EVERY group
    failed, since a scan against zero objects isn't a real screen.
    """
    catalog = []
    errors = []
    for group in groups:
        try:
            catalog.extend(_safe_fetch_group(group, limit_per_group))
        except HTTPException as e:
            if e.status_code == 400:
                raise  # bad group name is a caller mistake, not a service outage to degrade past
            errors.append(f"{group}: {e.detail}")

    if not catalog and errors:
        raise HTTPException(503, "No catalog data available for any requested group: " + "; ".join(errors))

    return catalog


@app.post("/api/cola/scan")
def cola_scan(req: COLAScanRequest):
    """
    Run a full COLA scan: for every candidate launch time in
    [search_start, search_end], check whether the ascent trajectory's
    moving safety bubble ever comes within bubble_radius_km of a real
    tracked catalog object. Returns a summary per candidate (NOT the full
    trajectory — see /api/cola/trajectory for drawing one specific candidate).
    """
    if req.site not in LAUNCH_SITES:
        raise HTTPException(400, f"Unknown site '{req.site}'. Valid sites: {list(LAUNCH_SITES)}")

    site = LAUNCH_SITES[req.site]
    catalog = _fetch_cola_catalog(req.catalog_groups, req.catalog_limit_per_group)

    results = find_cola_launch_windows(
        site, req.target_inclination_deg, catalog,
        search_start=req.search_start, search_end=req.search_end,
        candidate_step=timedelta(minutes=req.candidate_step_minutes),
        target_altitude_km=req.target_altitude_km, ascent_duration_s=req.ascent_duration_s,
        max_downrange_km=req.max_downrange_km, bubble_radius_km=req.bubble_radius_km,
        direction=req.direction,
    )

    return {
        "site": site,
        "objects_screened": len(catalog),
        "bubble_radius_km": req.bubble_radius_km,
        "candidates": [
            {
                "launch_time": r.launch_time.isoformat(),
                "azimuth_deg": r.azimuth_deg,
                "clear": r.clear,
                "closest_approach_km": r.closest_approach_km,
                "violations": [
                    {
                        "t_offset_s": v.t_offset_s,
                        "absolute_time": v.absolute_time.isoformat(),
                        "satellite_name": v.satellite.name,
                        "norad_id": v.satellite.norad_id,
                        "distance_km": v.distance_km,
                    }
                    for v in r.violations
                ],
            }
            for r in results
        ],
    }


class COLATrajectoryRequest(BaseModel):
    site: str
    target_inclination_deg: float
    target_altitude_km: float = 400.0
    launch_time: datetime
    ascent_duration_s: float = 600.0
    max_downrange_km: float = 2000.0
    bubble_radius_km: float = 50.0
    direction: str = "northeast"
    catalog_groups: list[str] = ["stations", "visual"]
    catalog_limit_per_group: int = 150


@app.post("/api/cola/trajectory")
def cola_trajectory(req: COLATrajectoryRequest):
    """
    Full ascent-path detail for ONE specific candidate launch time: every
    trajectory sample's lat/lon/alt plus its closest-catalog-object
    distance, for 3D visualization (color the path by how close each
    point came to a tracked object).
    """
    if req.site not in LAUNCH_SITES:
        raise HTTPException(400, f"Unknown site '{req.site}'. Valid sites: {list(LAUNCH_SITES)}")

    site = LAUNCH_SITES[req.site]
    catalog_sats = _fetch_cola_catalog(req.catalog_groups, req.catalog_limit_per_group)

    grid_end = req.launch_time + timedelta(seconds=req.ascent_duration_s + 30)
    catalog_grid = build_catalog_grid(catalog_sats, req.launch_time, grid_end)

    result = screen_launch_time(
        site, req.target_inclination_deg, req.launch_time, catalog_grid,
        target_altitude_km=req.target_altitude_km, ascent_duration_s=req.ascent_duration_s,
        max_downrange_km=req.max_downrange_km, bubble_radius_km=req.bubble_radius_km,
        direction=req.direction,
    )

    return {
        "launch_time": result.launch_time.isoformat(),
        "azimuth_deg": result.azimuth_deg,
        "clear": result.clear,
        "closest_approach_km": result.closest_approach_km,
        "bubble_radius_km": req.bubble_radius_km,
        "trajectory": [
            {
                "t_offset_s": s.t_offset_s,
                "lat_deg": s.lat_deg,
                "lon_deg": s.lon_deg,
                "alt_km": s.alt_km,
                "closest_distance_km": d,
            }
            for s, d in zip(result.trajectory, result.sample_min_distances_km)
        ],
        "violations": [
            {
                "t_offset_s": v.t_offset_s,
                "satellite_name": v.satellite.name,
                "norad_id": v.satellite.norad_id,
                "distance_km": v.distance_km,
            }
            for v in result.violations
        ],
    }


# ---------------------------------------------------------------------------
# NEOs / space weather
# ---------------------------------------------------------------------------

@app.get("/api/neo/risk-list")
def neo_risk_list(min_torino: int = 0):
    client = CNEOSClient()
    try:
        objects = client.objects_above_torino(min_torino) if min_torino > 0 else client.fetch_risk_list()
    except requests.RequestException:
        raise HTTPException(503, "NASA JPL Sentry API is currently unreachable — external service outage, try again shortly.")
    return {"count": len(objects), "objects": objects}


@app.get("/api/asteroid/{designation}")
def asteroid_lookup(designation: str):
    """Orbital elements + hazard classification for one asteroid/comet by designation or name (JPL SBDB)."""
    client = SBDBClient()
    try:
        body = client.lookup(designation)
    except requests.RequestException:
        raise HTTPException(503, "JPL Small-Body Database is currently unreachable — external service outage, try again shortly.")
    except ValueError as e:
        raise HTTPException(404, str(e))
    return body


@app.get("/api/neo/close-approach")
def neo_close_approach(start_date: str | None = None, end_date: str | None = None):
    """
    Near-Earth objects with a catalogued close approach in the given date
    range (NASA NeoWs; at most 7 days per call). Defaults to today.
    """
    start = date.fromisoformat(start_date) if start_date else date.today()
    end = date.fromisoformat(end_date) if end_date else start

    client = NeoWsClient()
    try:
        objects = client.feed(start, end)
    except requests.RequestException:
        raise HTTPException(503, "NASA NeoWs API is currently unreachable — external service outage, try again shortly.")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"count": len(objects), "objects": objects}


@app.get("/api/solar-system/bodies")
def solar_system_bodies():
    """Static metadata (real radius, display color) for bodies the 3D dashboard can travel to."""
    return SOLAR_SYSTEM_BODIES


@app.get("/api/solar-system/{body}/position")
def solar_system_position(body: str):
    """
    Real-time Earth-centered direction + distance to `body`, from JPL
    Horizons (see data.solar_system) — used by the 3D dashboard to fly
    the camera toward the body along its true current direction.
    """
    client = SolarSystemClient()
    try:
        position = client.fetch_position(body)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except requests.RequestException:
        raise HTTPException(503, "JPL Horizons is currently unreachable — external service outage, try again shortly.")
    return position


@app.get("/api/solar-system/moons")
def solar_system_moons():
    """Static metadata (real radius, display color, parent planet) for Mars's/Earth's real moons."""
    return SOLAR_SYSTEM_MOONS


@app.get("/api/solar-system/moons/{key}/position")
def solar_system_moon_position(key: str):
    """Real-time position of `key` relative to its own parent planet (not Earth) — see data.solar_system."""
    client = SolarSystemClient()
    try:
        position = client.fetch_moon_position(key)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except requests.RequestException:
        raise HTTPException(503, "JPL Horizons is currently unreachable — external service outage, try again shortly.")
    return position


@app.get("/api/small-bodies")
def small_bodies_list():
    """Metadata for the curated real asteroids/comets the 3D dashboard can plot (see data.small_bodies)."""
    return SMALL_BODIES


@app.get("/api/small-bodies/{key}/position")
def small_body_position(key: str):
    """
    Real-time Earth-relative position of one curated small body, from its
    real JPL SBDB orbital elements propagated to now (see
    data.small_bodies — two-body Kepler propagation, not a full
    perturbed ephemeris).
    """
    try:
        sun_client = SolarSystemClient()
        sun_position = sun_client.fetch_position("sun")
        earth_helio_r_km = tuple(-c for c in sun_position.r_km)

        tracker = SmallBodyTracker()
        state = tracker.current_position(key, earth_helio_r_km)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except requests.RequestException:
        raise HTTPException(503, "JPL Horizons/SBDB is currently unreachable — external service outage, try again shortly.")
    return state


@app.get("/api/space-weather")
def space_weather():
    try:
        return fetch_current_snapshot()
    except requests.RequestException:
        raise HTTPException(503, "NOAA space weather feed is currently unreachable — external service outage, try again shortly.")


# ---------------------------------------------------------------------------
# Sensor feeds — real live/near-live telemetry and imagery from NASA/JPL/STScI
# instruments, for the SENSOR FEEDS panel (static/cv.html). Each client
# follows the same core.resilient_fetch pattern as everything else in this
# project — real data with resilient caching, never fabricated fallback data.
# ---------------------------------------------------------------------------

@app.get("/api/sensors/dsn")
def sensors_dsn():
    """Live NASA Deep Space Network status (data.dsn) — which spacecraft each dish is tracking, signal rate, range."""
    client = DSNClient()
    try:
        status = client.fetch_status()
    except requests.RequestException:
        raise HTTPException(503, "DSN Now feed is currently unreachable — external service outage, try again shortly.")
    return {
        "fetched_at": status.fetched_at,
        "station_names": status.station_names,
        "dishes": status.dishes,
        "active_spacecraft": status.active_spacecraft,  # a @property, not a dataclass field — computed here so it actually appears in the JSON
    }


@app.get("/api/sensors/telescopes/{telescope}")
def sensors_telescope(telescope: str):
    """Most recently archived real Hubble/JWST observation (data.space_telescopes, via MAST)."""
    client = SpaceTelescopeClient()
    try:
        return client.fetch_latest_observation(telescope)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except requests.RequestException:
        raise HTTPException(503, "MAST archive is currently unreachable — external service outage, try again shortly.")


@app.get("/api/sensors/telescopes")
def sensors_telescopes_list():
    return TELESCOPES


@app.get("/api/sensors/earth-epic")
def sensors_earth_epic():
    """Latest real full-Earth photo from DSCOVR/EPIC (data.dscovr_epic)."""
    client = EPICClient()
    try:
        return client.fetch_latest()
    except requests.RequestException:
        raise HTTPException(503, "EPIC API is currently unreachable — external service outage, try again shortly.")


@app.get("/api/sensors/mars-rover/{rover}")
def sensors_mars_rover(rover: str):
    """Most recent real downlinked photos from a Mars rover (data.mars_rover_photos)."""
    client = MarsRoverPhotosClient()
    try:
        photos = client.fetch_latest(rover)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except requests.RequestException:
        raise HTTPException(503, "NASA Mars Rover Photos API is currently unreachable — external service outage, try again shortly.")
    return {"rover": rover, "count": len(photos), "photos": photos}


@app.get("/api/sensors/mars-rovers")
def sensors_mars_rovers_list():
    return list(MARS_ROVERS)


# ---------------------------------------------------------------------------
# Dashboard snapshot (same shape scripts/generate_dashboard_snapshot.py writes to disk)
# ---------------------------------------------------------------------------

@app.get("/api/dashboard-snapshot")
def dashboard_snapshot(active_group: str = "visual", active_limit: int = 200, frames: int = 12, step_hours: float = 1.0):
    stations = _safe_fetch_group("stations")
    active = _safe_fetch_group(active_group, active_limit)

    epoch = datetime.now(timezone.utc)
    snapshot = build_snapshot(stations + active, epoch, n_frames=frames, frame_step=timedelta(hours=step_hours))
    add_ground_tracks(snapshot, stations, epoch)
    return snapshot_to_dict(snapshot)


# ---------------------------------------------------------------------------
# WebSocket: live position streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/positions")
async def ws_positions(websocket: WebSocket, group: str = "stations", interval_seconds: float = 2.0):
    """
    Stream live propagated positions for a satellite group, one JSON
    message every `interval_seconds`, until the client disconnects.

    This is the real-time counterpart to /api/dashboard-snapshot's static
    multi-frame preview: each message here is "one frame," propagated at
    the moment it's sent, rather than precomputed.
    """
    await websocket.accept()

    if group not in GROUPS:
        await websocket.send_json({"error": f"Unknown group '{group}'"})
        await websocket.close()
        return

    try:
        while True:
            try:
                satellites = tle_manager.fetch_group(group)
            except (requests.RequestException, RuntimeError) as e:
                await websocket.send_json({"error": f"TLE source unavailable: {e.__class__.__name__}"})
                await asyncio.sleep(interval_seconds)
                continue

            now = datetime.now(timezone.utc)
            positions = []
            for sat in satellites:
                try:
                    prop = Propagator(sat.line1, sat.line2, name=sat.name)
                    state = prop.propagate(now)
                    positions.append({
                        "norad_id": sat.norad_id, "name": sat.name,
                        "classification": sat.classification,
                        "lat_deg": state.lat_deg, "lon_deg": state.lon_deg, "alt_km": state.alt_km,
                    })
                except RuntimeError:
                    continue

            await websocket.send_json({"epoch": now.isoformat(), "positions": positions})
            await asyncio.sleep(interval_seconds)
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# CV module — a separate page (/cv) with its own retro HUD-overlaid video
# pipeline (real YOLOv8 detection + cv.hud drawing from Phase 6/this
# phase), streamed over its own WebSocket rather than folded into the
# main dashboard's /ws/positions.
# ---------------------------------------------------------------------------

_cv_processor: FrameProcessor | None = None
_cv_upload_state: dict[str, Path] = {}


def _get_cv_processor() -> FrameProcessor:
    """Lazily instantiate the YOLO-backed processor once (loading weights is expensive; don't do it at import time)."""
    global _cv_processor
    if _cv_processor is None:
        _cv_processor = FrameProcessor()
    return _cv_processor


_open_vocab_detector: OpenVocabularyDetector | None = None


def _get_open_vocab_detector() -> OpenVocabularyDetector:
    """Lazily instantiate OWL-ViT once (its weights are a separate ~600MB download/load from YOLO's)."""
    global _open_vocab_detector
    if _open_vocab_detector is None:
        _open_vocab_detector = OpenVocabularyDetector()
    return _open_vocab_detector


@app.get("/cv")
def cv_page():
    return FileResponse(f"{STATIC_DIR}/cv.html")


@app.post("/api/cv/upload")
async def cv_upload(file: UploadFile = File(...)):
    """Upload a video file to use as the /ws/cv?source=upload frame source."""
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(400, f"Expected a video file, got content-type '{file.content_type}'")

    CV_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = CV_UPLOAD_DIR / (file.filename or "upload.mp4")
    contents = await file.read()
    dest.write_bytes(contents)
    _cv_upload_state["path"] = dest

    return {"filename": dest.name, "size_bytes": len(contents)}


@app.post("/api/cv/identify")
async def cv_identify(file: UploadFile = File(...), text_queries: str = Form("")):
    """
    One-shot "what is this image?" tool: runs classical streak detection
    (satellite/debris trails — cv.streak_detection) AND zero-shot
    open-vocabulary detection (cv.open_vocabulary's OWL-ViT, matched
    against `text_queries`) on a single uploaded image, and returns both
    results plus one HUD-annotated preview image.

    Unlike /ws/cv's real-time stream, this is a deliberately slower,
    one-off analysis endpoint — OWL-ViT's transformer forward pass runs
    at roughly 1-3 FPS on CPU, far below /ws/cv's target frame rate (see
    cv/open_vocabulary.py's module docstring for why it's kept separate).

    Args:
        file: an image file — FITS (.fits/.fit/.fts, real telescope data)
            or a standard format (JPEG/PNG/etc.).
        text_queries: comma-separated list of things to look for (e.g.
            "satellite, solar panel, asteroid"); if empty, uses
            cv.open_vocabulary.DEFAULT_SPACE_QUERIES.
    """
    suffix = Path(file.filename or "upload.jpg").suffix
    contents = await file.read()

    CV_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = CV_UPLOAD_DIR / f"identify_{int(datetime.now().timestamp())}{suffix}"
    temp_path.write_bytes(contents)

    try:
        if is_fits_path(temp_path):
            fits_image = await asyncio.to_thread(load_fits_image, temp_path)
            image, wcs = fits_image.data_bgr, fits_image.wcs
        else:
            image = await asyncio.to_thread(cv2.imread, str(temp_path))
            wcs = None
            if image is None:
                raise HTTPException(400, "Could not read image (not valid FITS or a format OpenCV supports)")
    finally:
        temp_path.unlink(missing_ok=True)

    queries = [q.strip() for q in text_queries.split(",") if q.strip()] or DEFAULT_SPACE_QUERIES

    streaks = await asyncio.to_thread(detect_streaks, image, wcs)
    # detect_streaks() is designed for sparse star-field astronomical
    # images; a busy real-world photo (cables, panel edges, equipment
    # frames) can legitimately match "long straight bright line" hundreds
    # of times over. Streaks are already sorted longest-first, so capping
    # here keeps the response sane without changing the detector itself
    # (which is correct for its actual intended input).
    MAX_STREAKS_IN_RESPONSE = 25
    streaks = streaks[:MAX_STREAKS_IN_RESPONSE]

    try:
        detector = _get_open_vocab_detector()
        zero_shot_detections = await asyncio.to_thread(detector.detect, image, queries)
    except Exception as e:
        # OWL-ViT weights failing to load/download shouldn't take down
        # streak detection, which has no such external dependency.
        zero_shot_detections = []
        zero_shot_error = str(e)
    else:
        zero_shot_error = None

    annotated = image.copy()
    draw_streaks(annotated, streaks)
    for det in zero_shot_detections:
        draw_reticle(annotated, det.box_xyxy, (0, 255, 255))
        draw_label(annotated, det.box_xyxy, f"{det.class_name.upper()} {det.confidence:.2f}", (0, 255, 255))

    jpeg_bytes = await asyncio.to_thread(encode_jpeg, annotated)

    return {
        "streaks": [
            {
                "start": s.start, "end": s.end, "length_px": s.length_px,
                "angle_deg": s.angle_deg, "mean_brightness": s.mean_brightness,
                "start_sky": s.start_sky, "end_sky": s.end_sky,
            }
            for s in streaks
        ],
        "detections": [
            {"class_name": d.class_name, "confidence": d.confidence, "box_xyxy": d.box_xyxy}
            for d in zero_shot_detections
        ],
        "zero_shot_error": zero_shot_error,
        "annotated_image_base64": base64.b64encode(jpeg_bytes).decode("ascii"),
        "has_wcs": wcs is not None,
    }


async def _open_video_capture(source: str) -> tuple[cv2.VideoCapture | None, str, str | None]:
    """
    Open a cv2.VideoCapture for the requested source.

    Returns (capture_or_None, effective_label, notice). `notice` is a
    human-readable string set when we fell back from what was actually
    requested (e.g. ISS live unreachable -> sample clip) — the caller
    sends this to the client so a fallback is disclosed, never silent.
    `capture_or_None` is None only when even the fallback failed.
    """
    if source == "upload":
        video_path = _cv_upload_state.get("path")
        if video_path is None or not video_path.exists():
            return None, "UPLOAD", "No video uploaded yet. POST /api/cv/upload first, or use source=sample."
        cap = await asyncio.to_thread(cv2.VideoCapture, str(video_path))
        if not cap.isOpened():
            return None, "UPLOAD", f"Could not open uploaded file: {video_path}"
        return cap, "UPLOAD", None

    if source == "iss_live":
        try:
            stream = await asyncio.to_thread(resolve_iss_stream_url)
        except RuntimeError as e:
            sample_path = await asyncio.to_thread(build_sample_video)
            cap = await asyncio.to_thread(cv2.VideoCapture, str(sample_path))
            return cap, "SAMPLE", f"ISS live stream unavailable ({e}) — showing sample clip instead."

        cap = await asyncio.to_thread(cv2.VideoCapture, stream.url)
        if not cap.isOpened():
            sample_path = await asyncio.to_thread(build_sample_video)
            cap = await asyncio.to_thread(cv2.VideoCapture, str(sample_path))
            return cap, "SAMPLE", "ISS live stream could not be opened — showing sample clip instead."
        return cap, "ISS LIVE", None

    sample_path = await asyncio.to_thread(build_sample_video)
    cap = await asyncio.to_thread(cv2.VideoCapture, str(sample_path))
    return cap, "SAMPLE", None


@app.websocket("/ws/cv")
async def ws_cv(websocket: WebSocket, source: str = "sample", target_fps: float = 8.0):
    """
    Stream HUD-annotated, YOLOv8-detected frames from a video source.

    Each iteration sends two WebSocket messages: a binary JPEG frame,
    then a JSON text message with that frame's detection list and
    rolling metrics (FPS, count, avg confidence) — kept separate rather
    than base64-embedding the image in the JSON, so the browser can hand
    the binary frame straight to an <img>/Blob URL without any decoding
    overhead on the hot path.

    Args:
        source: "sample" (bundled demo clip, built on first use from real
            Ultralytics sample imagery), "upload" (the most recently
            uploaded file via POST /api/cv/upload), or "iss_live" (NASA's
            real, currently-live ISS video feed, resolved via
            cv.iss_live — falls back to "sample" with a disclosed notice
            if the live feed is unreachable).
        target_fps: pacing cap for the stream.
    """
    await websocket.accept()

    cap, label, notice = await _open_video_capture(source)
    if cap is None:
        # Nothing playable at all (e.g. no upload yet) — this is a hard
        # failure, not a disclosed fallback, so it's an "error".
        await websocket.send_json({"error": notice})
        await websocket.close()
        return
    if notice:
        # We got a stream, but not the one requested (e.g. ISS live
        # unreachable -> sample clip) — disclose it, but keep streaming.
        await websocket.send_json({"notice": notice})

    processor = _get_cv_processor()
    interval_s = 1.0 / target_fps if target_fps > 0 else 0.1
    is_live = label == "ISS LIVE"

    try:
        while True:
            ok, frame = await asyncio.to_thread(cap.read)
            if not ok:
                if is_live:
                    # A live feed doesn't "end" the way a file does — a
                    # failed read means the signed URL likely expired or
                    # there was a transient network hiccup. Re-resolve
                    # and reopen rather than looping back to "frame 0"
                    # (which makes no sense for a live stream).
                    await asyncio.to_thread(cap.release)
                    try:
                        stream = await asyncio.to_thread(resolve_iss_stream_url, force=True)
                        cap = await asyncio.to_thread(cv2.VideoCapture, stream.url)
                        if not cap.isOpened():
                            raise RuntimeError("re-opened capture failed to open")
                        continue
                    except RuntimeError as e:
                        await websocket.send_json({"error": f"ISS live stream lost and could not reconnect: {e}"})
                        break
                else:
                    await asyncio.to_thread(cap.set, cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = await asyncio.to_thread(cap.read)
                    if not ok:
                        await websocket.send_json({"error": "Video ended and could not be replayed"})
                        break

            result = await asyncio.to_thread(processor.process, frame, label)
            jpeg_bytes = await asyncio.to_thread(encode_jpeg, result.frame_bgr)

            await websocket.send_bytes(jpeg_bytes)
            await websocket.send_json({
                "detections": [
                    {"class_name": d.class_name, "confidence": d.confidence, "box_xyxy": d.box_xyxy}
                    for d in result.detections
                ],
                "fps": result.metrics.fps,
                "frame_index": result.metrics.frame_index,
                "avg_confidence": result.metrics.avg_confidence,
            })
            await asyncio.sleep(interval_s)
    except WebSocketDisconnect:
        pass
    finally:
        await asyncio.to_thread(cap.release)


# ---------------------------------------------------------------------------
# Static dashboard hosting — mounted LAST so it acts as a catch-all fallback
# and never shadows the /api/* and /ws/* routes above. html=True serves
# index.html for "/" and lets the page's relative asset paths (retro.css,
# dashboard.js, snapshot.json) resolve at their natural root-relative URLs.
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
