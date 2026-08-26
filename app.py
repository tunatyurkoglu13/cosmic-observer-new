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
from datetime import datetime, timedelta, timezone

import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.launch_window import LAUNCH_SITES, LaunchTarget
from core.propagator import Propagator
from core.tle_manager import GROUPS, TLEManager
from data.nasa_cneos import CNEOSClient
from data.space_weather import fetch_current_snapshot
from reports.launch_analysis import generate_launch_analysis
from reports.risk_report import generate_risk_report
from reports.statistics import compute_population_statistics
from stm.cola import build_catalog_grid, find_cola_launch_windows, screen_launch_time
from viz.dashboard_3d import add_ground_tracks, build_snapshot, snapshot_to_dict

app = FastAPI(title="COSMIC OBSERVER", version="1.0.0")

tle_manager = TLEManager()

STATIC_DIR = __file__.rsplit("/", 1)[0] + "/static"


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


@app.get("/api/space-weather")
def space_weather():
    try:
        return fetch_current_snapshot()
    except requests.RequestException:
        raise HTTPException(503, "NOAA space weather feed is currently unreachable — external service outage, try again shortly.")


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
# Static dashboard hosting — mounted LAST so it acts as a catch-all fallback
# and never shadows the /api/* and /ws/* routes above. html=True serves
# index.html for "/" and lets the page's relative asset paths (retro.css,
# dashboard.js, snapshot.json) resolve at their natural root-relative URLs.
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
