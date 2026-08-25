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
from viz.dashboard_3d import add_ground_tracks, build_snapshot, snapshot_to_dict

app = FastAPI(title="COSMIC OBSERVER", version="1.0.0")

tle_manager = TLEManager()

STATIC_DIR = __file__.rsplit("/", 1)[0] + "/static"


# ---------------------------------------------------------------------------
# Satellites
# ---------------------------------------------------------------------------

@app.get("/api/satellites")
def list_satellites(group: str = "stations", limit: int = 200):
    """List satellites in a CelesTrak group (cached; see core.tle_manager.GROUPS for valid names)."""
    if group not in GROUPS:
        raise HTTPException(400, f"Unknown group '{group}'. Valid groups: {list(GROUPS)}")
    satellites = tle_manager.fetch_group(group)[:limit]
    return {"group": group, "count": len(satellites), "satellites": satellites}


@app.get("/api/satellites/{norad_id}/position")
def satellite_position(norad_id: int, group: str = "stations"):
    """Current propagated position (lat/lon/alt) of a single satellite."""
    if group not in GROUPS:
        raise HTTPException(400, f"Unknown group '{group}'. Valid groups: {list(GROUPS)}")
    satellites = tle_manager.fetch_group(group)
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
    if group not in GROUPS:
        raise HTTPException(400, f"Unknown group '{group}'. Valid groups: {list(GROUPS)}")
    satellites = tle_manager.fetch_group(group)
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
    if group not in GROUPS:
        raise HTTPException(400, f"Unknown group '{group}'. Valid groups: {list(GROUPS)}")
    satellites = tle_manager.fetch_group(group)[:limit]
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
# NEOs / space weather
# ---------------------------------------------------------------------------

@app.get("/api/neo/risk-list")
def neo_risk_list(min_torino: int = 0):
    client = CNEOSClient()
    objects = client.objects_above_torino(min_torino) if min_torino > 0 else client.fetch_risk_list()
    return {"count": len(objects), "objects": objects}


@app.get("/api/space-weather")
def space_weather():
    return fetch_current_snapshot()


# ---------------------------------------------------------------------------
# Dashboard snapshot (same shape scripts/generate_dashboard_snapshot.py writes to disk)
# ---------------------------------------------------------------------------

@app.get("/api/dashboard-snapshot")
def dashboard_snapshot(active_group: str = "visual", active_limit: int = 200, frames: int = 12, step_hours: float = 1.0):
    if active_group not in GROUPS:
        raise HTTPException(400, f"Unknown group '{active_group}'. Valid groups: {list(GROUPS)}")

    stations = tle_manager.fetch_group("stations")
    active = tle_manager.fetch_group(active_group)[:active_limit]

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
            satellites = tle_manager.fetch_group(group)
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
