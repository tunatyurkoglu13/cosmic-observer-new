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
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import cv2
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.alert_store import AlertStore
from core.conjunction_watch import screen_close_approaches
from core.launch_window import LAUNCH_SITES, LaunchTarget
from core.timeseries_store import TimeSeriesStore
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

# ---------------------------------------------------------------------------
# Temporal layer — a background task that periodically samples a curated
# set of real, already-existing metrics (Kp index, DSN link activity, NEO
# risk-list size) into core.timeseries_store, so the dashboard can show
# genuine trend history instead of only ever a live snapshot. Each
# metric's fetch failure is isolated (a NOAA/JPL hiccup skips just that
# one sample, not the whole loop) — same "degrade, don't crash" spirit as
# every resilient-fetch client elsewhere in this project.
# ---------------------------------------------------------------------------

_timeseries_store = TimeSeriesStore()
TIMESERIES_SAMPLE_INTERVAL_S = 120

METRIC_DEFINITIONS: dict[str, dict] = {
    "kp_index": {"display_name": "Kp Index (Geomagnetic Activity)", "unit": "", "color_hex": "#ffcc00"},
    "f107_flux": {"display_name": "F10.7 Solar Radio Flux", "unit": "sfu", "color_hex": "#ff9900"},
    "dsn_active_spacecraft": {"display_name": "DSN Active Spacecraft Links", "unit": "", "color_hex": "#00ffff"},
    "dsn_total_downlink_mbps": {"display_name": "DSN Total Downlink Rate", "unit": "Mb/s", "color_hex": "#00ff66"},
    "neo_risk_count": {"display_name": "NEO Sentry Risk-List Size", "unit": "objects", "color_hex": "#ff0066"},
    "neo_max_torino": {"display_name": "NEO Max Torino Scale", "unit": "", "color_hex": "#ff3355"},
    "cv_anomaly_score": {"display_name": "CV Live Anomaly Reconstruction Error", "unit": "", "color_hex": "#ff3355"},
}


async def _sample_metrics_once() -> None:
    """One real sampling pass across the curated metric set."""
    try:
        snapshot = await asyncio.to_thread(fetch_current_snapshot)
        _timeseries_store.record("kp_index", snapshot.kp_index)
        _timeseries_store.record("f107_flux", snapshot.f107_flux)
    except Exception:
        pass

    try:
        dsn_status = await asyncio.to_thread(DSNClient().fetch_status)
        active = sum(1 for d in dsn_status.dishes if any(s.active for s in d.signals))
        total_down_bps = sum(
            s.data_rate_bps for d in dsn_status.dishes for s in d.signals
            if s.direction == "down" and s.active
        )
        _timeseries_store.record("dsn_active_spacecraft", float(active))
        _timeseries_store.record("dsn_total_downlink_mbps", total_down_bps / 1e6)
    except Exception:
        pass

    try:
        objects = await asyncio.to_thread(CNEOSClient().fetch_risk_list)
        _timeseries_store.record("neo_risk_count", float(len(objects)))
        max_torino = max((o.torino_scale_max for o in objects), default=0)
        _timeseries_store.record("neo_max_torino", float(max_torino))
    except Exception:
        pass


async def _sample_metrics_loop() -> None:
    while True:
        await _sample_metrics_once()
        await asyncio.to_thread(_timeseries_store.prune)
        await asyncio.sleep(TIMESERIES_SAMPLE_INTERVAL_S)


# ---------------------------------------------------------------------------
# Active alert layer — periodically screens real conditions (SGP4-propagated
# close approaches to the ISS, JPL Sentry NEO risk ratings, NEO close
# approaches) and, for the live CV stream, real anomaly detections — and
# turns genuinely concerning ones into persisted, deduplicated AlertEvents,
# broadcast live to any connected /ws/alerts client. This is the one part
# of the project that's actively watching rather than only answering "what
# is the state right now" when asked.
# ---------------------------------------------------------------------------

_alert_store = AlertStore()
ALERT_SCAN_INTERVAL_S = 600  # 10 min — real orbital geometry/NEO catalogs don't move meaningfully faster than this
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL")  # optional; unset by default, no external calls unless the user configures their own

_alert_websockets: set[WebSocket] = set()


async def _broadcast_alert(event) -> None:
    """Push one new AlertEvent to every connected /ws/alerts client, and to the optional webhook."""
    payload = {
        "id": event.id, "category": event.category, "severity": event.severity,
        "title": event.title, "description": event.description,
        "timestamp": event.timestamp.isoformat(), "metadata": event.metadata,
    }
    dead = []
    for ws in list(_alert_websockets):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _alert_websockets.discard(ws)

    if ALERT_WEBHOOK_URL:
        try:
            await asyncio.to_thread(
                requests.post, ALERT_WEBHOOK_URL,
                json={"text": f"[{event.severity.upper()}] {event.title} — {event.description}", **payload},
                timeout=10,
            )
        except Exception:
            pass  # webhook delivery is best-effort; never let it break the alert pipeline


def _record_alert(**kwargs):
    """Sync helper (AlertStore is plain sqlite3) — call via asyncio.to_thread, then broadcast if it wasn't deduped."""
    return _alert_store.record(**kwargs)


async def _scan_conjunctions_once() -> None:
    """
    Real SGP4-propagated close-approach screening: the ISS against a
    bounded subset of the tracked debris/active catalog (same "subset of
    the full catalog" convention this project already uses elsewhere —
    stm.cola, viz.dashboard_3d) over the next 24 hours.
    """
    try:
        stations = await asyncio.to_thread(tle_manager.fetch_group, "stations")
        iss = next((s for s in stations if s.norad_id == 25544), None)
        if iss is None:
            return

        debris = await asyncio.to_thread(tle_manager.fetch_group, "debris")
        active = await asyncio.to_thread(tle_manager.fetch_group, "active")
        catalog = (debris + active)[:250]

        events = await asyncio.to_thread(
            screen_close_approaches, [iss], catalog, datetime.now(timezone.utc), 24.0,
        )
        for close_approach in events:
            other_name = close_approach.other.name.strip()
            recorded = await asyncio.to_thread(
                _record_alert,
                category="conjunction", severity=close_approach.severity,
                title=f"Close approach: ISS & {other_name}",
                description=(
                    f"{other_name} (NORAD {close_approach.other.norad_id}) is predicted (real SGP4-propagated "
                    f"TLE screening) to pass within {close_approach.min_distance_km:.2f} km of the ISS at "
                    f"{close_approach.time_of_closest_approach.isoformat()} UTC. No covariance data is available "
                    f"from TLEs alone, so this is a miss-distance screen, not a rigorous Pc."
                ),
                metadata={
                    "watched_norad_id": close_approach.watched.norad_id,
                    "other_norad_id": close_approach.other.norad_id,
                    "other_name": other_name,
                    "min_distance_km": close_approach.min_distance_km,
                    "time_of_closest_approach": close_approach.time_of_closest_approach.isoformat(),
                },
                dedup_key=f"conjunction:{close_approach.watched.norad_id}:{close_approach.other.norad_id}",
                cooldown_minutes=180.0,
            )
            if recorded is not None:
                await _broadcast_alert(recorded)
    except Exception:
        pass


async def _scan_neo_alerts_once() -> None:
    """Real NEO risk (JPL Sentry) + close-approach (NeoWs) monitoring."""
    try:
        risky = await asyncio.to_thread(CNEOSClient().objects_above_torino, 1)
        for obj in risky:
            recorded = await asyncio.to_thread(
                _record_alert,
                category="neo_risk", severity="critical" if obj.torino_scale_max >= 5 else "warning",
                title=f"NEO risk: {obj.full_name} (Torino {obj.torino_scale_max})",
                description=(
                    f"{obj.full_name} has a real JPL Sentry Torino Scale rating of "
                    f"{obj.torino_scale_max} ({obj.torino_description}) — cumulative impact probability "
                    f"{obj.impact_probability_cum:.2e}."
                ),
                metadata={
                    "designation": obj.designation, "torino": obj.torino_scale_max,
                    "palermo_cum": obj.palermo_scale_cum, "impact_probability_cum": obj.impact_probability_cum,
                },
                dedup_key=f"neo_risk:{obj.designation}",
                cooldown_minutes=1440.0,  # Sentry ratings don't change fast — once/day is plenty
            )
            if recorded is not None:
                await _broadcast_alert(recorded)
    except Exception:
        pass

    try:
        today = date.today()
        objects = await asyncio.to_thread(NeoWsClient().feed, today, today + timedelta(days=1))
        CLOSE_APPROACH_ALERT_LD = 2.0  # lunar distances — a real, commonly-used "notably close" amateur/professional threshold
        for obj in objects:
            for approach in obj.close_approaches:
                if approach.orbiting_body != "Earth" or approach.miss_distance_lunar > CLOSE_APPROACH_ALERT_LD:
                    continue
                recorded = await asyncio.to_thread(
                    _record_alert,
                    category="neo_close_approach",
                    severity="critical" if approach.miss_distance_lunar < 1.0 else "warning",
                    title=f"NEO close approach: {obj.name}",
                    description=(
                        f"{obj.name} passes Earth at {approach.miss_distance_lunar:.2f} lunar distances "
                        f"({approach.miss_distance_km:,.0f} km) on {approach.approach_date} "
                        f"(relative velocity {approach.relative_velocity_km_s:.1f} km/s)."
                        + (" Flagged by NASA as potentially hazardous." if obj.is_potentially_hazardous else "")
                    ),
                    metadata={
                        "neo_id": obj.neo_reference_id, "miss_distance_lunar": approach.miss_distance_lunar,
                        "miss_distance_km": approach.miss_distance_km, "approach_date": approach.approach_date,
                        "is_potentially_hazardous": obj.is_potentially_hazardous,
                    },
                    dedup_key=f"neo_close_approach:{obj.neo_reference_id}:{approach.approach_date}",
                    cooldown_minutes=1440.0,
                )
                if recorded is not None:
                    await _broadcast_alert(recorded)
    except Exception:
        pass


async def _alert_scan_loop() -> None:
    while True:
        await _scan_conjunctions_once()
        await _scan_neo_alerts_once()
        await asyncio.sleep(ALERT_SCAN_INTERVAL_S)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    sampler_task = asyncio.create_task(_sample_metrics_loop())
    alert_task = asyncio.create_task(_alert_scan_loop())
    yield
    sampler_task.cancel()
    alert_task.cancel()


app = FastAPI(title="COSMIC OBSERVER", version="1.0.0", lifespan=_lifespan)

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
# Temporal trends — real historical samples of the metrics above, recorded
# by the background sampler (see _sample_metrics_loop) and (for CV anomaly
# score) directly from the live /ws/cv stream. Powers static/timeline.html.
# ---------------------------------------------------------------------------

@app.get("/api/timeseries")
def timeseries_list():
    """Metadata (display name, unit, color) + latest real value for every tracked metric."""
    result = {}
    for metric, meta in METRIC_DEFINITIONS.items():
        latest = _timeseries_store.latest(metric)
        result[metric] = {
            **meta,
            "latest_value": latest.value if latest else None,
            "latest_timestamp": latest.timestamp.isoformat() if latest else None,
        }
    return result


@app.get("/api/timeseries/{metric}")
def timeseries_query(metric: str, hours: float = 24.0, limit: int = 2000):
    """Real historical samples for one metric, over the last `hours` (default 24)."""
    if metric not in METRIC_DEFINITIONS:
        raise HTTPException(404, f"Unknown metric '{metric}'. Supported: {list(METRIC_DEFINITIONS)}")
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    samples = _timeseries_store.query(metric, since=since, limit=limit)
    return {
        "metric": metric,
        "count": len(samples),
        "samples": [{"timestamp": s.timestamp.isoformat(), "value": s.value} for s in samples],
    }


@app.get("/timeline")
def timeline_page():
    return FileResponse(f"{STATIC_DIR}/timeline.html")


# ---------------------------------------------------------------------------
# Active alert layer API — real alert history + live push. Powers
# static/alerts.html and the alert badge on the main dashboard.
# ---------------------------------------------------------------------------

@app.get("/api/alerts")
def alerts_list(limit: int = 100, category: str | None = None, unacknowledged_only: bool = False):
    return {"alerts": _alert_store.query(limit=limit, category=category, unacknowledged_only=unacknowledged_only)}


@app.get("/api/alerts/unacknowledged-count")
def alerts_unacknowledged_count():
    return {"count": _alert_store.count_unacknowledged()}


@app.post("/api/alerts/{alert_id}/acknowledge")
def alerts_acknowledge(alert_id: int):
    if not _alert_store.acknowledge(alert_id):
        raise HTTPException(404, f"Alert {alert_id} not found")
    return {"acknowledged": True}


@app.get("/alerts")
def alerts_page():
    return FileResponse(f"{STATIC_DIR}/alerts.html")


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    """Live push of newly-recorded alerts — see _broadcast_alert. Clients don't need to send anything; this just keeps the connection open."""
    await websocket.accept()
    _alert_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _alert_websockets.discard(websocket)


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

# In-memory anomaly event log — a lightweight, deliberately simple
# substitute for a real time-series store (this project's overall scope
# doesn't currently warrant one; see the "next level" roadmap discussion
# this was requested alongside). Bounded so it can't grow unbounded over
# a long-running stream; resets on server restart, which is an honest
# limitation, not a hidden one.
_anomaly_log: deque[dict] = deque(maxlen=200)


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


@app.get("/api/cv/anomaly-status")
def cv_anomaly_status():
    """Whether a trained anomaly-detection model is currently loaded, and its deployed threshold."""
    processor = _get_cv_processor()
    if processor.anomaly_detector is None:
        return {"model_loaded": False, "threshold": None}
    return {"model_loaded": True, "threshold": processor.anomaly_detector.threshold}


@app.get("/api/cv/anomaly-log")
def cv_anomaly_log(limit: int = 50):
    """Recent real anomaly events flagged by the live stream (in-memory, resets on server restart)."""
    events = list(_anomaly_log)[-limit:]
    events.reverse()  # most recent first
    return {"count": len(events), "events": events}


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
    last_anomaly_sample_time = 0.0
    ANOMALY_SAMPLE_INTERVAL_S = 5.0  # throttled — a continuous per-frame trend, not one row per frame

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

            anomaly_payload = None
            if result.anomaly is not None:
                anomaly_payload = {
                    "is_anomaly": result.anomaly.is_anomaly,
                    "reconstruction_error": result.anomaly.reconstruction_error,
                    "threshold": result.anomaly.threshold,
                    "severity": result.anomaly.severity,
                }
                if result.anomaly.is_anomaly:
                    _anomaly_log.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": label,
                        "frame_index": result.metrics.frame_index,
                        "reconstruction_error": result.anomaly.reconstruction_error,
                        "threshold": result.anomaly.threshold,
                        "severity": result.anomaly.severity,
                    })
                    recorded = _alert_store.record(
                        category="anomaly",
                        severity="critical" if result.anomaly.severity >= 1.0 else "warning",
                        title=f"CV anomaly detected — {label}",
                        description=(
                            f"Live anomaly autoencoder flagged frame {result.metrics.frame_index} from {label}: "
                            f"reconstruction error {result.anomaly.reconstruction_error:.4f} vs threshold "
                            f"{result.anomaly.threshold:.4f} (severity {result.anomaly.severity:.2f})."
                        ),
                        metadata={
                            "source": label, "frame_index": result.metrics.frame_index,
                            "reconstruction_error": result.anomaly.reconstruction_error,
                            "threshold": result.anomaly.threshold, "severity": result.anomaly.severity,
                        },
                        dedup_key=f"anomaly:{label}",
                        cooldown_minutes=15.0,
                    )
                    if recorded is not None:
                        await _broadcast_alert(recorded)

                now_s = time.monotonic()
                if now_s - last_anomaly_sample_time >= ANOMALY_SAMPLE_INTERVAL_S:
                    last_anomaly_sample_time = now_s
                    _timeseries_store.record(
                        "cv_anomaly_score", result.anomaly.reconstruction_error,
                        metadata={"threshold": result.anomaly.threshold, "source": label},
                    )

            await websocket.send_bytes(jpeg_bytes)
            await websocket.send_json({
                "detections": [
                    {"class_name": d.class_name, "confidence": d.confidence, "box_xyxy": d.box_xyxy}
                    for d in result.detections
                ],
                "fps": result.metrics.fps,
                "frame_index": result.metrics.frame_index,
                "avg_confidence": result.metrics.avg_confidence,
                "anomaly": anomaly_payload,
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
