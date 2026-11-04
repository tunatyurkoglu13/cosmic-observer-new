"""
scripts/build_static_site.py — build the static GitHub Pages showcase.

The hosted demo has no server. Instead, this script runs the real FastAPI
app in-process (FastAPI TestClient — no network port is ever opened),
requests every read-only endpoint the frontend uses, and writes each
response to <out>/api/<path>.json next to a copy of static/.
pages/static-adapter.js, injected into every page, answers the frontend's
/api/* fetches and /ws/* sockets from those files, so the dashboard JS is
the same code that runs against the live backend.

State that has to accumulate across builds (the metric history behind
/timeline, the alert history behind /alerts, and a last-good copy of every
response for when an upstream API is briefly down) lives in --state-dir,
which .github/workflows/pages.yml persists between runs with actions/cache.

Usage:
    python scripts/build_static_site.py [--out _site] [--state-dir .pages-state]
                                        [--site-url URL] [--repo-url URL]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app as backend  # noqa: E402
from core.alert_store import AlertStore  # noqa: E402
from core.timeseries_store import TimeSeriesStore  # noqa: E402
from data.space_weather import fetch_f107_history, fetch_kp_history  # noqa: E402
from scripts.generate_dashboard_snapshot import write_dashboard_snapshot  # noqa: E402

STATIC_DIR = ROOT / "static"
PAGES_DIR = ROOT / "pages"

# Keep in sync with the cron schedule in .github/workflows/pages.yml.
REFRESH_HOURS = 3

PAGES = ("index.html", "cv.html", "timeline.html", "alerts.html")
PAGE_LINKS = {"/": "./", "/cv": "cv.html", "/timeline": "timeline.html", "/alerts": "alerts.html"}

# Live-backend wording that doesn't hold for the snapshot. Each `old` must
# exist verbatim — the build fails loudly if a page's copy drifts.
PAGE_TEXT = {
    "timeline.html": [
        ("SAMPLED EVERY 2 MIN", f"SAMPLED EVERY {REFRESH_HOURS} H &middot; KP &amp; F10.7 HISTORY FROM NOAA"),
    ],
    "alerts.html": [("LIVE PUSH", f"REFRESHED EVERY {REFRESH_HOURS} H")],
}

# The mission planner's form defaults (static/index.html) and the request
# dashboard.js builds from them: one scan per launch site is precomputed
# with exactly these parameters, and the adapter locks the form to them.
COLA_WINDOW_HOURS = 6
COLA_REQUEST = {
    "target_inclination_deg": 51.6,
    "target_altitude_km": 400.0,
    "bubble_radius_km": 50.0,
    "candidate_step_minutes": max(1, round(COLA_WINDOW_HOURS * 60 / 60)),
    "catalog_groups": ["stations", "visual"],
    "catalog_limit_per_group": 150,
}

CV_RECORD_FRAMES = 64
CV_RECORD_FPS = 8
CV_FRAME_MAX_SIDE = 960
CV_FRAME_JPEG_QUALITY = 72

# Environment variables whose values must never appear in published output.
SECRET_ENV_VARS = ("NASA_API_KEY", "SPACE_TRACK_USERNAME", "SPACE_TRACK_PASSWORD", "ALERT_WEBHOOK_URL")

CSP_TEMPLATE = (
    "default-src 'self'; "
    "script-src 'self' {script_hashes} https://unpkg.com https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "img-src 'self' data: blob: https:; "
    "connect-src 'self' https://cdn.jsdelivr.net https://unpkg.com; "
    "media-src 'self' blob:; "
    "worker-src 'self' blob:; "
    "object-src 'none'; base-uri 'self'; form-action 'none'; upgrade-insecure-requests"
)

PALETTE = {
    "background": "#0a0a14", "cyan": "#00ffff", "amber": "#ffcc00",
    "magenta": "#ff0066", "green": "#00ff66",
}


def log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Writing API responses
# ---------------------------------------------------------------------------

class SiteWriter:
    """Writes captured responses to the site and to the last-good cache, falling back to the cache on failure."""

    def __init__(self, out: Path, state: Path):
        self.api = out / "api"
        self.last_good = state / "last-good"
        self.live: list[str] = []
        self.carried_over: list[str] = []
        self.missing: list[str] = []

    def write(self, rel: str, payload) -> None:
        text = json.dumps(payload, separators=(",", ":"), default=str)
        for base in (self.api, self.last_good):
            path = base / f"{rel}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def capture(self, client: TestClient, rel: str, url: str | None = None, method: str = "GET",
                body: dict | None = None, required: bool = False):
        url = url or f"/api/{rel}"
        resp = None
        try:
            resp = client.request(method, url, json=body)
            error = None if resp.status_code == 200 else f"HTTP {resp.status_code}: {resp.text[:160]}"
        except Exception as e:  # an upstream client raising straight through the endpoint
            error = f"{type(e).__name__}: {e}"

        if error is None:
            payload = resp.json()
            self.write(rel, payload)
            self.live.append(rel)
            return payload
        if resp is not None and resp.status_code == 400:
            # A deterministic answer about the request itself (e.g. an
            # inclination a site can't reach) — publish it as-is; the
            # adapter replays it with the same status and message.
            self.write(rel, {"__status": 400, "detail": resp.json().get("detail")})
            self.live.append(rel)
            return None

        cached = self.last_good / f"{rel}.json"
        if cached.exists():
            dest = self.api / f"{rel}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, dest)
            self.carried_over.append(rel)
            log(f"  ! {url} failed ({error}) — using previous build's copy")
            return json.loads(cached.read_text())

        if required:
            raise SystemExit(f"Required endpoint {url} failed with no previous copy: {error}")
        self.missing.append(rel)
        log(f"  ! {url} failed ({error}) — publishing the error")
        # Publish the backend's own error (e.g. "MAST archive unreachable")
        # so the panel shows its normal outage message — but keep it out of
        # last-good, so it never replaces a real response later.
        detail = None
        if resp is not None:
            try:
                detail = resp.json().get("detail")
            except ValueError:
                pass
        dest = self.api / f"{rel}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"__status": 503, "detail": detail or "Upstream data source unavailable at build time."}))
        return None


# ---------------------------------------------------------------------------
# Background state (normally maintained by the app's lifespan loops)
# ---------------------------------------------------------------------------

def _extend_history(store: TimeSeriesStore, metric: str, points: list[tuple[datetime, float]]) -> int:
    """Add points older than the stored history or newer than it — never re-insert the overlap."""
    existing = store.query(metric, limit=10**9)
    first = existing[0].timestamp if existing else None
    last = existing[-1].timestamp if existing else None
    added = 0
    for t, value in points:
        if first is None or t < first or t > last:
            store.record(metric, value, timestamp=t)
            added += 1
    return added


def refresh_background_state() -> None:
    """One pass of what the live server does continuously: metric sampling and the alert scan."""
    store = backend._timeseries_store
    for metric, fetch in (("kp_index", fetch_kp_history), ("f107_flux", fetch_f107_history)):
        try:
            log(f"  backfilled {_extend_history(store, metric, fetch())} {metric} samples from NOAA")
        except Exception as e:
            log(f"  ! {metric} history unavailable ({e})")

    async def one_pass():
        await backend._sample_metrics_once()
        await backend._scan_conjunctions_once()
        await backend._scan_neo_alerts_once()
        await backend._scan_space_weather_alerts_once()

    asyncio.run(one_pass())
    store.prune()


# ---------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------

def capture_api(client: TestClient, site: SiteWriter) -> None:
    site.capture(client, "launch-sites", required=True)

    site.capture(client, "solar-system/bodies")
    for body in dict.fromkeys(["sun", *backend.SOLAR_SYSTEM_BODIES]):
        site.capture(client, f"solar-system/{body}/position")
    site.capture(client, "solar-system/moons")
    for key in backend.SOLAR_SYSTEM_MOONS:
        site.capture(client, f"solar-system/moons/{key}/position")
    site.capture(client, "small-bodies")
    for key in backend.SMALL_BODIES:
        site.capture(client, f"small-bodies/{key}/position")

    site.capture(client, "timeseries")
    for metric in backend.METRIC_DEFINITIONS:
        site.capture(client, f"timeseries/{metric}", f"/api/timeseries/{metric}?hours=720&limit={backend.MAX_QUERY_LIMIT}")
    site.capture(client, "alerts", f"/api/alerts?limit={backend.MAX_QUERY_LIMIT}")
    site.capture(client, "alerts/unacknowledged-count")

    site.capture(client, "sensors/dsn")
    site.capture(client, "sensors/telescopes")
    for key in backend.TELESCOPES:
        site.capture(client, f"sensors/telescopes/{key}")
    site.capture(client, "sensors/earth-epic")
    site.capture(client, "sensors/apod")

    site.capture(client, "imagery")
    site.capture(client, "earth/events")
    site.capture(client, "space-weather/events")

    site.capture(client, "cv/footage")
    site.capture(client, "cv/anomaly-status")
    site.capture(client, "cv/space-model-status")
    site.capture(client, "cv/anomaly-log", f"/api/cv/anomaly-log?limit={backend.MAX_QUERY_LIMIT}")


def trajectory_key(launch_time: str) -> str:
    # Mirrors pages/static-adapter.js: launch_time with every non-digit stripped.
    return re.sub(r"\D", "", launch_time)


def capture_cola(client: TestClient, site: SiteWriter, sites: dict) -> None:
    start = datetime.now(timezone.utc)
    end = start + timedelta(hours=COLA_WINDOW_HOURS)
    for key in sites:
        rel = f"cola/scan/{key}"
        body = {**COLA_REQUEST, "site": key, "search_start": start.isoformat(), "search_end": end.isoformat()}
        scan = site.capture(client, rel, "/api/cola/scan", "POST", body)
        if scan is None or rel in site.carried_over:
            continue  # carried-over scans keep their carried-over trajectories

        shutil.rmtree(site.last_good / "cola" / "trajectory" / key, ignore_errors=True)
        for candidate in scan["candidates"]:
            traj_body = {
                k: v for k, v in body.items()
                if k not in ("search_start", "search_end", "candidate_step_minutes")
            } | {"launch_time": candidate["launch_time"]}
            site.capture(client, f"cola/trajectory/{key}/{trajectory_key(candidate['launch_time'])}",
                         "/api/cola/trajectory", "POST", traj_body)
        clear = sum(c["clear"] for c in scan["candidates"])
        log(f"  COLA {key}: {clear}/{len(scan['candidates'])} windows clear")


def _web_jpeg(jpeg: bytes) -> bytes:
    """Re-encode a full-resolution stream frame at a size suited to replaying over the web."""
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    scale = CV_FRAME_MAX_SIDE / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, CV_FRAME_JPEG_QUALITY])
    return encoded.tobytes() if ok else jpeg


def record_cv_stream(client: TestClient, out: Path, scratch: Path) -> None:
    """
    Record real annotated frames from /ws/cv for the adapter to replay.

    The stream normally feeds anomaly scores into the metric history and
    raises CV-anomaly alerts. A build-time recording isn't a real
    observation (it's usually the sample clip), so those writes go to
    throwaway stores instead of the persisted history.
    """
    persistent = backend._timeseries_store, backend._alert_store
    backend._timeseries_store = TimeSeriesStore(scratch / "cv-timeseries.db")
    backend._alert_store = AlertStore(scratch / "cv-alerts.db")
    try:
        _record_cv_frames(client, out)
    finally:
        backend._timeseries_store, backend._alert_store = persistent
        for db in ("cv-timeseries.db", "cv-alerts.db"):
            (scratch / db).unlink(missing_ok=True)


def _record_cv_frames(client: TestClient, out: Path) -> None:
    stream_dir = out / "api" / "cv" / "stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    clip = backend.FOOTAGE_CLIPS[backend.DEFAULT_FOOTAGE_CLIP]
    frames, label = [], f"{clip.title}, {clip.recorded} · NASA Image and Video Library"

    with client.websocket_connect(f"/ws/cv?source=nasa_footage&model=space&target_fps={CV_RECORD_FPS}") as ws:
        pending_jpeg = None
        while len(frames) < CV_RECORD_FRAMES:
            message = ws.receive()
            if message.get("bytes") is not None:
                pending_jpeg = message["bytes"]
                continue
            data = json.loads(message["text"])
            if "error" in data:
                raise RuntimeError(data["error"])
            if "notice" in data:
                if "sample clip" in data["notice"]:
                    label = "SAMPLE CLIP — NASA footage unavailable at build time"
                log(f"  CV notice: {data['notice']}")
                continue
            if pending_jpeg is not None:
                name = f"{len(frames):03d}.jpg"
                (stream_dir / name).write_bytes(_web_jpeg(pending_jpeg))
                frames.append({"file": name, "meta": data})
                pending_jpeg = None

    (stream_dir / "meta.json").write_text(json.dumps({
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "label": label, "model": "space", "details_url": clip.details_url,
        "fps": CV_RECORD_FPS, "frames": frames,
    }, separators=(",", ":")))
    log(f"  recorded {len(frames)} CV frames ({label})")


# ---------------------------------------------------------------------------
# Social preview image
# ---------------------------------------------------------------------------

def render_og_image(snapshot_path: Path, dest: Path, generated_at: datetime) -> None:
    """1200x630 link-preview card drawn from the real snapshot: object positions plus station ground tracks."""
    snapshot = json.loads(snapshot_path.read_text())
    colors = {"stations": PALETTE["amber"], "debris": PALETTE["magenta"]}

    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=PALETTE["background"])
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=PALETTE["background"])
    ax.set_xlim(-180, 180)
    ax.set_ylim(-95, 128)  # headroom above the poles keeps the title band clear
    ax.axis("off")

    for lon in range(-180, 181, 30):
        ax.axvline(lon, color=PALETTE["cyan"], alpha=0.06, lw=0.8)
    for lat in range(-90, 91, 30):
        ax.axhline(lat, color=PALETTE["cyan"], alpha=0.06, lw=0.8)
    ax.axhline(98, color=PALETTE["cyan"], alpha=0.35, lw=1)

    # Each ground track is already split into segments at the antimeridian.
    for segments in snapshot["ground_tracks"].values():
        for segment in segments:
            ax.plot([p[1] for p in segment], [p[0] for p in segment], color=PALETTE["amber"], alpha=0.35, lw=1)

    for sat in snapshot["satellites"]:
        lat, lon, _ = sat["track"][0]
        color = PALETTE["green"] if sat["norad_id"] == 25544 else colors.get(sat["classification"], PALETTE["cyan"])
        size = 60 if sat["norad_id"] == 25544 else 10
        ax.scatter(lon, lat, s=size * 4, color=color, alpha=0.12, linewidths=0)
        ax.scatter(lon, lat, s=size, color=color, alpha=0.95, linewidths=0)

    text = {"family": "monospace", "color": PALETTE["cyan"]}
    ax.text(-172, 118, "COSMIC OBSERVER", fontsize=34, fontweight="bold", va="center", **text)
    ax.text(-172, 106, "SPACE DOMAIN AWARENESS // REAL DATA FROM NASA · JPL · NOAA · CELESTRAK",
            fontsize=11.5, alpha=0.75, va="center", **text)
    ax.text(172, 106, f"{len(snapshot['satellites'])} OBJECTS · {generated_at:%d %b %Y %H:%M} UTC",
            fontsize=11, ha="right", va="center", alpha=0.8, **(text | {"color": PALETTE["green"]}))

    fig.savefig(dest, facecolor=PALETTE["background"])
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _inline_script_hashes(html: str) -> list[str]:
    hashes = []
    for content in re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, flags=re.S):
        digest = base64.b64encode(hashlib.sha256(content.encode()).digest()).decode()
        hashes.append(f"'sha256-{digest}'")
    return hashes


def _head_injection(html: str, repo_url: str | None, extra: str = "") -> str:
    csp = CSP_TEMPLATE.format(script_hashes=" ".join(_inline_script_hashes(html)))
    lines = [f'<meta http-equiv="Content-Security-Policy" content="{csp}" />']
    if repo_url:
        lines.append(f'<meta name="co-source" content="{repo_url}" />')
    lines.append(extra)
    lines.append('<link rel="stylesheet" href="static-adapter.css" />')
    lines.append('<script src="static-adapter.js"></script>')
    return "\n".join(line for line in lines if line) + "\n"


def build_pages(out: Path, site_url: str | None, repo_url: str | None) -> None:
    charset = '<meta charset="UTF-8" />\n'
    for name in PAGES:
        html = (STATIC_DIR / name).read_text()

        for route, target in PAGE_LINKS.items():
            html = html.replace(f'href="{route}"', f'href="{target}"')
        for old, new in PAGE_TEXT.get(name, []):
            if old not in html:
                raise SystemExit(f"{name}: expected text {old!r} not found — update PAGE_TEXT")
            html = html.replace(old, new)

        extra = ""
        if name == "index.html" and site_url:
            extra = (
                f'<link rel="canonical" href="{site_url}" />\n'
                f'<meta property="og:url" content="{site_url}" />\n'
                f'<meta property="og:image" content="{site_url}og-image.png" />\n'
                '<meta property="og:image:width" content="1200" />\n'
                '<meta property="og:image:height" content="630" />'
            )
        assert html.count(charset) == 1, name
        html = html.replace(charset, charset + _head_injection(html, repo_url, extra))
        (out / name).write_text(html)

    # GitHub Pages serves 404.html for any missing path, at any depth, so
    # its relative asset links need an explicit base.
    html = (PAGES_DIR / "404.html").read_text()
    base = f'<base href="{urlparse(site_url).path}" />\n' if site_url else ""
    csp = CSP_TEMPLATE.format(script_hashes="")
    html = html.replace(charset, charset + base + f'<meta http-equiv="Content-Security-Policy" content="{csp}" />\n')
    (out / "404.html").write_text(html)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def assert_no_secrets(out: Path) -> None:
    secrets = {
        name: value for name in SECRET_ENV_VARS
        if len(value := os.environ.get(name, "")) >= 8 and value != "DEMO_KEY"
    }
    if not secrets:
        return
    for path in out.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        for name, value in secrets.items():
            if value.encode() in data:
                raise SystemExit(f"Refusing to publish: the value of ${name} appears in {path.relative_to(out)}")
    log(f"secret scan clean ({len(secrets)} secret(s) checked)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _default_repo_url() -> str | None:
    if os.environ.get("GITHUB_REPOSITORY"):
        return f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ['GITHUB_REPOSITORY']}"
    try:
        remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return remote.removesuffix(".git") if remote.startswith("https://") else None


def build(out: Path, state: Path, site_url: str | None, repo_url: str | None) -> None:
    started = time.monotonic()
    generated_at = datetime.now(timezone.utc)

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    state.mkdir(parents=True, exist_ok=True)

    # The snapshot is a read-only view of the app — build it the way it's
    # deployed. Point the app's stores at the persisted build state (never
    # the local dev databases), and make sure a build can't fire a webhook.
    backend.PUBLIC_DEMO = True
    backend._timeseries_store = TimeSeriesStore(state / "timeseries.db")
    backend._alert_store = AlertStore(state / "alerts.db")
    backend.ALERT_WEBHOOK_URL = None

    for path in STATIC_DIR.iterdir():
        if path.is_file() and path.suffix in {".js", ".css", ".svg"}:
            shutil.copyfile(path, out / path.name)
    for name in ("static-adapter.js", "static-adapter.css"):
        shutil.copyfile(PAGES_DIR / name, out / name)
    (out / ".nojekyll").touch()

    log("dashboard snapshot (CelesTrak TLEs + SGP4)")
    try:
        write_dashboard_snapshot(out / "snapshot.json")
        shutil.copyfile(out / "snapshot.json", state / "snapshot.json")
    except Exception as e:
        if not (state / "snapshot.json").exists():
            raise SystemExit(f"Dashboard snapshot failed with no previous copy: {e}")
        log(f"  ! snapshot failed ({e}) — using previous build's copy")
        shutil.copyfile(state / "snapshot.json", out / "snapshot.json")

    log("metric sampling + alert scan")
    refresh_background_state()

    site = SiteWriter(out, state)
    client = TestClient(backend.app, raise_server_exceptions=False)

    log("CV stream recording")
    try:
        record_cv_stream(client, out, state)
    except Exception as e:
        log(f"  ! CV recording failed ({type(e).__name__}: {e}) — stream tab will report backend-only")

    log("API responses")
    capture_api(client, site)

    log("COLA launch-window scans")
    capture_cola(client, site, json.loads((out / "api" / "launch-sites.json").read_text()))

    log("pages")
    build_pages(out, site_url, repo_url)
    render_og_image(out / "snapshot.json", out / "og-image.png", generated_at)

    (out / "api" / "_build.json").write_text(json.dumps({
        "generated_at": generated_at.isoformat(),
        "refresh_hours": REFRESH_HOURS,
        "commit": os.environ.get("GITHUB_SHA"),
        "endpoints": {"live": len(site.live), "carried_over": site.carried_over, "missing": site.missing},
    }, indent=2))

    assert_no_secrets(out)

    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    log(f"done in {time.monotonic() - started:.0f}s — {len(site.live)} live, "
        f"{len(site.carried_over)} carried over, {len(site.missing)} missing, {size_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "_site", help="Output directory (default: _site)")
    parser.add_argument("--state-dir", type=Path, default=ROOT / ".pages-state",
                        help="Persistent build state: metric/alert history + last-good responses")
    parser.add_argument("--site-url", help="Public URL of the deployed site, for canonical/Open Graph tags")
    parser.add_argument("--repo-url", default=_default_repo_url(), help="Source repository URL linked from the site")
    args = parser.parse_args()

    site_url = args.site_url.rstrip("/") + "/" if args.site_url else None
    build(args.out.resolve(), args.state_dir.resolve(), site_url, args.repo_url)


if __name__ == "__main__":
    main()
