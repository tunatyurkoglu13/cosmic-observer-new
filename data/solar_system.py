"""
data/solar_system.py — Real-time Earth-relative position of solar-system
bodies, for the 3D dashboard's "travel to a celestial body" feature.

Wraps data.horizons.HorizonsClient (JPL Horizons) to answer one specific
question per body: "right now, which direction is it from Earth, and how
far?" — the Earth-centered Cartesian vector Horizons returns directly
gives both (direction = normalized vector, distance = its magnitude), no
further astrodynamics needed for this purpose.

This is a real, verified geometric direction — not a stylized guess —
but the DISPLAY distance the frontend places the body at is a
deliberately compressed dramatization (real interplanetary distances are
many orders of magnitude larger than this dashboard's scene scale) so
the "travel" animation stays navigable; see static/dashboard.js's
DISPLAY_DISTANCE_SCALE for exactly how that compression works, and
BODIES below for each body's true physical radius (which the frontend
DOES render to real relative scale, matching how it already sizes Earth).

Cached via core.resilient_fetch.ResilientFetcher (same pattern as
core.tle_manager.TLEManager and data.nasa_cneos.CNEOSClient) since
Horizons queries take a second or two and planetary geometry changes
slowly enough that an hours-old position is still a perfectly good
approximation for this dashboard's purposes. No bundled seed fallback is
provided (unlike TLE data) — there's no small last-known-good planetary
ephemeris checked into the repo, so a cold cache with no network simply
raises, same as data.nasa_cneos.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from core.resilient_fetch import ResilientFetcher

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "solar_system"

# Real physical radius (km) and a muted, realistic (not photographic,
# not garish) base display color for each supported body. Radii are
# well-established astronomical constants. Colors approximate each
# body's true low-saturation appearance (e.g. Mercury's dark, warm-gray
# regolith, comparable in albedo to the Moon's) rather than an
# "enhanced/false color" scientific-imagery palette.
BODIES: dict[str, dict] = {
    "sun": {
        "display_name": "Sun",
        "radius_km": 696000.0,
        "color_hex": 0xfff2c2,
    },
    "mercury": {
        "display_name": "Mercury",
        "radius_km": 2439.7,
        "color_hex": 0x8a8175,
    },
}


@dataclass
class BodyPosition:
    """Earth-centered real-time position of one body, from JPL Horizons."""

    body: str
    fetched_at: str          # ISO 8601 UTC — when this was fetched from Horizons
    jd_tdb: float             # Julian date (TDB) of the Horizons sample used
    r_km: tuple[float, float, float]   # Earth-centered Cartesian position, km
    distance_km: float
    direction: tuple[float, float, float]  # unit vector, same frame as r_km
    # Real direction FROM this body TOWARD the Sun (unit vector, same
    # frame) — used to light the body correctly (its true terminator),
    # rather than an arbitrary fixed light direction. Meaningless for the
    # Sun itself, which reports (0.0, 0.0, 0.0).
    sun_direction: tuple[float, float, float] = (0.0, 0.0, 0.0)


class SolarSystemClient(ResilientFetcher[BodyPosition]):
    """Real-time Earth-relative body positions, resilient-cached per body."""

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        staleness: timedelta = timedelta(hours=6),
        failure_retry_cooldown: timedelta = timedelta(minutes=2),
        timeout: int = 60,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.staleness = staleness
        self.failure_retry_cooldown = failure_retry_cooldown
        self.timeout = timeout

    def fetch_position(self, body: str, force: bool = False) -> BodyPosition:
        """
        Real-time Earth-centered position of `body` (a key in BODIES).

        Raises:
            ValueError: unknown body.
            requests.RequestException: Horizons unreachable and no cache available.
        """
        if body not in BODIES:
            raise ValueError(f"Unknown body '{body}'. Supported: {list(BODIES)}")
        return self.fetch(body, force=force)

    # --- ResilientFetcher hooks -------------------------------------------------

    def _fetch_live(self, key: str) -> BodyPosition:
        from data.horizons import BODY_CODES, HorizonsClient

        now = datetime.now(timezone.utc)
        start_str = now.strftime("%Y-%m-%d %H:%M")
        end_str = (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")

        client = HorizonsClient(timeout=self.timeout)
        samples = client.fetch_vectors(key, start_str, end_str, step_size="1h", center="500@399")
        if not samples:
            raise RuntimeError(f"Horizons returned no samples for '{key}'")

        sample = samples[0]
        x, y, z = sample.r_km
        distance_km = math.sqrt(x * x + y * y + z * z)
        direction = (x / distance_km, y / distance_km, z / distance_km)

        sun_direction = (0.0, 0.0, 0.0)
        if key != "sun":
            # Real direction from THIS body toward the Sun (queried
            # centered on the body itself, not on Earth) — what actually
            # determines where its terminator/day-side falls, as
            # opposed to a stylized fixed light direction.
            body_code = BODY_CODES[key]
            sun_samples = client.fetch_vectors("sun", start_str, end_str, step_size="1h", center=f"500@{body_code}")
            if sun_samples:
                sx, sy, sz = sun_samples[0].r_km
                sun_dist = math.sqrt(sx * sx + sy * sy + sz * sz)
                sun_direction = (sx / sun_dist, sy / sun_dist, sz / sun_dist)

        return BodyPosition(
            body=key,
            fetched_at=now.isoformat(),
            jd_tdb=sample.jd_tdb,
            r_km=(x, y, z),
            distance_km=distance_km,
            direction=direction,
            sun_direction=sun_direction,
        )

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache_file(self, key: str) -> dict | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _load_cache(self, key: str) -> BodyPosition | None:
        payload = self._read_cache_file(key)
        if not payload or "position" not in payload:
            return None
        rec = payload["position"]
        return BodyPosition(
            body=rec["body"], fetched_at=rec["fetched_at"], jd_tdb=rec["jd_tdb"],
            r_km=tuple(rec["r_km"]), distance_km=rec["distance_km"], direction=tuple(rec["direction"]),
            sun_direction=tuple(rec.get("sun_direction", (0.0, 0.0, 0.0))),
        )

    def _save_cache(self, key: str, data: BodyPosition) -> None:
        self._cache_path(key).write_text(json.dumps({
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "position": asdict(data),
        }))

    def _store_fallback(self, key: str, data: BodyPosition) -> None:
        self._save_cache(key, data)

    def _load_seed(self, key: str) -> BodyPosition | None:
        return None  # no bundled last-known-good ephemeris is checked into the repo

    def _is_stale(self, key: str) -> bool:
        payload = self._read_cache_file(key)
        if not payload or "cached_at" not in payload:
            return True
        cached_at = datetime.fromisoformat(payload["cached_at"])
        return datetime.now(timezone.utc) - cached_at > self.staleness

    def _recently_failed(self, key: str) -> bool:
        payload = self._read_cache_file(key)
        if not payload or "failed_at" not in payload:
            return False
        failed_at = datetime.fromisoformat(payload["failed_at"])
        return datetime.now(timezone.utc) - failed_at <= self.failure_retry_cooldown

    def _record_failure(self, key: str) -> None:
        payload = self._read_cache_file(key) or {}
        payload["failed_at"] = datetime.now(timezone.utc).isoformat()
        self._cache_path(key).write_text(json.dumps(payload))

    def _clear_failure(self, key: str) -> None:
        payload = self._read_cache_file(key)
        if payload and "failed_at" in payload:
            del payload["failed_at"]
            self._cache_path(key).write_text(json.dumps(payload))
