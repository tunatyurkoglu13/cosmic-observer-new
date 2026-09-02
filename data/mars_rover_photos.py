"""
data/mars_rover_photos.py — NASA Mars Rover Photos API (Curiosity/Perseverance).

Real, most-recent raw imagery the rovers have actually downlinked to
Earth — not renders, not stock photos. Part of api.nasa.gov, same
NASA_API_KEY convention as data.space_weather/data.neows.

Endpoint: https://api.nasa.gov/mars-photos/api/v1/rovers/{rover}/latest_photos
Docs: https://api.nasa.gov/ ("Mars Rover Photos")

Honest status note (found during development, not hypothetical): this
API's backing service was UNREACHABLE at the time this client was built
— every request (with a real key and with DEMO_KEY) returned a raw
Heroku "No such app" error page, a known, occasionally-reported
reliability problem with this specific NASA API. The client is written
correctly against the real, stable, long-documented response contract
regardless (this is not a guess — the {photos: [{id, sol, camera, img_src,
earth_date, rover}]} shape is NASA's own long-standing documented
format) and is wired into the same resilient-fetch chain as everything
else in this project: if the service is unreachable and there's no
cache yet, it raises rather than fabricating rover photos — see
_fetch_live's honest failure path in core.resilient_fetch.ResilientFetcher.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from core.resilient_fetch import ResilientFetcher

MARS_PHOTOS_BASE_URL = "https://api.nasa.gov/mars-photos/api/v1/rovers"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "mars_rover_photos"

ROVERS = ("curiosity", "perseverance")


@dataclass
class RoverPhoto:
    photo_id: int
    sol: int                  # Martian day since the rover's landing
    camera_name: str
    camera_full_name: str
    img_src: str
    earth_date: str
    rover_name: str
    rover_status: str


class MarsRoverPhotosClient(ResilientFetcher[list[RoverPhoto]]):
    """Resilient-cached client for each rover's latest real downlinked photos."""

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        api_key: str | None = None,
        timeout: int = 20,
        # Rovers downlink new photos roughly daily at most — matches
        # TLEManager's own default staleness for a similarly slow-moving feed.
        staleness: timedelta = timedelta(hours=6),
        failure_retry_cooldown: timedelta = timedelta(minutes=2),
    ):
        self.api_key = api_key or os.environ.get("NASA_API_KEY", "DEMO_KEY")
        self.timeout = timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.staleness = staleness
        self.failure_retry_cooldown = failure_retry_cooldown

    def fetch_latest(self, rover: str, force: bool = False) -> list[RoverPhoto]:
        if rover not in ROVERS:
            raise ValueError(f"Unknown rover '{rover}'. Supported: {list(ROVERS)}")
        return self.fetch(rover, force=force)

    # --- ResilientFetcher hooks -------------------------------------------------

    def _fetch_live(self, key: str) -> list[RoverPhoto]:
        resp = requests.get(
            f"{MARS_PHOTOS_BASE_URL}/{key}/latest_photos",
            params={"api_key": self.api_key},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        photos = payload.get("latest_photos", [])
        return [
            RoverPhoto(
                photo_id=p["id"], sol=p["sol"],
                camera_name=p["camera"]["name"], camera_full_name=p["camera"]["full_name"],
                img_src=p["img_src"], earth_date=p["earth_date"],
                rover_name=p["rover"]["name"], rover_status=p["rover"]["status"],
            )
            for p in photos
        ]

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

    def _load_cache(self, key: str) -> list[RoverPhoto] | None:
        payload = self._read_cache_file(key)
        if not payload or "photos" not in payload:
            return None
        return [RoverPhoto(**p) for p in payload["photos"]]

    def _save_cache(self, key: str, data: list[RoverPhoto]) -> None:
        self._cache_path(key).write_text(json.dumps({
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "photos": [asdict(p) for p in data],
        }))

    def _store_fallback(self, key: str, data: list[RoverPhoto]) -> None:
        self._save_cache(key, data)

    def _load_seed(self, key: str) -> list[RoverPhoto] | None:
        return None

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
