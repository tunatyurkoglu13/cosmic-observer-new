"""
data/dscovr_epic.py — NASA EPIC (Earth Polychromatic Imaging Camera).

EPIC rides on the NOAA DSCOVR spacecraft, parked at the Earth-Sun L1
Lagrange point ~1.5 million km from Earth (sunward), and takes a real,
genuine full-disk photograph of the sunlit side of Earth roughly every
1-2 hours. This is a real image, not a composite/render — the entire
planet lit by the Sun, photographed from deep space.

Metadata endpoint: https://epic.gsfc.nasa.gov/api/natural (keyless)
Image archive: https://epic.gsfc.nasa.gov/archive/natural/{Y}/{m}/{d}/png/{image}.png
Both confirmed live during development (real, current-day imagery).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from core.resilient_fetch import ResilientFetcher

EPIC_METADATA_URL = "https://epic.gsfc.nasa.gov/api/natural"
EPIC_ARCHIVE_BASE = "https://epic.gsfc.nasa.gov/archive/natural"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "epic_latest.json"


@dataclass
class EPICImage:
    identifier: str
    caption: str
    image_url: str
    date: str            # "YYYY-MM-DD HH:MM:SS" UTC, real DSCOVR capture time
    centroid_lat: float   # sub-solar point on Earth, degrees
    centroid_lon: float


class EPICClient(ResilientFetcher[EPICImage]):
    """Resilient-cached client for the latest real full-Earth EPIC image."""

    LATEST_KEY = "latest"

    def __init__(
        self,
        cache_path: Path | str = DEFAULT_CACHE_PATH,
        timeout: int = 20,
        # EPIC captures roughly every 1-2 hours — no point re-fetching more often.
        staleness: timedelta = timedelta(hours=1),
        failure_retry_cooldown: timedelta = timedelta(minutes=2),
    ):
        self.timeout = timeout
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.staleness = staleness
        self.failure_retry_cooldown = failure_retry_cooldown

    def fetch_latest(self, force: bool = False) -> EPICImage:
        return self.fetch(self.LATEST_KEY, force=force)

    # --- ResilientFetcher hooks -------------------------------------------------

    def _fetch_live(self, key: str) -> EPICImage:
        resp = requests.get(EPIC_METADATA_URL, timeout=self.timeout)
        resp.raise_for_status()
        records = resp.json()
        if not records:
            raise RuntimeError("EPIC API returned no images")

        rec = records[-1]  # chronologically last = most recent capture
        date_part = rec["date"].split(" ")[0]
        year, month, day = date_part.split("-")
        image_url = f"{EPIC_ARCHIVE_BASE}/{year}/{month}/{day}/png/{rec['image']}.png"

        centroid = rec.get("centroid_coordinates", {})
        return EPICImage(
            identifier=rec.get("identifier", ""),
            caption=rec.get("caption", ""),
            image_url=image_url,
            date=rec.get("date", ""),
            centroid_lat=float(centroid.get("lat", 0.0)),
            centroid_lon=float(centroid.get("lon", 0.0)),
        )

    def _read_cache_file(self) -> dict | None:
        if not self.cache_path.exists():
            return None
        try:
            return json.loads(self.cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _load_cache(self, key: str) -> EPICImage | None:
        payload = self._read_cache_file()
        if not payload or "image" not in payload:
            return None
        return EPICImage(**payload["image"])

    def _save_cache(self, key: str, data: EPICImage) -> None:
        self.cache_path.write_text(json.dumps({
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "image": asdict(data),
        }))

    def _store_fallback(self, key: str, data: EPICImage) -> None:
        self._save_cache(key, data)

    def _load_seed(self, key: str) -> EPICImage | None:
        return None

    def _is_stale(self, key: str) -> bool:
        payload = self._read_cache_file()
        if not payload or "cached_at" not in payload:
            return True
        cached_at = datetime.fromisoformat(payload["cached_at"])
        return datetime.now(timezone.utc) - cached_at > self.staleness

    def _recently_failed(self, key: str) -> bool:
        payload = self._read_cache_file()
        if not payload or "failed_at" not in payload:
            return False
        failed_at = datetime.fromisoformat(payload["failed_at"])
        return datetime.now(timezone.utc) - failed_at <= self.failure_retry_cooldown

    def _record_failure(self, key: str) -> None:
        payload = self._read_cache_file() or {}
        payload["failed_at"] = datetime.now(timezone.utc).isoformat()
        self.cache_path.write_text(json.dumps(payload))

    def _clear_failure(self, key: str) -> None:
        payload = self._read_cache_file()
        if payload and "failed_at" in payload:
            del payload["failed_at"]
            self.cache_path.write_text(json.dumps(payload))
