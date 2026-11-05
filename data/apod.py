"""
data/apod.py — NASA Astronomy Picture of the Day.

Endpoint (NASA API key): https://api.nasa.gov/planetary/apod
The key travels in the X-Api-Key header (see data/donki.py for why).

APOD is usually an image but is sometimes a video (YouTube/Vimeo embed);
`media_type` tells the frontend which it got. The `thumbs=true` option
(video thumbnails) is deliberately not sent: the API answers it with a 500.

Since APOD moved to science.nasa.gov the API has been observed returning a
placeholder for every date — the right explanation, but the NASA Science
logo as the image and "NASA Science" as the title. That is rejected as
bad data (RuntimeError, so ResilientFetcher won't cache it) rather than
shown as "today's picture".
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import requests

from core.resilient_fetch import JsonFileFetcher

APOD_URL = "https://api.nasa.gov/planetary/apod"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "apod.json"

FIELDS = ("date", "title", "explanation", "media_type", "url", "hdurl", "thumbnail_url", "copyright")
PLACEHOLDER_IMAGE_MARKER = "nasa-logo"


def is_placeholder(apod: dict) -> bool:
    return PLACEHOLDER_IMAGE_MARKER in (apod.get("url") or "") or apod.get("title") == "NASA Science"


class APODClient(JsonFileFetcher[dict]):
    """Resilient-cached client for today's Astronomy Picture of the Day."""

    KEY = "today"

    def __init__(self, cache_path: Path | str = DEFAULT_CACHE_PATH, timeout: int = 45,
                 staleness: timedelta = timedelta(hours=2)):
        super().__init__(cache_path, staleness)
        self.timeout = timeout

    def fetch_today(self, force: bool = False) -> dict:
        return self.fetch(self.KEY, force=force)

    def _fetch_live(self, key: str) -> dict:
        resp = requests.get(
            APOD_URL,
            headers={"X-Api-Key": os.environ.get("NASA_API_KEY") or "DEMO_KEY"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        raw = resp.json()
        apod = {field: raw.get(field) for field in FIELDS}
        if is_placeholder(apod):
            raise RuntimeError(
                "NASA's APOD API is currently returning placeholder data (the NASA Science logo) "
                "instead of the picture of the day"
            )
        if apod["copyright"]:
            apod["copyright"] = " ".join(apod["copyright"].split())  # APOD pads credits with newlines
        return apod
