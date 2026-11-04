"""
data/apod.py — NASA Astronomy Picture of the Day.

Endpoint (NASA API key): https://api.nasa.gov/planetary/apod
The key travels in the X-Api-Key header (see data/donki.py for why).

APOD is usually an image but is sometimes a video (YouTube/Vimeo embed);
`media_type` tells the frontend which it got.
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


class APODClient(JsonFileFetcher[dict]):
    """Resilient-cached client for today's Astronomy Picture of the Day."""

    KEY = "today"

    def __init__(self, cache_path: Path | str = DEFAULT_CACHE_PATH, timeout: int = 20,
                 staleness: timedelta = timedelta(hours=2)):
        super().__init__(cache_path, staleness)
        self.timeout = timeout

    def fetch_today(self, force: bool = False) -> dict:
        return self.fetch(self.KEY, force=force)

    def _fetch_live(self, key: str) -> dict:
        resp = requests.get(
            APOD_URL,
            params={"thumbs": "true"},
            headers={"X-Api-Key": os.environ.get("NASA_API_KEY") or "DEMO_KEY"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        raw = resp.json()
        apod = {field: raw.get(field) for field in FIELDS}
        if apod["copyright"]:
            apod["copyright"] = " ".join(apod["copyright"].split())  # APOD pads credits with newlines
        return apod
