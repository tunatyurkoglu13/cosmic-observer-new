"""
data/nasa_library.py — recent imagery from the NASA Image and Video Library.

Endpoint (keyless): https://images-api.nasa.gov/search?q=...&media_type=image&year_start=YYYY
Every item carries title, date, the NASA center that released it, often a
photographer credit, a description, and rendition links (~thumb ~640 px at
a few KB, ~small, ~medium ~1280 px, ~large ~1920 px, ~orig).

The search ranks by relevance, not date, and photo releases come in
bursts that share one title ("A SpaceX Dragon approaches..." x5), so a
collection is: a curated query, restricted to recent years, sorted newest
first, one image per title.
"""

from __future__ import annotations

import html
import re
from datetime import date, timedelta
from pathlib import Path

import requests

from core.resilient_fetch import JsonFileFetcher

SEARCH_URL = "https://images-api.nasa.gov/search"
DETAILS_URL = "https://images.nasa.gov/details/{nasa_id}"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "nasa_library.json"

COLLECTIONS: dict[str, dict] = {
    "orbit": {"label": "Orbit", "query": "International Space Station"},
    "earth": {"label": "Earth", "query": "Earth from space"},
    "launch": {"label": "Launch", "query": "rocket launch"},
    "moon": {"label": "Moon", "query": "Artemis Moon"},
}
DEFAULT_COLLECTION = "orbit"

ITEMS_PER_COLLECTION = 16
DESCRIPTION_MAX_CHARS = 700

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def plain_text(raw: str | None, limit: int = DESCRIPTION_MAX_CHARS) -> str:
    """Descriptions arrive with HTML tags and entities; reduce them to one clean paragraph."""
    text = _SPACE.sub(" ", html.unescape(_TAG.sub(" ", raw or ""))).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _rendition(links: list[dict], *suffixes: str) -> str | None:
    """First link whose href ends with one of `suffixes` (in preference order), upgraded to HTTPS."""
    for suffix in suffixes:
        for link in links:
            href = link.get("href", "")
            if href.endswith(suffix):
                return href.replace("http://", "https://", 1)
    return None


def normalize_item(raw: dict) -> dict | None:
    data = (raw.get("data") or [{}])[0]
    links = raw.get("links") or []
    image = _rendition(links, "~medium.jpg", "~large.jpg", "~small.jpg")
    if not data.get("nasa_id") or not image:
        return None
    return {
        "nasa_id": data["nasa_id"],
        "title": (data.get("title") or "").strip(),
        "date": (data.get("date_created") or "")[:10],
        "center": data.get("center"),
        "photographer": data.get("photographer") or data.get("secondary_creator"),
        "location": data.get("location"),
        "description": plain_text(data.get("description")),
        "thumb_url": _rendition(links, "~thumb.jpg", "~small.jpg") or image,
        "image_url": image,
        "details_url": DETAILS_URL.format(nasa_id=data["nasa_id"]),
    }


def curate(items: list[dict], limit: int = ITEMS_PER_COLLECTION) -> list[dict]:
    """Newest first, one image per title."""
    seen, curated = set(), []
    for item in sorted(items, key=lambda i: i["date"], reverse=True):
        key = item["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        curated.append(item)
        if len(curated) == limit:
            break
    return curated


class NasaLibraryClient(JsonFileFetcher[dict]):
    """Resilient-cached curated collections of recent NASA imagery."""

    def __init__(self, cache_path: Path | str = DEFAULT_CACHE_PATH, timeout: int = 30,
                 staleness: timedelta = timedelta(hours=3)):
        super().__init__(cache_path, staleness)
        self.timeout = timeout

    def fetch_collection(self, key: str, force: bool = False) -> dict:
        if key not in COLLECTIONS:
            raise ValueError(f"Unknown collection '{key}'. Valid: {list(COLLECTIONS)}")
        return self.fetch(key, force=force)

    def _fetch_live(self, key: str) -> dict:
        meta = COLLECTIONS[key]
        resp = requests.get(SEARCH_URL, timeout=self.timeout, params={
            "q": meta["query"], "media_type": "image",
            # Last year and this one: recent, but never empty in early January.
            "year_start": date.today().year - 1,
        })
        resp.raise_for_status()
        items = [i for i in (normalize_item(raw) for raw in resp.json()["collection"]["items"]) if i]
        return {"key": key, "label": meta["label"], "query": meta["query"], "items": curate(items)}
