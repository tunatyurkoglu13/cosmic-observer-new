"""
data/eonet.py — NASA EONET (Earth Observatory Natural Event Tracker).

EONET curates natural events that are being observed from orbit right now
— wildfires, tropical cyclones, volcanic eruptions, sea and lake ice,
floods — each with a dated geometry history: one Point for a wildfire,
a track of Points for a cyclone, occasionally a Polygon for an area.

Endpoint (keyless): https://eonet.gsfc.nasa.gov/api/v3/events?status=open&days=N

Events are normalised to: the latest position (for a globe marker), the
full point track (for storms), and the latest magnitude if EONET reports
one (e.g. wind speed in kts for cyclones, acres for some wildfires).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import requests

from core.resilient_fetch import JsonFileFetcher

EONET_EVENTS_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "eonet_events.json"

# Display order and colours for the globe legend. Anything else EONET
# adds later is still returned, under its own category id.
CATEGORIES: dict[str, dict] = {
    "wildfires": {"label": "Wildfire", "color_hex": "#ff5a1f"},
    "severeStorms": {"label": "Severe storm", "color_hex": "#7df9ff"},
    "volcanoes": {"label": "Volcano", "color_hex": "#ff2d55"},
    "seaLakeIce": {"label": "Sea & lake ice", "color_hex": "#e6f4ff"},
    "floods": {"label": "Flood", "color_hex": "#3d8bff"},
    "dustHaze": {"label": "Dust & haze", "color_hex": "#d9b26f"},
}


def _centroid(geometry: dict) -> tuple[float, float] | None:
    """(lat, lon) for a GeoJSON Point or the mean vertex of a Polygon's outer ring."""
    coords = geometry.get("coordinates")
    if geometry.get("type") == "Point" and coords:
        return float(coords[1]), float(coords[0])
    if geometry.get("type") == "Polygon" and coords and coords[0]:
        ring = coords[0]
        return sum(p[1] for p in ring) / len(ring), sum(p[0] for p in ring) / len(ring)
    return None


def normalize_event(raw: dict) -> dict | None:
    """One EONET event -> the shape the dashboard draws, or None if it has no usable geometry."""
    geometries = sorted(raw.get("geometry", []), key=lambda g: g.get("date", ""))
    track = []
    for g in geometries:
        position = _centroid(g)
        if position is not None:
            track.append({"lat": position[0], "lon": position[1], "date": g.get("date")})
    if not track:
        return None

    latest = geometries[-1]
    category = (raw.get("categories") or [{}])[0]
    return {
        "id": raw["id"],
        "title": raw.get("title", ""),
        "category": category.get("id", "unknown"),
        "category_title": category.get("title", ""),
        "lat": track[-1]["lat"],
        "lon": track[-1]["lon"],
        "last_observed": track[-1]["date"],
        "magnitude": latest.get("magnitudeValue"),
        "magnitude_unit": latest.get("magnitudeUnit"),
        "track": track,
        "sources": [{"id": s.get("id"), "url": s.get("url")} for s in raw.get("sources", [])],
        "link": raw.get("link"),
    }


class EONETClient(JsonFileFetcher[dict]):
    """Resilient-cached client for currently open EONET events."""

    def __init__(self, cache_path: Path | str = DEFAULT_CACHE_PATH, timeout: int = 20,
                 staleness: timedelta = timedelta(minutes=30)):
        super().__init__(cache_path, staleness)
        self.timeout = timeout

    def fetch_open_events(self, days: int = 30, force: bool = False) -> dict:
        return self.fetch(f"open:{days}", force=force)

    def _fetch_live(self, key: str) -> dict:
        days = int(key.split(":")[1])
        resp = requests.get(EONET_EVENTS_URL, params={"status": "open", "days": days}, timeout=self.timeout)
        resp.raise_for_status()
        events = [e for e in (normalize_event(raw) for raw in resp.json().get("events", [])) if e]
        events.sort(key=lambda e: e["last_observed"] or "", reverse=True)
        return {"days": days, "categories": CATEGORIES, "events": events}
