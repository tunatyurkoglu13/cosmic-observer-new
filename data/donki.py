"""
data/donki.py — NASA DONKI (Space Weather Database Of Notifications,
Knowledge, Information): solar flares, coronal mass ejections and
geomagnetic storms as analysed by the Moon to Mars Space Weather Analysis
Office at NASA GSFC.

Endpoints (NASA API key; DEMO_KEY works but is heavily rate-limited):
    https://api.nasa.gov/DONKI/FLR   solar flares
    https://api.nasa.gov/DONKI/CME   coronal mass ejections (+ WSA-ENLIL runs)
    https://api.nasa.gov/DONKI/GST   geomagnetic storms (+ observed Kp)

The key travels in the X-Api-Key header rather than the query string, so
it never appears in a URL — and therefore never in an exception message,
a log line or a published error.

Raw DONKI records are normalised into compact events, and `assess_*`
turns each into an alert severity for the active alert layer:
    flares   X-class -> critical, M-class -> warning
    CMEs     a modelled Earth arrival -> warning (critical at >= 1000 km/s)
    storms   Kp >= 7 (G3+) -> critical, Kp >= 5 (G1+) -> warning
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import requests

from core.resilient_fetch import JsonFileFetcher

DONKI_BASE_URL = "https://api.nasa.gov/DONKI"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "donki_events.json"

FLARE_CLASS_ORDER = "ABCMX"
FAST_CME_KM_S = 1000.0
NOTE_MAX_CHARS = 280


def _api_key() -> str:
    return os.environ.get("NASA_API_KEY") or "DEMO_KEY"


def donki_get(endpoint: str, start: date, end: date, timeout: int = 60) -> list[dict]:
    resp = requests.get(
        f"{DONKI_BASE_URL}/{endpoint}",
        params={"startDate": start.isoformat(), "endDate": end.isoformat()},
        headers={"X-Api-Key": _api_key()},
        timeout=timeout,
    )
    resp.raise_for_status()
    # DONKI answers a window with no events with an empty body, not "[]".
    return resp.json() if resp.content.strip() else []


def flare_rank(class_type: str | None) -> float:
    """Sortable flare strength: "M2.4" -> 3.24 (letter index + magnitude / 10)."""
    if not class_type or class_type[0] not in FLARE_CLASS_ORDER:
        return 0.0
    try:
        magnitude = float(class_type[1:])
    except ValueError:
        magnitude = 1.0
    return FLARE_CLASS_ORDER.index(class_type[0]) + min(magnitude, 9.9) / 10


def normalize_flare(raw: dict) -> dict:
    return {
        "id": raw.get("flrID"),
        "class": raw.get("classType"),
        "begin": raw.get("beginTime"),
        "peak": raw.get("peakTime"),
        "end": raw.get("endTime"),
        "source_location": raw.get("sourceLocation"),
        "active_region": raw.get("activeRegionNum"),
        "link": raw.get("link"),
    }


def normalize_cme(raw: dict) -> dict:
    analyses = raw.get("cmeAnalyses") or []
    best = next((a for a in analyses if a.get("isMostAccurate")), analyses[0] if analyses else {})
    arrivals = [
        run.get("estimatedShockArrivalTime")
        for a in analyses for run in (a.get("enlilList") or [])
        if run.get("estimatedShockArrivalTime")
    ]
    note = (raw.get("note") or "").strip()
    return {
        "id": raw.get("activityID"),
        "start": raw.get("startTime"),
        "source_location": raw.get("sourceLocation"),
        "speed_km_s": best.get("speed"),
        "half_angle_deg": best.get("halfAngle"),
        "type": best.get("type"),
        "earth_arrival": min(arrivals) if arrivals else None,
        "note": note[:NOTE_MAX_CHARS] + ("…" if len(note) > NOTE_MAX_CHARS else ""),
        "link": raw.get("link"),
    }


def normalize_storm(raw: dict) -> dict:
    kp_values = [k["kpIndex"] for k in raw.get("allKpIndex") or [] if k.get("kpIndex") is not None]
    return {
        "id": raw.get("gstID"),
        "start": raw.get("startTime"),
        "kp_max": max(kp_values) if kp_values else None,
        "link": raw.get("link"),
    }


def assess_flare(flare: dict) -> str | None:
    letter = (flare.get("class") or " ")[0]
    return {"X": "critical", "M": "warning"}.get(letter)


def assess_cme(cme: dict) -> str | None:
    if not cme.get("earth_arrival"):
        return None
    return "critical" if (cme.get("speed_km_s") or 0) >= FAST_CME_KM_S else "warning"


def assess_storm(storm: dict) -> str | None:
    kp = storm.get("kp_max") or 0
    return "critical" if kp >= 7 else "warning" if kp >= 5 else None


class DONKIClient(JsonFileFetcher[dict]):
    """Resilient-cached client for recent DONKI flares, CMEs and geomagnetic storms."""

    def __init__(self, cache_path: Path | str = DEFAULT_CACHE_PATH, timeout: int = 60,
                 staleness: timedelta = timedelta(minutes=30)):
        super().__init__(cache_path, staleness)
        self.timeout = timeout

    def fetch_recent(self, days: int = 14, force: bool = False) -> dict:
        return self.fetch(f"recent:{days}", force=force)

    def _fetch_live(self, key: str) -> dict:
        days = int(key.split(":")[1])
        end = date.today()
        start = end - timedelta(days=days)
        flares = [normalize_flare(r) for r in donki_get("FLR", start, end, self.timeout)]
        cmes = [normalize_cme(r) for r in donki_get("CME", start, end, self.timeout)]
        storms = [normalize_storm(r) for r in donki_get("GST", start, end, self.timeout)]

        flares.sort(key=lambda f: f["peak"] or f["begin"] or "", reverse=True)
        cmes.sort(key=lambda c: c["start"] or "", reverse=True)
        storms.sort(key=lambda s: s["start"] or "", reverse=True)
        strongest = max(flares, key=lambda f: flare_rank(f["class"]), default=None)
        return {
            "days": days,
            "summary": {
                "flares": len(flares),
                "flares_by_class": {c: sum(1 for f in flares if (f["class"] or "").startswith(c)) for c in "CMX"},
                "strongest_flare": strongest["class"] if strongest else None,
                "cmes": len(cmes),
                "earth_directed_cmes": sum(1 for c in cmes if c["earth_arrival"]),
                "storms": len(storms),
                "max_kp": max((s["kp_max"] for s in storms if s["kp_max"] is not None), default=None),
            },
            "flares": flares,
            "cmes": cmes,
            "storms": storms,
        }
