"""
data/space_telescopes.py — What Hubble/JWST most recently observed.

Honest scope note (found during development): there is no public, live
"current telescope pointing" feed — spacetelescopelive.org's "current
observation" view is rendered client-side from an undocumented internal
API with no stable public contract, and hubblesite.org's old public API
domain (api.hubblesite.org) no longer resolves at all. What IS real,
public, keyless, and independently verified working is MAST (the Space
Telescope Science Institute's own archive) — its CAOM Filtered Search
API returns real observations (target name, sky coordinates, instrument,
proposal title) as they're archived, with a lag of roughly minutes to a
few hours after each exposure completes (sometimes longer for
proprietary-period science data). This module surfaces the MOST
RECENTLY ARCHIVED real observation within a several-day lookback window
— genuinely real data, honestly framed as "most recent," not literally
this-instant telemetry.

Endpoint: https://mast.stsci.edu/api/v0/invoke (Mast.Caom.Filtered
service), POST with a single urlencoded `request` field containing a
JSON body. No API key. Time filters use Modified Julian Date (MJD).
"""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from core.resilient_fetch import ResilientFetcher

MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "space_telescopes"

# This project's own keys -> MAST's own obs_collection values.
TELESCOPES = {
    "hubble": {"display_name": "Hubble Space Telescope", "obs_collection": "HST"},
    "jwst": {"display_name": "James Webb Space Telescope", "obs_collection": "JWST"},
}

# How far back to search for a recent observation — MAST archive
# ingestion lag plus scheduling gaps mean a 1-day window can come back
# empty; wide enough to reliably find something without the query
# getting so large it's slow.
LOOKBACK = timedelta(days=8)


def _datetime_to_mjd(dt: datetime) -> float:
    return dt.timestamp() / 86400 + 40587


def _mjd_to_datetime(mjd: float) -> datetime:
    return datetime.fromtimestamp((mjd - 40587) * 86400, tz=timezone.utc)


@dataclass
class TelescopeObservation:
    telescope: str
    display_name: str
    target_name: str
    ra_deg: float
    dec_deg: float
    instrument: str
    proposal_id: str
    obs_title: str
    observed_at_utc: str  # exposure end time, from MAST's t_max (MJD), converted to ISO 8601 UTC


class SpaceTelescopeClient(ResilientFetcher[TelescopeObservation]):
    """Resilient-cached client for each telescope's most recently archived real observation."""

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        timeout: int = 20,
        # New observations get archived continuously but not
        # instantaneously — no point polling MAST more than every few minutes.
        staleness: timedelta = timedelta(minutes=15),
        failure_retry_cooldown: timedelta = timedelta(minutes=2),
    ):
        self.timeout = timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.staleness = staleness
        self.failure_retry_cooldown = failure_retry_cooldown

    def fetch_latest_observation(self, telescope: str, force: bool = False) -> TelescopeObservation:
        if telescope not in TELESCOPES:
            raise ValueError(f"Unknown telescope '{telescope}'. Supported: {list(TELESCOPES)}")
        return self.fetch(telescope, force=force)

    # --- ResilientFetcher hooks -------------------------------------------------

    def _fetch_live(self, key: str) -> TelescopeObservation:
        meta = TELESCOPES[key]
        now = datetime.now(timezone.utc)
        window_start = now - LOOKBACK

        mast_request = {
            "service": "Mast.Caom.Filtered",
            "format": "json",
            "params": {
                "columns": "target_name,s_ra,s_dec,t_min,t_max,instrument_name,obs_collection,proposal_id,obs_title",
                "filters": [
                    {"paramName": "obs_collection", "values": [meta["obs_collection"]]},
                    {"paramName": "t_min", "values": [{
                        "min": _datetime_to_mjd(window_start), "max": _datetime_to_mjd(now),
                    }]},
                ],
            },
        }
        body = "request=" + urllib.parse.quote(json.dumps(mast_request))
        resp = requests.post(
            MAST_INVOKE_URL, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("data", [])
        if not rows:
            raise RuntimeError(f"MAST returned no {meta['display_name']} observations in the last {LOOKBACK.days} days")

        latest = max(rows, key=lambda r: r.get("t_min", 0))
        end_mjd = latest.get("t_max") or latest.get("t_min", 0)

        return TelescopeObservation(
            telescope=key,
            display_name=meta["display_name"],
            target_name=latest.get("target_name") or "Unknown",
            ra_deg=float(latest.get("s_ra") or 0.0),
            dec_deg=float(latest.get("s_dec") or 0.0),
            instrument=latest.get("instrument_name") or "",
            proposal_id=str(latest.get("proposal_id") or ""),
            obs_title=latest.get("obs_title") or "",
            observed_at_utc=_mjd_to_datetime(end_mjd).isoformat(),
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

    def _load_cache(self, key: str) -> TelescopeObservation | None:
        payload = self._read_cache_file(key)
        if not payload or "observation" not in payload:
            return None
        return TelescopeObservation(**payload["observation"])

    def _save_cache(self, key: str, data: TelescopeObservation) -> None:
        self._cache_path(key).write_text(json.dumps({
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "observation": asdict(data),
        }))

    def _store_fallback(self, key: str, data: TelescopeObservation) -> None:
        self._save_cache(key, data)

    def _load_seed(self, key: str) -> TelescopeObservation | None:
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
