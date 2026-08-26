"""
data/nasa_cneos.py — NASA/JPL CNEOS Sentry risk list (Near-Earth Objects).

Sentry is JPL's automated system that continually scans the known
asteroid catalog for future impact possibilities and publishes a "risk
list" of objects with a non-zero computed impact probability. This module
wraps the public Sentry API (no key required) and classifies objects on
the Torino Scale, the 0-10 public-communication scale for impact hazard
(0 = no hazard, 10 = certain global catastrophe).

Endpoint: https://ssd-api.jpl.nasa.gov/sentry.api
API docs: https://ssd-api.jpl.nasa.gov/doc/sentry.html

Two other risk scales appear in the raw API response and are carried
through here rather than recomputed:
  - Palermo Scale (ps): log10 of (object's impact probability / annual
    background risk from objects of the same size). ps < -2 -> not
    worth attention; ps > 0 -> merits attention.
  - Torino Scale (ts): the simplified 0-10 scale derived from combining
    impact probability and kinetic energy, intended for public
    communication (Palermo is the finer-grained research scale).

Caching: the risk list is cached to a small local JSON file and served
through core.resilient_fetch.ResilientFetcher's network -> cache ->
cooldown chain (the same pattern core.tle_manager.TLEManager uses for
CelesTrak) — this client previously had no resilience at all (a bare
requests.get() with no fallback), so a transient JPL outage would
surface directly as an error to callers instead of degrading to
recently-cached data. There's no bundled "seed" fallback for this one
(unlike TLE data, there's no small last-known-good risk list checked
into the repo) — on a cold cache with no network, this simply raises.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from core.resilient_fetch import ResilientFetcher

SENTRY_API_URL = "https://ssd-api.jpl.nasa.gov/sentry.api"

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "cneos_risk_list.json"

TORINO_DESCRIPTIONS = {
    0: "No Hazard",
    1: "Normal",
    2: "Meriting Attention",
    3: "Meriting Attention",
    4: "Meriting Attention",
    5: "Threatening",
    6: "Threatening",
    7: "Threatening",
    8: "Certain Collision — Local",
    9: "Certain Collision — Regional",
    10: "Certain Collision — Global",
}

# The one key this client's resilient cache is keyed on — there's only
# ever one Sentry risk list, but ResilientFetcher's hooks are keyed, so
# we give it a fixed name rather than special-casing a keyless variant.
RISK_LIST_KEY = "risk_list"


@dataclass
class SentryObject:
    """One object on the current Sentry risk list."""

    designation: str
    full_name: str
    diameter_km: float | None       # estimated diameter, may be unknown
    torino_scale_max: int
    palermo_scale_cum: float        # cumulative Palermo scale over all virtual impactors
    palermo_scale_max: float        # Palermo scale of the single most dangerous virtual impactor
    impact_probability_cum: float   # cumulative impact probability, all potential impacts
    potential_impact_count: int     # "n_imp": number of distinct potential impact dates
    years_observed: str             # "range": date range the object is being monitored over
    last_observed: str | None       # "last_obs" date string, if provided

    @property
    def torino_description(self) -> str:
        return TORINO_DESCRIPTIONS.get(self.torino_scale_max, "Unknown")

    @property
    def risk_flag(self) -> str:
        """Simple traffic-light classification for dashboard display."""
        if self.torino_scale_max >= 5:
            return "red"
        if self.torino_scale_max >= 1:
            return "yellow"
        return "green"

    @classmethod
    def from_api_record(cls, rec: dict) -> "SentryObject":
        def _float(key, default=None):
            v = rec.get(key)
            try:
                return float(v) if v not in (None, "") else default
            except (TypeError, ValueError):
                return default

        def _int(key, default=0):
            v = rec.get(key)
            try:
                return int(float(v)) if v not in (None, "") else default
            except (TypeError, ValueError):
                return default

        return cls(
            designation=rec.get("des", "unknown"),
            full_name=rec.get("fullname", rec.get("des", "unknown")).strip(),
            diameter_km=_float("diameter"),
            torino_scale_max=_int("ts_max", 0),
            palermo_scale_cum=_float("ps_cum", -99.0),
            palermo_scale_max=_float("ps_max", -99.0),
            impact_probability_cum=_float("ip", 0.0),
            potential_impact_count=_int("n_imp", 0),
            years_observed=rec.get("range", ""),
            last_observed=rec.get("last_obs"),
        )


class CNEOSClient(ResilientFetcher[list[SentryObject]]):
    """Thin, cache-resilient client over the JPL Sentry API."""

    def __init__(
        self,
        timeout: int = 30,
        cache_path: Path | str = DEFAULT_CACHE_PATH,
        staleness: timedelta = timedelta(hours=6),
        failure_retry_cooldown: timedelta = timedelta(minutes=2),
    ):
        self.timeout = timeout
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.staleness = staleness
        self.failure_retry_cooldown = failure_retry_cooldown

    def fetch_risk_list(self, force: bool = False, allow_stale_fallback: bool = True) -> list[SentryObject]:
        """
        Fetch the full current Sentry risk list (all monitored objects),
        served through the network -> cache -> cooldown resilience chain.

        Returns:
            list[SentryObject], sorted by descending Torino scale then
            descending cumulative Palermo scale (most concerning first).
        """
        return self.fetch(RISK_LIST_KEY, force=force, allow_fallback=allow_stale_fallback)

    def fetch_object(self, designation: str) -> dict:
        """
        Fetch full Sentry detail for a single object by designation (e.g.
        "99942" for Apophis), including its individual virtual-impactor
        table. Returned as the raw JSON dict since the per-impactor schema
        is nested and use-case-specific (unlike the flat risk-list schema
        modeled by SentryObject). Not cached: this is a targeted lookup,
        not the bulk resource the resilience chain is built around.
        """
        resp = requests.get(SENTRY_API_URL, params={"des": designation}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def objects_above_torino(self, threshold: int = 1) -> list[SentryObject]:
        """Convenience filter: risk-list objects at or above a given Torino Scale value."""
        return [o for o in self.fetch_risk_list() if o.torino_scale_max >= threshold]

    # --- ResilientFetcher hooks -------------------------------------------------

    def _fetch_live(self, key: str) -> list[SentryObject]:
        resp = requests.get(SENTRY_API_URL, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()

        records = payload.get("data", [])
        objects = [SentryObject.from_api_record(r) for r in records]
        objects.sort(key=lambda o: (o.torino_scale_max, o.palermo_scale_cum), reverse=True)
        return objects

    def _read_cache_file(self) -> dict | None:
        if not self.cache_path.exists():
            return None
        try:
            return json.loads(self.cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _load_cache(self, key: str) -> list[SentryObject]:
        payload = self._read_cache_file()
        if not payload:
            return []
        return [SentryObject(**rec) for rec in payload.get("objects", [])]

    def _save_cache(self, key: str, data: list[SentryObject]) -> None:
        self.cache_path.write_text(json.dumps({
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "objects": [asdict(o) for o in data],
        }))

    def _store_fallback(self, key: str, data: list[SentryObject]) -> None:
        # No bundled seed exists for this client (see module docstring),
        # so this is never actually exercised, but is implemented for
        # interface completeness/consistency with TLEManager.
        self._save_cache(key, data)

    def _load_seed(self, key: str) -> list[SentryObject]:
        return []  # no bundled last-known-good risk list is checked into the repo

    def _is_stale(self, key: str) -> bool:
        payload = self._read_cache_file()
        if not payload or "fetched_at" not in payload:
            return True
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        return datetime.now(timezone.utc) - fetched_at > self.staleness

    def _recently_failed(self, key: str) -> bool:
        payload = self._read_cache_file()
        if not payload or "failed_at" not in payload:
            return False
        failed_at = datetime.fromisoformat(payload["failed_at"])
        return datetime.now(timezone.utc) - failed_at <= self.failure_retry_cooldown

    def _record_failure(self, key: str) -> None:
        payload = self._read_cache_file() or {"objects": []}
        payload["failed_at"] = datetime.now(timezone.utc).isoformat()
        self.cache_path.write_text(json.dumps(payload))

    def _clear_failure(self, key: str) -> None:
        payload = self._read_cache_file()
        if payload and "failed_at" in payload:
            del payload["failed_at"]
            self.cache_path.write_text(json.dumps(payload))
