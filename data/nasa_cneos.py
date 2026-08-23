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
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

SENTRY_API_URL = "https://ssd-api.jpl.nasa.gov/sentry.api"

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


class CNEOSClient:
    """Thin client over the JPL Sentry API."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def fetch_risk_list(self) -> list[SentryObject]:
        """
        Fetch the full current Sentry risk list (all monitored objects).

        Returns:
            list[SentryObject], sorted by descending Torino scale then
            descending cumulative Palermo scale (most concerning first).
        """
        resp = requests.get(SENTRY_API_URL, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()

        records = payload.get("data", [])
        objects = [SentryObject.from_api_record(r) for r in records]
        objects.sort(key=lambda o: (o.torino_scale_max, o.palermo_scale_cum), reverse=True)
        return objects

    def fetch_object(self, designation: str) -> dict:
        """
        Fetch full Sentry detail for a single object by designation (e.g.
        "99942" for Apophis), including its individual virtual-impactor
        table. Returned as the raw JSON dict since the per-impactor schema
        is nested and use-case-specific (unlike the flat risk-list schema
        modeled by SentryObject).
        """
        resp = requests.get(SENTRY_API_URL, params={"des": designation}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def objects_above_torino(self, threshold: int = 1) -> list[SentryObject]:
        """Convenience filter: risk-list objects at or above a given Torino Scale value."""
        return [o for o in self.fetch_risk_list() if o.torino_scale_max >= threshold]
