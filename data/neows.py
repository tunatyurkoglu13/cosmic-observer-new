"""
data/neows.py — NASA NeoWs (Near Earth Object Web Service) API client.

Distinct from data.nasa_cneos (JPL Sentry's impact-risk list): NeoWs
answers "which known near-Earth objects pass close to Earth in this date
range, and how close/fast" — ordinary catalogued close approaches, not
computed future impact probabilities. The two are complementary risk
signals, not overlapping ones.

Endpoint base: https://api.nasa.gov/neo/rest/v1/
API docs: https://api.nasa.gov/ (see "Asteroids - NeoWs")
Requires an api.nasa.gov API key (get one free at https://api.nasa.gov/;
DEMO_KEY works for light/occasional use but is rate-limited far below a
real key — same NASA_API_KEY convention as data.space_weather).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta

import requests

NEOWS_BASE_URL = "https://api.nasa.gov/neo/rest/v1/"


@dataclass
class CloseApproach:
    """One close-approach event for a near-Earth object."""

    approach_date: str
    orbiting_body: str
    relative_velocity_km_s: float
    miss_distance_km: float
    miss_distance_lunar: float


@dataclass
class NearEarthObject:
    """One NeoWs object: identity, size estimate, hazard flag, and its close-approach history."""

    neo_reference_id: str
    name: str
    estimated_diameter_min_km: float
    estimated_diameter_max_km: float
    is_potentially_hazardous: bool
    close_approaches: list[CloseApproach]

    @classmethod
    def from_api_record(cls, rec: dict) -> "NearEarthObject":
        diam = rec.get("estimated_diameter", {}).get("kilometers", {})
        approaches = [
            CloseApproach(
                approach_date=ca.get("close_approach_date", ""),
                orbiting_body=ca.get("orbiting_body", ""),
                relative_velocity_km_s=float(ca["relative_velocity"]["kilometers_per_second"]),
                miss_distance_km=float(ca["miss_distance"]["kilometers"]),
                miss_distance_lunar=float(ca["miss_distance"]["lunar"]),
            )
            for ca in rec.get("close_approach_data", [])
        ]
        return cls(
            neo_reference_id=rec.get("neo_reference_id", rec.get("id", "unknown")),
            name=rec.get("name", "unknown"),
            estimated_diameter_min_km=float(diam.get("estimated_diameter_min", 0.0)),
            estimated_diameter_max_km=float(diam.get("estimated_diameter_max", 0.0)),
            is_potentially_hazardous=bool(rec.get("is_potentially_hazardous_asteroid", False)),
            close_approaches=approaches,
        )


class NeoWsClient:
    """Thin client over NASA's NeoWs API."""

    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("NASA_API_KEY", "DEMO_KEY")
        self.timeout = timeout

    def feed(self, start_date: date, end_date: date | None = None) -> list[NearEarthObject]:
        """
        Fetch all near-Earth objects making a close approach within
        [start_date, end_date] (NeoWs caps the range at 7 days per call).

        Args:
            end_date: defaults to start_date (a single day).
        """
        end_date = end_date or start_date
        if (end_date - start_date) > timedelta(days=7):
            raise ValueError("NeoWs feed() covers at most a 7-day range per call.")

        resp = requests.get(
            NEOWS_BASE_URL + "feed",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "api_key": self.api_key,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()

        objects = []
        for day_records in payload.get("near_earth_objects", {}).values():
            objects.extend(NearEarthObject.from_api_record(r) for r in day_records)
        return objects

    def lookup(self, neo_id: str) -> NearEarthObject:
        """Fetch full detail (including complete close-approach history) for one object by its NeoWs id."""
        resp = requests.get(
            NEOWS_BASE_URL + f"neo/{neo_id}", params={"api_key": self.api_key}, timeout=self.timeout
        )
        resp.raise_for_status()
        return NearEarthObject.from_api_record(resp.json())

    def closest_approach(self, start_date: date, end_date: date | None = None) -> NearEarthObject | None:
        """Convenience: the single object with the smallest miss distance in the given date range."""
        objects = self.feed(start_date, end_date)
        candidates = [o for o in objects if o.close_approaches]
        if not candidates:
            return None
        return min(candidates, key=lambda o: min(ca.miss_distance_km for ca in o.close_approaches))
