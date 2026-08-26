"""
data/sbdb.py — JPL Small-Body Database (SBDB) API client.

SBDB holds orbital elements and physical parameters for every known
asteroid and comet, keyed by designation or name (e.g. "433", "Eros",
"1P/Halley"). Unlike data.nasa_cneos (impact-risk objects only) or
data.neows (near-Earth objects' close-approach data only), this is the
general-purpose lookup for "give me this body's orbit" — no keyless
restriction to a curated risk/near-Earth subset.

Endpoint: https://ssd-api.jpl.nasa.gov/sbdb.api
API docs: https://ssd-api.jpl.nasa.gov/doc/sbdb.html
Keyless — no API key required.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

SBDB_API_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"


@dataclass
class OrbitalElements:
    """Osculating orbital elements for one small body, from SBDB's `orbit.elements` array."""

    eccentricity: float
    semi_major_axis_au: float
    perihelion_distance_au: float
    inclination_deg: float
    longitude_ascending_node_deg: float
    argument_perihelion_deg: float
    mean_anomaly_deg: float
    period_days: float
    epoch_jd: float

    @classmethod
    def from_orbit_record(cls, orbit: dict) -> "OrbitalElements":
        values = {el["name"]: float(el["value"]) for el in orbit["elements"]}
        return cls(
            eccentricity=values["e"],
            semi_major_axis_au=values["a"],
            perihelion_distance_au=values["q"],
            inclination_deg=values["i"],
            longitude_ascending_node_deg=values["om"],
            argument_perihelion_deg=values["w"],
            mean_anomaly_deg=values["ma"],
            period_days=values["per"],
            epoch_jd=float(orbit["epoch"]),
        )


@dataclass
class SmallBody:
    """One SBDB record: identity + orbital elements + a few key physical/hazard flags."""

    designation: str
    full_name: str
    orbit_class_name: str
    orbit_class_code: str
    is_neo: bool
    is_potentially_hazardous: bool
    elements: OrbitalElements

    @classmethod
    def from_api_response(cls, payload: dict) -> "SmallBody":
        obj = payload["object"]
        orbit_class = obj.get("orbit_class", {})
        return cls(
            designation=obj.get("des", "unknown"),
            full_name=obj.get("fullname", obj.get("des", "unknown")).strip(),
            orbit_class_name=orbit_class.get("name", "Unknown"),
            orbit_class_code=orbit_class.get("code", ""),
            is_neo=bool(obj.get("neo", False)),
            is_potentially_hazardous=bool(obj.get("pha", False)),
            elements=OrbitalElements.from_orbit_record(payload["orbit"]),
        )


class SBDBClient:
    """Thin, keyless client over the JPL Small-Body Database API."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def lookup(self, designation_or_name: str) -> SmallBody:
        """
        Look up one small body by designation or name (e.g. "433", "Eros",
        "1P", "Halley", "Ceres", "Apophis" all resolve).

        Raises:
            ValueError: if SBDB has no match (or more than one — SBDB's
                own disambiguation-list response for an ambiguous name).
            requests.RequestException: on network failure.
        """
        resp = requests.get(
            SBDB_API_URL, params={"sstr": designation_or_name, "full-prec": "true"}, timeout=self.timeout
        )
        resp.raise_for_status()
        payload = resp.json()

        if "object" not in payload:
            if "list" in payload:
                candidates = ", ".join(c.get("name", c.get("pdes", "?")) for c in payload["list"])
                raise ValueError(f"'{designation_or_name}' is ambiguous in SBDB; candidates: {candidates}")
            raise ValueError(f"SBDB has no match for '{designation_or_name}': {payload.get('message', payload)}")

        return SmallBody.from_api_response(payload)
