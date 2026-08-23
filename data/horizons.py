"""
data/horizons.py — JPL Horizons ephemerides for solar-system bodies.

Horizons is JPL's high-precision ephemeris service (integrated numerically
from the DE440-class planetary/lunar ephemeris plus small-body orbital
solutions) — the authoritative source for planet/asteroid/comet state
vectors, far more precise than a Keplerian propagation for anything
outside near-Earth catalog objects (which is what SGP4/TLEs are for).

Endpoint: https://ssd.jpl.nasa.gov/api/horizons.api
API docs: https://ssd-api.jpl.nasa.gov/doc/horizons.html

We request VECTORS ephemeris type (Cartesian state vectors) in km / km-s
units directly, centered on the solar system barycenter by default, so
callers don't need to do AU<->km or day<->second conversions themselves.

The Horizons API returns one big preformatted text blob (the same text
you'd get from the telnet/webtool interface) inside a JSON envelope; the
actual vector table is delimited by "$$SOE" / "$$EOE" markers. Parsing
that text block is unavoidable — Horizons does not offer a structured
(pre-parsed) vector format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

# Common target body codes accepted by the COMMAND parameter.
BODY_CODES = {
    "sun": "10",
    "mercury": "199",
    "venus": "299",
    "earth": "399",
    "moon": "301",
    "mars": "499",
    "jupiter": "599",
    "saturn": "699",
    "uranus": "799",
    "neptune": "899",
}


@dataclass
class StateVectorSample:
    """One Cartesian state-vector sample from a Horizons vector table."""

    jd_tdb: float          # Julian date, Barycentric Dynamical Time
    r_km: tuple[float, float, float]
    v_km_s: tuple[float, float, float]


class HorizonsClient:
    """Thin client over the JPL Horizons VECTORS ephemeris API."""

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def fetch_vectors(
        self,
        command: str,
        start_time: str,
        stop_time: str,
        step_size: str = "1d",
        center: str = "500@0",
    ) -> list[StateVectorSample]:
        """
        Fetch a Cartesian state-vector time series for one body.

        Args:
            command: Horizons target body code or name (e.g. "399" for
                Earth, "499" for Mars, or a small-body designation like
                "433" for Eros / "DES=2024 YR4" style strings). See
                BODY_CODES for common planet shortcuts.
            start_time, stop_time: Horizons-format date strings, e.g.
                "2026-01-01" or "2026-01-01 00:00".
            step_size: sampling interval, e.g. "1d", "6h", "1h".
            center: origin body for the vectors, Horizons "@body" syntax.
                Default "500@0" = solar system barycenter (geometric
                center of body 0). Use "500@399" for Earth-centered.

        Returns:
            List of StateVectorSample in km / km-s, ascending time order.
        """
        command_code = BODY_CODES.get(command.lower(), command)

        params = {
            "format": "json",
            "COMMAND": f"'{command_code}'",
            "OBJ_DATA": "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "VECTORS",
            "CENTER": f"'{center}'",
            "START_TIME": f"'{start_time}'",
            "STOP_TIME": f"'{stop_time}'",
            "STEP_SIZE": f"'{step_size}'",
            "VEC_TABLE": "2",       # position + velocity only (no light-time/range extras)
            "OUT_UNITS": "KM-S",
            "REF_PLANE": "ECLIPTIC",
            "REF_SYSTEM": "ICRF",
            "VEC_LABELS": "NO",
            "CSV_FORMAT": "YES",
        }

        resp = requests.get(HORIZONS_API_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()

        if "error" in payload:
            raise RuntimeError(f"Horizons API error: {payload['error']}")

        return self._parse_vector_table(payload["result"])

    @staticmethod
    def _parse_vector_table(raw_text: str) -> list[StateVectorSample]:
        """
        Extract the CSV-formatted vector rows between $$SOE and $$EOE.

        With CSV_FORMAT=YES and VEC_TABLE=2, each row is:
            JDTDB, Calendar Date, X, Y, Z, VX, VY, VZ,
        (trailing comma is part of Horizons' own CSV output).
        """
        match = re.search(r"\$\$SOE(.*?)\$\$EOE", raw_text, re.DOTALL)
        if not match:
            raise ValueError("Could not find $$SOE/$$EOE vector block in Horizons response")

        samples = []
        for line in match.group(1).strip().splitlines():
            line = line.strip()
            if not line:
                continue
            fields = [f.strip() for f in line.split(",") if f.strip() != ""]
            # fields: [JDTDB, Calendar Date, X, Y, Z, VX, VY, VZ]
            jd_tdb = float(fields[0])
            x, y, z, vx, vy, vz = (float(f) for f in fields[2:8])
            samples.append(
                StateVectorSample(jd_tdb=jd_tdb, r_km=(x, y, z), v_km_s=(vx, vy, vz))
            )
        return samples
