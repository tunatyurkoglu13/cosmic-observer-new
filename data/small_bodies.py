"""
data/small_bodies.py — Real near-Earth asteroids/comets, tracked via JPL
SBDB orbital elements + two-body Keplerian propagation to "now".

A small, curated set of well-known real small bodies (CURATED_BODIES) —
not an exhaustive catalog, just enough real, named objects (Eros, Bennu,
Apophis, Halley, Ceres) to populate the 3D dashboard's solar-system view
with genuine tracked objects rather than none at all.

Method: fetch each body's real osculating orbital elements from
data.sbdb.SBDBClient (JPL Small-Body Database), then propagate its mean
anomaly forward from the elements' epoch to right now and solve the
two-body (Keplerian) orbit equation — core.kepler's existing
solve_kepler_equation/true_anomaly_from_eccentric/keplerian_to_cartesian,
the same machinery this project already uses for Earth-orbit geometry,
just with the Sun's GM instead of Earth's — to get a real heliocentric
position. Earth's own heliocentric position (needed to express the
result relative to Earth, matching data.solar_system's convention) is
obtained for free: it is the negative of the Sun's Earth-centered
position data.solar_system.SolarSystemClient already fetches from
Horizons, no extra network call needed.

Honest scope: two-body Kepler propagation ignores planetary
perturbations, so the predicted position drifts from reality over time
(worse for short-period/close-approach objects) — adequate for this
dashboard's "roughly where is it right now" visualization, not for
precision targeting (JPL Horizons, used elsewhere in this project via
data.solar_system, DOES include perturbations for exactly that reason).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sgp4.api import jday

from core.kepler import (
    KeplerianElements,
    keplerian_to_cartesian,
    solve_kepler_equation,
    true_anomaly_from_eccentric,
)
from core.resilient_fetch import ResilientFetcher
from data.sbdb import OrbitalElements, SBDBClient, SmallBody

GM_SUN_KM3_S2 = 1.32712440018e11  # IAU 2015 nominal heliocentric gravitational parameter
AU_KM = 1.495978707e8             # IAU-defined astronomical unit (exact)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "small_bodies"

CURATED_BODIES: dict[str, dict] = {
    "eros": {"designation": "433", "display_name": "433 Eros", "color_hex": 0x9c8c78},
    "bennu": {"designation": "101955", "display_name": "101955 Bennu", "color_hex": 0x4a4238},
    "apophis": {"designation": "99942", "display_name": "99942 Apophis", "color_hex": 0x7a746a},
    "halley": {"designation": "1P", "display_name": "1P/Halley", "color_hex": 0xbfe0ff},
    "ceres": {"designation": "1", "display_name": "1 Ceres", "color_hex": 0x8f8b82},
}


@dataclass
class SmallBodyState:
    """Real-time Earth-relative position of one curated small body."""

    key: str
    designation: str
    display_name: str
    fetched_at: str
    distance_km: float
    direction: tuple[float, float, float]  # unit vector, Earth-relative, ecliptic frame
    orbit_class_name: str
    is_potentially_hazardous: bool


def _julian_date_now() -> float:
    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second + now.microsecond * 1e-6)
    return jd + fr


def _propagate_heliocentric_position(body: SmallBody, jd_now: float) -> np.ndarray:
    els = body.elements
    a_km = els.semi_major_axis_au * AU_KM
    n_rad_day = math.sqrt(GM_SUN_KM3_S2 / a_km**3) * 86400.0

    dt_days = jd_now - els.epoch_jd
    m_now = math.radians(els.mean_anomaly_deg) + n_rad_day * dt_days

    e = els.eccentricity
    E = solve_kepler_equation(m_now, e)
    nu = true_anomaly_from_eccentric(E, e)

    kepler_elements = KeplerianElements(
        a=a_km, e=e,
        i=math.radians(els.inclination_deg),
        raan=math.radians(els.longitude_ascending_node_deg),
        argp=math.radians(els.argument_perihelion_deg),
        nu=nu,
    )
    r_helio, _v = keplerian_to_cartesian(kepler_elements, mu=GM_SUN_KM3_S2)
    return r_helio


class SmallBodyTracker(ResilientFetcher[SmallBody]):
    """Resilient-cached SBDB orbital elements for the curated small bodies."""

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        # Osculating elements barely change day to day (unlike a live
        # position), so a much longer staleness window than TLE/CNEOS
        # data is appropriate here.
        staleness: timedelta = timedelta(days=7),
        failure_retry_cooldown: timedelta = timedelta(minutes=2),
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.staleness = staleness
        self.failure_retry_cooldown = failure_retry_cooldown
        self._sbdb = SBDBClient()

    def fetch_elements(self, key: str) -> SmallBody:
        if key not in CURATED_BODIES:
            raise ValueError(f"Unknown small body '{key}'. Supported: {list(CURATED_BODIES)}")
        return self.fetch(key)

    def current_position(self, key: str, earth_helio_r_km: tuple[float, float, float]) -> SmallBodyState:
        """
        Real-time Earth-relative position of `key`, given Earth's own
        current heliocentric position (see module docstring — callers
        typically derive this as -1 * data.solar_system's "sun" body
        position, at no extra network cost).
        """
        body = self.fetch_elements(key)
        jd_now = _julian_date_now()
        r_helio = _propagate_heliocentric_position(body, jd_now)
        r_earth = np.array(earth_helio_r_km)
        r_rel = r_helio - r_earth
        distance_km = float(np.linalg.norm(r_rel))
        direction = tuple(r_rel / distance_km)

        return SmallBodyState(
            key=key,
            designation=body.designation,
            display_name=CURATED_BODIES[key]["display_name"],
            fetched_at=datetime.now(timezone.utc).isoformat(),
            distance_km=distance_km,
            direction=direction,
            orbit_class_name=body.orbit_class_name,
            is_potentially_hazardous=body.is_potentially_hazardous,
        )

    # --- ResilientFetcher hooks -------------------------------------------------

    def _fetch_live(self, key: str) -> SmallBody:
        designation = CURATED_BODIES[key]["designation"]
        return self._sbdb.lookup(designation)

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

    def _load_cache(self, key: str) -> SmallBody | None:
        payload = self._read_cache_file(key)
        if not payload or "body" not in payload:
            return None
        rec = payload["body"]
        return SmallBody(
            designation=rec["designation"], full_name=rec["full_name"],
            orbit_class_name=rec["orbit_class_name"], orbit_class_code=rec["orbit_class_code"],
            is_neo=rec["is_neo"], is_potentially_hazardous=rec["is_potentially_hazardous"],
            elements=OrbitalElements(**rec["elements"]),
        )

    def _save_cache(self, key: str, data: SmallBody) -> None:
        self._cache_path(key).write_text(json.dumps({
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "body": asdict(data),
        }))

    def _store_fallback(self, key: str, data: SmallBody) -> None:
        self._save_cache(key, data)

    def _load_seed(self, key: str) -> SmallBody | None:
        return None  # no bundled last-known-good elements are checked into the repo

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
