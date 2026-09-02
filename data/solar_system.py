"""
data/solar_system.py — Real-time Earth-relative position of solar-system
bodies, for the 3D dashboard's "travel to a celestial body" feature.

Wraps data.horizons.HorizonsClient (JPL Horizons) to answer one specific
question per body: "right now, which direction is it from Earth, and how
far?" — the Earth-centered Cartesian vector Horizons returns directly
gives both (direction = normalized vector, distance = its magnitude), no
further astrodynamics needed for this purpose.

This is a real, verified geometric direction — not a stylized guess —
but the DISPLAY distance the frontend places the body at is a
deliberately compressed dramatization (real interplanetary distances are
many orders of magnitude larger than this dashboard's scene scale) so
the "travel" animation stays navigable; see static/dashboard.js's
DISPLAY_DISTANCE_SCALE for exactly how that compression works, and
BODIES below for each body's true physical radius (which the frontend
DOES render to real relative scale, matching how it already sizes Earth).

Cached via core.resilient_fetch.ResilientFetcher (same pattern as
core.tle_manager.TLEManager and data.nasa_cneos.CNEOSClient) since
Horizons queries take a second or two and planetary geometry changes
slowly enough that an hours-old position is still a perfectly good
approximation for this dashboard's purposes. No bundled seed fallback is
provided (unlike TLE data) — there's no small last-known-good planetary
ephemeris checked into the repo, so a cold cache with no network simply
raises, same as data.nasa_cneos.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from core.resilient_fetch import ResilientFetcher

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "solar_system"

# Real physical radius (km) and a muted, realistic (not photographic,
# not garish) base display color for each supported body. Radii are
# well-established astronomical constants. Colors approximate each
# body's true low-saturation appearance (e.g. Mercury's dark, warm-gray
# regolith, comparable in albedo to the Moon's) rather than an
# "enhanced/false color" scientific-imagery palette.
BODIES: dict[str, dict] = {
    "sun": {
        "display_name": "Sun",
        "radius_km": 696000.0,
        "color_hex": 0xfff2c2,
    },
    "mercury": {
        "display_name": "Mercury",
        "radius_km": 2439.7,
        "color_hex": 0x8a8175,
    },
    "venus": {
        "display_name": "Venus",
        "radius_km": 6051.8,
        # Venus's real appearance: a near-featureless, bright pale
        # yellow-white sulfuric-acid cloud deck (true color, not the
        # false-color radar-mapped surface images most people picture).
        "color_hex": 0xd9c48a,
    },
    "mars": {
        "display_name": "Mars",
        "radius_km": 3389.5,
        "color_hex": 0xb1552f,
    },
    "jupiter": {
        "display_name": "Jupiter",
        "radius_km": 69911.0,
        # Jupiter's real true-color appearance: pale cream/tan cloud
        # bands (the tactical shader's own scanline pattern happens to
        # double as a fitting stylization of its actual banded structure).
        "color_hex": 0xc9a876,
    },
    "saturn": {
        "display_name": "Saturn",
        "radius_km": 58232.0,
        # Saturn's real true-color appearance: pale gold/cream, paler
        # and less contrasty-banded than Jupiter. Has a real ring system
        # rendered separately — see static/dashboard.js's
        # _addPlanetRings, keyed off bodyKey === "saturn"/"uranus".
        "color_hex": 0xd9c99a,
    },
    "uranus": {
        "display_name": "Uranus",
        "radius_km": 25362.0,
        # Uranus's real true-color appearance: a pale, near-featureless
        # cyan (methane absorbing red light) — genuinely paler/less
        # saturated than most illustrations. Also has real rings (fainter
        # and narrower than Saturn's, discovered 1977) — see
        # static/dashboard.js's _addPlanetRings.
        "color_hex": 0x9fd6d6,
    },
    "neptune": {
        "display_name": "Neptune",
        "radius_km": 24622.0,
        # Neptune's real true-color appearance: the most vividly blue
        # planet — methane absorption alone doesn't fully explain its
        # saturation; an additional, still-unidentified atmospheric
        # absorber is the leading explanation.
        "color_hex": 0x3d63c9,
    },
}

# Mars's two real natural satellites (Mercury and Venus have none — not
# an omission, a fact: this project doesn't invent moons that don't
# exist). Unlike BODIES above (positioned relative to Earth), a moon's
# meaningful position is relative to its OWN parent planet — see
# fetch_moon_position()/relative_to on BodyPosition.
MOONS: dict[str, dict] = {
    "phobos": {
        "display_name": "Phobos",
        "parent": "mars",
        "horizons_command": "401",  # not in data.horizons.BODY_CODES (planets/Sun/Moon only) — passed straight through as a raw Horizons body code
        "radius_km": 11.08,
        "color_hex": 0x8a7a68,
    },
    "deimos": {
        "display_name": "Deimos",
        "parent": "mars",
        "horizons_command": "402",
        "radius_km": 6.27,
        "color_hex": 0x9c8f7c,
    },
    "moon": {
        "display_name": "Moon",
        "parent": "earth",
        "horizons_command": "moon",  # matches data.horizons.BODY_CODES directly
        "radius_km": 1737.4,
        "color_hex": 0xb8b3aa,
    },
    # Jupiter's four real Galilean moons — by far its most significant
    # satellites (of Jupiter's 90+ known moons, these are the only ones
    # large enough to be individually meaningful at this project's scale
    # and the only ones worth naming here; not claiming Jupiter has only
    # four moons, just that these are the ones this dashboard renders).
    "io": {
        "display_name": "Io",
        "parent": "jupiter",
        "horizons_command": "501",
        "radius_km": 1821.6,
        "color_hex": 0xd9c36b,  # sulfur-yellow, Io's real, famously vivid surface
    },
    "europa": {
        "display_name": "Europa",
        "parent": "jupiter",
        "horizons_command": "502",
        "radius_km": 1560.8,
        "color_hex": 0xd8c9ab,  # icy pale tan
    },
    "ganymede": {
        "display_name": "Ganymede",
        "parent": "jupiter",
        "horizons_command": "503",
        "radius_km": 2634.1,  # the solar system's largest moon, bigger than Mercury
        "color_hex": 0x8f8577,
    },
    "callisto": {
        "display_name": "Callisto",
        "parent": "jupiter",
        "horizons_command": "504",
        "radius_km": 2410.3,
        "color_hex": 0x6f6459,  # Callisto is the darkest, most heavily cratered Galilean moon
    },
    # Saturn's most significant real moons, of its 140+ known — same
    # "only the individually meaningful ones" scope as Jupiter's four.
    "mimas": {
        "display_name": "Mimas",
        "parent": "saturn",
        "horizons_command": "601",
        "radius_km": 198.2,
        "color_hex": 0xaba398,  # icy pale gray
    },
    "enceladus": {
        "display_name": "Enceladus",
        "parent": "saturn",
        "horizons_command": "602",
        "radius_km": 252.1,
        "color_hex": 0xe8e6de,  # near-white — the most reflective body in the solar system
    },
    "rhea": {
        "display_name": "Rhea",
        "parent": "saturn",
        "horizons_command": "605",
        "radius_km": 763.5,
        "color_hex": 0xb8b0a4,
    },
    "titan": {
        "display_name": "Titan",
        "parent": "saturn",
        "horizons_command": "606",
        "radius_km": 2574.7,  # larger than Mercury; the solar system's 2nd-largest moon
        "color_hex": 0xd9a85c,  # Titan's real hazy orange nitrogen/methane atmosphere
    },
    # Helene: a real "Trojan moon" — not just a nearby object, but one
    # genuinely co-orbital with a larger moon (Dione), librating around
    # Dione's leading L4 Lagrange point, the same real dynamical
    # relationship Jupiter's Trojan asteroids have with Jupiter itself.
    "helene": {
        "display_name": "Helene",
        "parent": "saturn",
        "horizons_command": "612",
        "radius_km": 17.6,
        "color_hex": 0x9c9488,
    },
    # Uranus's five real major moons — all named after Shakespeare/Pope
    # characters (uniquely, among moons of the solar system, which are
    # otherwise almost all named from Greco-Roman myth).
    "miranda": {
        "display_name": "Miranda",
        "parent": "uranus",
        "horizons_command": "705",
        "radius_km": 235.8,
        "color_hex": 0x9a9690,
    },
    "ariel": {
        "display_name": "Ariel",
        "parent": "uranus",
        "horizons_command": "701",
        "radius_km": 578.9,
        "color_hex": 0xaeada8,  # Ariel has the brightest surface of Uranus's moons
    },
    "umbriel": {
        "display_name": "Umbriel",
        "parent": "uranus",
        "horizons_command": "702",
        "radius_km": 584.7,
        "color_hex": 0x655f5a,  # the darkest of Uranus's five major moons
    },
    "titania": {
        "display_name": "Titania",
        "parent": "uranus",
        "horizons_command": "703",
        "radius_km": 788.4,  # Uranus's largest moon
        "color_hex": 0x8d8880,
    },
    "oberon": {
        "display_name": "Oberon",
        "parent": "uranus",
        "horizons_command": "704",
        "radius_km": 761.4,
        "color_hex": 0x7d766e,
    },
    # Neptune's real moons. Triton is overwhelmingly its dominant one —
    # captured from the Kuiper Belt (a real, well-established fact: it's
    # the solar system's only large moon on a retrograde orbit, powerful
    # evidence it didn't form alongside Neptune).
    "triton": {
        "display_name": "Triton",
        "parent": "neptune",
        "horizons_command": "801",
        "radius_km": 1353.4,
        "color_hex": 0xd8cdb8,  # pale pinkish nitrogen-frost surface
    },
    "proteus": {
        "display_name": "Proteus",
        "parent": "neptune",
        "horizons_command": "808",
        "radius_km": 209.0,  # Neptune's 2nd-largest moon, despite being irregularly shaped and very dark
        "color_hex": 0x6b6560,
    },
    "nereid": {
        "display_name": "Nereid",
        "parent": "neptune",
        "horizons_command": "802",
        # One of the most eccentric orbits of any moon in the solar
        # system (e~0.75) — its real Neptune-relative distance varies
        # enormously, unlike every other moon here.
        "radius_km": 170.0,
        "color_hex": 0x9a9488,
    },
}


@dataclass
class BodyPosition:
    """Earth-centered real-time position of one body, from JPL Horizons."""

    body: str
    fetched_at: str          # ISO 8601 UTC — when this was fetched from Horizons
    jd_tdb: float             # Julian date (TDB) of the Horizons sample used
    r_km: tuple[float, float, float]   # Cartesian position, km, centered on `relative_to`
    distance_km: float
    direction: tuple[float, float, float]  # unit vector, same frame as r_km
    # Real direction FROM this body TOWARD the Sun (unit vector, same
    # frame) — used to light the body correctly (its true terminator),
    # rather than an arbitrary fixed light direction. Meaningless for the
    # Sun itself, which reports (0.0, 0.0, 0.0).
    sun_direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # What r_km/direction/distance_km are centered on: "earth" for every
    # BODIES entry (planets, Sun); a parent planet's key (e.g. "mars")
    # for a MOONS entry, since a moon's meaningful position is relative
    # to what it actually orbits, not to Earth.
    relative_to: str = "earth"


class SolarSystemClient(ResilientFetcher[BodyPosition]):
    """Real-time Earth-relative body positions, resilient-cached per body."""

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        staleness: timedelta = timedelta(hours=6),
        failure_retry_cooldown: timedelta = timedelta(minutes=2),
        timeout: int = 60,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.staleness = staleness
        self.failure_retry_cooldown = failure_retry_cooldown
        self.timeout = timeout

    def fetch_position(self, body: str, force: bool = False) -> BodyPosition:
        """
        Real-time Earth-centered position of `body` (a key in BODIES).

        Raises:
            ValueError: unknown body.
            requests.RequestException: Horizons unreachable and no cache available.
        """
        if body not in BODIES:
            raise ValueError(f"Unknown body '{body}'. Supported: {list(BODIES)}")
        return self.fetch(body, force=force)

    def fetch_moon_position(self, moon: str, force: bool = False) -> BodyPosition:
        """
        Real-time position of `moon` (a key in MOONS) relative to its own
        parent planet (see BodyPosition.relative_to) — NOT relative to
        Earth, since that's what actually places it correctly next to
        its planet in the scene.
        """
        if moon not in MOONS:
            raise ValueError(f"Unknown moon '{moon}'. Supported: {list(MOONS)}")
        return self.fetch(moon, force=force)

    # --- ResilientFetcher hooks -------------------------------------------------

    def _fetch_live(self, key: str) -> BodyPosition:
        from data.horizons import BODY_CODES, HorizonsClient

        now = datetime.now(timezone.utc)
        start_str = now.strftime("%Y-%m-%d %H:%M")
        end_str = (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")

        client = HorizonsClient(timeout=self.timeout)

        if key in MOONS:
            moon_meta = MOONS[key]
            parent_key = moon_meta["parent"]
            parent_code = BODY_CODES[parent_key]
            command = moon_meta["horizons_command"]

            samples = client.fetch_vectors(command, start_str, end_str, step_size="1h", center=f"500@{parent_code}")
            if not samples:
                raise RuntimeError(f"Horizons returned no samples for moon '{key}'")
            sample = samples[0]
            x, y, z = sample.r_km
            distance_km = math.sqrt(x * x + y * y + z * z)
            direction = (x / distance_km, y / distance_km, z / distance_km)

            # `command` may be a raw numeric Horizons code ("401") or a
            # mnemonic ("moon") that only fetch_vectors' own COMMAND
            # parameter auto-translates via BODY_CODES — CENTER needs
            # the resolved numeric code either way.
            center_code = BODY_CODES.get(command.lower(), command)
            sun_direction = (0.0, 0.0, 0.0)
            sun_samples = client.fetch_vectors("sun", start_str, end_str, step_size="1h", center=f"500@{center_code}")
            if sun_samples:
                sx, sy, sz = sun_samples[0].r_km
                sun_dist = math.sqrt(sx * sx + sy * sy + sz * sz)
                sun_direction = (sx / sun_dist, sy / sun_dist, sz / sun_dist)

            return BodyPosition(
                body=key, fetched_at=now.isoformat(), jd_tdb=sample.jd_tdb,
                r_km=(x, y, z), distance_km=distance_km, direction=direction,
                sun_direction=sun_direction, relative_to=parent_key,
            )

        samples = client.fetch_vectors(key, start_str, end_str, step_size="1h", center="500@399")
        if not samples:
            raise RuntimeError(f"Horizons returned no samples for '{key}'")

        sample = samples[0]
        x, y, z = sample.r_km
        distance_km = math.sqrt(x * x + y * y + z * z)
        direction = (x / distance_km, y / distance_km, z / distance_km)

        sun_direction = (0.0, 0.0, 0.0)
        if key != "sun":
            # Real direction from THIS body toward the Sun (queried
            # centered on the body itself, not on Earth) — what actually
            # determines where its terminator/day-side falls, as
            # opposed to a stylized fixed light direction.
            body_code = BODY_CODES[key]
            sun_samples = client.fetch_vectors("sun", start_str, end_str, step_size="1h", center=f"500@{body_code}")
            if sun_samples:
                sx, sy, sz = sun_samples[0].r_km
                sun_dist = math.sqrt(sx * sx + sy * sy + sz * sz)
                sun_direction = (sx / sun_dist, sy / sun_dist, sz / sun_dist)

        return BodyPosition(
            body=key,
            fetched_at=now.isoformat(),
            jd_tdb=sample.jd_tdb,
            r_km=(x, y, z),
            distance_km=distance_km,
            direction=direction,
            sun_direction=sun_direction,
            relative_to="earth",
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

    def _load_cache(self, key: str) -> BodyPosition | None:
        payload = self._read_cache_file(key)
        if not payload or "position" not in payload:
            return None
        rec = payload["position"]
        return BodyPosition(
            body=rec["body"], fetched_at=rec["fetched_at"], jd_tdb=rec["jd_tdb"],
            r_km=tuple(rec["r_km"]), distance_km=rec["distance_km"], direction=tuple(rec["direction"]),
            sun_direction=tuple(rec.get("sun_direction", (0.0, 0.0, 0.0))),
            relative_to=rec.get("relative_to", "earth"),
        )

    def _save_cache(self, key: str, data: BodyPosition) -> None:
        self._cache_path(key).write_text(json.dumps({
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "position": asdict(data),
        }))

    def _store_fallback(self, key: str, data: BodyPosition) -> None:
        self._save_cache(key, data)

    def _load_seed(self, key: str) -> BodyPosition | None:
        return None  # no bundled last-known-good ephemeris is checked into the repo

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
