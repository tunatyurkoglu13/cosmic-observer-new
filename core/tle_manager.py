"""
core/tle_manager.py — Fetch, parse, and cache Two-Line Element sets.

Data source priority (per project convention): CelesTrak first (free, no
auth, updated multiple times daily) with Space-Track as an optional
authenticated fallback for users who configure credentials.

TLEs are cached in a local SQLite database keyed by NORAD catalog ID so the
rest of the platform can work offline between refreshes, and so we only
re-parse/re-fetch data older than a configurable staleness window.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from core.constants import MU_EARTH
from core.kepler import semi_major_axis_from_mean_motion

CELESTRAK_BASE = "https://celestrak.org/NORAD/elements/gp.php"

# GROUP names CelesTrak recognizes; see https://celestrak.org/NORAD/elements/
GROUPS = {
    "active": "active",
    # CelesTrak retired the generic GROUP=debris query; debris is now only
    # available as named fragmentation-event clouds. cosmos-2251-debris
    # (the 2009 Iridium-33/Cosmos-2251 collision) is the largest tracked
    # cloud and stands in as our default "debris" sample.
    "debris": "cosmos-2251-debris",
    "stations": "stations",
    "visual": "visual",
    "starlink": "starlink",
    "gps-ops": "gps-ops",
}

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cache" / "tle_cache.sqlite3"


@dataclass
class Satellite:
    """A single catalog object: identity + its most recent TLE."""

    norad_id: int
    name: str
    line1: str
    line2: str
    epoch: datetime          # TLE epoch (UTC)
    inclination_deg: float
    eccentricity: float
    mean_motion_rev_per_day: float
    semi_major_axis_km: float
    classification: str = "active"  # "active" | "debris" | "station"

    @classmethod
    def from_tle(cls, name: str, line1: str, line2: str, classification: str = "active") -> "Satellite":
        """Parse the fields we need directly out of the raw TLE lines.

        TLE line1/line2 column layout: NORAD Spacetrack Report #3.
        """
        norad_id = int(line1[2:7])

        epoch_year = int(line1[18:20])
        epoch_year += 2000 if epoch_year < 57 else 1900
        epoch_day = float(line1[20:32])
        epoch = datetime(epoch_year, 1, 1, tzinfo=timezone.utc) + timedelta(days=epoch_day - 1)

        inclination_deg = float(line2[8:16])
        eccentricity = float("0." + line2[26:33].strip())
        mean_motion = float(line2[52:63])  # rev/day

        n_rad_s = mean_motion * 2 * 3.14159265358979 / 86400.0
        a_km = semi_major_axis_from_mean_motion(n_rad_s, mu=MU_EARTH)

        return cls(
            norad_id=norad_id,
            name=name.strip(),
            line1=line1.strip(),
            line2=line2.strip(),
            epoch=epoch,
            inclination_deg=inclination_deg,
            eccentricity=eccentricity,
            mean_motion_rev_per_day=mean_motion,
            semi_major_axis_km=a_km,
            classification=classification,
        )


class TLEManager:
    """Fetches TLE groups from CelesTrak, parses them, and caches to SQLite."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH, staleness: timedelta = timedelta(hours=6)):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.staleness = staleness
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS satellites (
                    norad_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    line1 TEXT NOT NULL,
                    line2 TEXT NOT NULL,
                    epoch TEXT NOT NULL,
                    inclination_deg REAL,
                    eccentricity REAL,
                    mean_motion REAL,
                    semi_major_axis_km REAL,
                    classification TEXT,
                    fetched_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fetch_log (
                    group_name TEXT PRIMARY KEY,
                    fetched_at TEXT NOT NULL
                )
                """
            )
            # A satellite can legitimately belong to more than one
            # CelesTrak group at once (e.g. the ISS is both "stations"
            # and "visual" — one of the brightest visible objects).
            # `satellites.classification` only ever records the group it
            # was *first* seen under (see _upsert), so group membership
            # for load_cached()'s filtering is tracked separately here,
            # rather than being clobbered by whichever group happens to
            # be fetched most recently.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS satellite_groups (
                    norad_id INTEGER NOT NULL,
                    group_name TEXT NOT NULL,
                    PRIMARY KEY (norad_id, group_name)
                )
                """
            )

    def _group_is_stale(self, group_name: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT fetched_at FROM fetch_log WHERE group_name = ?", (group_name,)
            ).fetchone()
        if row is None:
            return True
        fetched_at = datetime.fromisoformat(row[0])
        return datetime.now(timezone.utc) - fetched_at > self.staleness

    def fetch_group(self, group: str, force: bool = False) -> list[Satellite]:
        """
        Fetch one CelesTrak group ("active", "debris", "stations"), parse
        every TLE in the response, and upsert into the cache.

        If the cache for this group was refreshed within `self.staleness`
        and force=False, skip the network call and return cached rows.
        """
        if group not in GROUPS:
            raise ValueError(f"Unknown TLE group '{group}'. Known groups: {list(GROUPS)}")

        if not force and not self._group_is_stale(group):
            return self.load_cached(classification=group)

        url = f"{CELESTRAK_BASE}?GROUP={GROUPS[group]}&FORMAT=tle"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        if resp.text.lstrip().startswith("Invalid query") or resp.text.lstrip().startswith("No GP data found"):
            raise RuntimeError(f"CelesTrak rejected group '{group}' (GROUP={GROUPS[group]}): {resp.text.strip()}")

        satellites = self._parse_tle_text(resp.text, classification=group)
        self._upsert(satellites)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO fetch_log (group_name, fetched_at) VALUES (?, ?) "
                "ON CONFLICT(group_name) DO UPDATE SET fetched_at = excluded.fetched_at",
                (group, datetime.now(timezone.utc).isoformat()),
            )

        return satellites

    @staticmethod
    def _parse_tle_text(text: str, classification: str) -> list[Satellite]:
        """Parse a CelesTrak-format TLE text block (3 lines per object) into Satellite objects."""
        lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
        satellites = []
        for i in range(0, len(lines) - 2, 3):
            name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
            if not (line1.startswith("1 ") and line2.startswith("2 ")):
                continue
            try:
                satellites.append(Satellite.from_tle(name, line1, line2, classification=classification))
            except (ValueError, IndexError):
                # Malformed record in the feed — skip rather than abort the whole batch.
                continue
        return satellites

    def _upsert(self, satellites: list[Satellite]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO satellites
                    (norad_id, name, line1, line2, epoch, inclination_deg,
                     eccentricity, mean_motion, semi_major_axis_km, classification, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(norad_id) DO UPDATE SET
                    name=excluded.name, line1=excluded.line1, line2=excluded.line2,
                    epoch=excluded.epoch, inclination_deg=excluded.inclination_deg,
                    eccentricity=excluded.eccentricity, mean_motion=excluded.mean_motion,
                    semi_major_axis_km=excluded.semi_major_axis_km,
                    fetched_at=excluded.fetched_at
                """,
                # Note: classification is intentionally NOT in the UPDATE SET
                # above, so a satellite's first-seen group "wins" and later
                # fetches under a different group (e.g. ISS also appearing
                # in "visual") don't overwrite it. Full multi-group
                # membership is tracked in satellite_groups below instead.
                [
                    (
                        s.norad_id, s.name, s.line1, s.line2, s.epoch.isoformat(),
                        s.inclination_deg, s.eccentricity, s.mean_motion_rev_per_day,
                        s.semi_major_axis_km, s.classification, now,
                    )
                    for s in satellites
                ],
            )
            conn.executemany(
                "INSERT OR IGNORE INTO satellite_groups (norad_id, group_name) VALUES (?, ?)",
                [(s.norad_id, s.classification) for s in satellites],
            )

    def load_cached(self, classification: str | None = None) -> list[Satellite]:
        """
        Load satellites from the local cache without touching the network.

        Filtering by `classification` matches on *group membership*
        (satellite_groups), not the satellites table's own classification
        column — a satellite can belong to multiple CelesTrak groups
        (e.g. the ISS is both "stations" and "visual"), and this ensures
        it's returned for every group it's actually in rather than only
        whichever group happened to be recorded first.
        """
        if classification is None:
            query = (
                "SELECT norad_id, name, line1, line2, epoch, inclination_deg, "
                "eccentricity, mean_motion, semi_major_axis_km, classification FROM satellites"
            )
            params: tuple = ()
        else:
            query = (
                "SELECT s.norad_id, s.name, s.line1, s.line2, s.epoch, s.inclination_deg, "
                "s.eccentricity, s.mean_motion, s.semi_major_axis_km, ? "
                "FROM satellites s "
                "JOIN satellite_groups sg ON sg.norad_id = s.norad_id "
                "WHERE sg.group_name = ?"
            )
            params = (classification, classification)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            Satellite(
                norad_id=r[0], name=r[1], line1=r[2], line2=r[3],
                epoch=datetime.fromisoformat(r[4]),
                inclination_deg=r[5], eccentricity=r[6],
                mean_motion_rev_per_day=r[7], semi_major_axis_km=r[8],
                classification=r[9],
            )
            for r in rows
        ]

    def recent_only(self, satellites: list[Satellite], max_age: timedelta = timedelta(days=30)) -> list[Satellite]:
        """Filter to TLEs whose epoch is within `max_age` of now — propagation error grows with staleness."""
        cutoff = datetime.now(timezone.utc) - max_age
        return [s for s in satellites if s.epoch >= cutoff]
