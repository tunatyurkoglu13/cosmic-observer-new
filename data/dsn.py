"""
data/dsn.py — NASA/JPL Deep Space Network real-time status ("DSN Now").

The DSN is the global antenna network (Goldstone/California, Madrid/Spain,
Canberra/Australia) that talks to every NASA (and many partner) deep-space
mission — Voyager, JWST, Mars rovers, New Horizons, everything beyond low
Earth orbit. JPL publishes its live per-dish status (which spacecraft it's
tracking, uplink/downlink signal state, data rate, target range) as a
small XML feed that updates roughly every few seconds — the same feed
that powers the public "DSN Now" visualization at eyes.nasa.gov/dsn.

Endpoint: https://eyes.nasa.gov/dsn/data/dsn.xml
Keyless, no official docs page, but this is the real, live, production
feed the public DSN Now site itself uses — confirmed by fetching it
directly and cross-checking reported target distances (e.g. Mars rover
range) against known real Earth-Mars distance during development.

XML shape: a flat sequence of <station> elements (Goldstone/Madrid/
Canberra), each followed by that station's <dish> elements until the
next <station> — dishes are NOT nested inside <station> in the XML, so
parsing tracks "current station" while iterating siblings in document
order. Each <dish> may have zero, one, or several <upSignal>/<downSignal>
children (a dish can track more than one spacecraft signal at once) and
exactly one <target> (name/id/range — range is in km; -1 means unknown/
not applicable, e.g. while idle or in maintenance).

Not every mission is in contact at any given moment — dishes rotate
between scheduled passes, so a spacecraft (e.g. a Voyager) simply not
appearing right now is normal, real behavior, not a fetch failure.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from core.resilient_fetch import ResilientFetcher

DSN_FEED_URL = "https://eyes.nasa.gov/dsn/data/dsn.xml"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "dsn_status.json"


@dataclass
class DSNSignal:
    direction: str          # "up" (Earth -> spacecraft) or "down" (spacecraft -> Earth)
    active: bool
    data_rate_bps: float
    band: str                # radio band, e.g. "X", "S", "K", "Ka"
    power: float              # dBm (downlink) or kW (uplink), per DSN Now's own convention
    spacecraft: str


@dataclass
class DSNDish:
    name: str                 # e.g. "DSS14"
    station: str               # friendly station name, e.g. "Goldstone"
    azimuth_deg: float
    elevation_deg: float
    activity: str               # human-readable current activity, e.g. "Spacecraft Telemetry, Tracking, and Command"
    target_name: str
    uplink_range_km: float | None
    downlink_range_km: float | None
    signals: list[DSNSignal] = field(default_factory=list)


@dataclass
class DSNStatus:
    fetched_at: str
    station_names: list[str]
    dishes: list[DSNDish]

    @property
    def active_spacecraft(self) -> list[str]:
        """Real spacecraft currently being tracked by at least one dish (excludes idle/maintenance targets like 'DSN'/'DSS')."""
        names = {d.target_name for d in self.dishes if d.target_name not in ("DSN", "DSS", "")}
        return sorted(names)


def _parse_dsn_xml(raw_xml: str) -> DSNStatus:
    root = ET.fromstring(raw_xml)
    station_names: list[str] = []
    dishes: list[DSNDish] = []
    current_station = "Unknown"

    for el in root:
        if el.tag == "station":
            current_station = el.get("friendlyName", el.get("name", "Unknown"))
            station_names.append(current_station)
        elif el.tag == "dish":
            target_el = el.find("target")
            target_name = target_el.get("name", "") if target_el is not None else ""

            def _range(attr: str) -> float | None:
                if target_el is None:
                    return None
                raw = target_el.get(attr)
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    return None
                return val if val >= 0 else None

            signals = []
            for sig_tag, direction in (("upSignal", "up"), ("downSignal", "down")):
                for sig_el in el.findall(sig_tag):
                    signals.append(DSNSignal(
                        direction=direction,
                        active=sig_el.get("active") == "true",
                        data_rate_bps=float(sig_el.get("dataRate", 0) or 0),
                        band=sig_el.get("band", ""),
                        power=float(sig_el.get("power", 0) or 0),
                        spacecraft=sig_el.get("spacecraft", ""),
                    ))

            def _float_attr(attr: str) -> float:
                raw = el.get(attr, "")
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return 0.0

            dishes.append(DSNDish(
                name=el.get("name", ""),
                station=current_station,
                azimuth_deg=_float_attr("azimuthAngle"),
                elevation_deg=_float_attr("elevationAngle"),
                activity=el.get("activity", ""),
                target_name=target_name,
                uplink_range_km=_range("uplegRange"),
                downlink_range_km=_range("downlegRange"),
                signals=signals,
            ))

    return DSNStatus(
        fetched_at=datetime.now(timezone.utc).isoformat(),
        station_names=station_names,
        dishes=dishes,
    )


class DSNClient(ResilientFetcher[DSNStatus]):
    """Resilient-cached client for the live DSN Now feed."""

    STATUS_KEY = "status"

    def __init__(
        self,
        cache_path: Path | str = DEFAULT_CACHE_PATH,
        timeout: int = 15,
        # Short staleness — this is a genuinely fast-changing live feed —
        # but still cached so a transient outage degrades to "slightly
        # stale real data" instead of a hard failure, same pattern as
        # every other resilient client in this project.
        staleness: timedelta = timedelta(seconds=20),
        failure_retry_cooldown: timedelta = timedelta(minutes=1),
    ):
        self.timeout = timeout
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.staleness = staleness
        self.failure_retry_cooldown = failure_retry_cooldown

    def fetch_status(self, force: bool = False) -> DSNStatus:
        return self.fetch(self.STATUS_KEY, force=force)

    # --- ResilientFetcher hooks -------------------------------------------------

    def _fetch_live(self, key: str) -> DSNStatus:
        resp = requests.get(DSN_FEED_URL, timeout=self.timeout)
        resp.raise_for_status()
        return _parse_dsn_xml(resp.text)

    def _read_cache_file(self) -> dict | None:
        if not self.cache_path.exists():
            return None
        try:
            return json.loads(self.cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _load_cache(self, key: str) -> DSNStatus | None:
        payload = self._read_cache_file()
        if not payload or "status" not in payload:
            return None
        rec = payload["status"]
        dishes = [
            DSNDish(
                name=d["name"], station=d["station"], azimuth_deg=d["azimuth_deg"],
                elevation_deg=d["elevation_deg"], activity=d["activity"], target_name=d["target_name"],
                uplink_range_km=d["uplink_range_km"], downlink_range_km=d["downlink_range_km"],
                signals=[DSNSignal(**s) for s in d["signals"]],
            )
            for d in rec["dishes"]
        ]
        return DSNStatus(fetched_at=rec["fetched_at"], station_names=rec["station_names"], dishes=dishes)

    def _save_cache(self, key: str, data: DSNStatus) -> None:
        self.cache_path.write_text(json.dumps({
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "status": asdict(data),
        }))

    def _store_fallback(self, key: str, data: DSNStatus) -> None:
        self._save_cache(key, data)

    def _load_seed(self, key: str) -> DSNStatus | None:
        return None  # no bundled seed — a stale/cold cache simply raises, same as data.nasa_cneos

    def _is_stale(self, key: str) -> bool:
        payload = self._read_cache_file()
        if not payload or "cached_at" not in payload:
            return True
        cached_at = datetime.fromisoformat(payload["cached_at"])
        return datetime.now(timezone.utc) - cached_at > self.staleness

    def _recently_failed(self, key: str) -> bool:
        payload = self._read_cache_file()
        if not payload or "failed_at" not in payload:
            return False
        failed_at = datetime.fromisoformat(payload["failed_at"])
        return datetime.now(timezone.utc) - failed_at <= self.failure_retry_cooldown

    def _record_failure(self, key: str) -> None:
        payload = self._read_cache_file() or {}
        payload["failed_at"] = datetime.now(timezone.utc).isoformat()
        self.cache_path.write_text(json.dumps(payload))

    def _clear_failure(self, key: str) -> None:
        payload = self._read_cache_file()
        if payload and "failed_at" in payload:
            del payload["failed_at"]
            self.cache_path.write_text(json.dumps(payload))
