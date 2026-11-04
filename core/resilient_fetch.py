"""
core/resilient_fetch.py — Generic network -> cache -> seed fallback chain.

This factors out the resilience pattern that grew organically inside
core.tle_manager.TLEManager while building this project: CelesTrak (and,
in general, any free public space-data API) is occasionally slow or
unreachable, so every client that depends on one needs the same three
answers to "the network call just failed, now what":

    1. Serve whatever we already have cached, however stale, rather
       than hard-fail an interactive tool over a transient outage.
    2. If there's no cache at all (e.g. a fresh clone with no prior
       fetches), fall back to a small bundled "seed" dataset checked
       into the repo, so the app still shows something real on first
       run with no network.
    3. Don't hammer a host that just failed — after a failure, skip
       re-attempting the network for a short cooldown window and go
       straight to whatever fallback is available.

`TLEManager` was the first real, working implementation of this control
flow (see its own history/tests). `ResilientFetcher` pulls just the
CONTROL FLOW out into a reusable base class; subclasses supply the
storage-specific hooks (SQLite for TLEManager's structured satellite
records, a flat JSON file for data.nasa_cneos's simpler risk list) —
whatever storage suits their data. This class doesn't know or care how
caching/seeding are actually implemented.

Only `requests.RequestException` (network/transport failures) trigger
the fallback chain — a successful response that the caller judges to be
bad data (e.g. an API's "invalid query" error text) is a different kind
of failure and should propagate directly rather than being treated as
"the network is down, serve stale data instead."
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generic, TypeVar

import requests

T = TypeVar("T")


class ResilientFetcher(ABC, Generic[T]):
    """
    Base class for "fetch a keyed resource with cache/seed fallback."

    Subclasses implement the storage-specific hooks below; `fetch()`
    owns the control flow and should not need to be overridden.
    """

    staleness: timedelta = timedelta(hours=6)
    # How long to skip re-attempting a key's network fetch after it just
    # failed, before trying again. Short on purpose: long enough that a
    # sustained outage doesn't force every single fetch() call to eat a
    # slow connect-timeout, short enough that a real recovery is noticed
    # quickly rather than waiting out the full `staleness` window.
    failure_retry_cooldown: timedelta = timedelta(minutes=2)

    @abstractmethod
    def _fetch_live(self, key: str) -> T:
        """Perform the actual network call and return parsed data. Raise on failure."""

    @abstractmethod
    def _load_cache(self, key: str) -> T | None:
        """Return cached data for key, or None/empty if none exists."""

    @abstractmethod
    def _save_cache(self, key: str, data: T) -> None:
        """Persist freshly live-fetched data for key and mark it fresh (resets staleness)."""

    @abstractmethod
    def _store_fallback(self, key: str, data: T) -> None:
        """
        Persist seed-fallback data for key WITHOUT marking it fresh — a
        subsequent call outside the failure cooldown should still retry
        the network rather than treating seed data as up to date.
        """

    @abstractmethod
    def _load_seed(self, key: str) -> T | None:
        """Return bundled last-known-good fallback data for key, or None/empty if none exists."""

    @abstractmethod
    def _is_stale(self, key: str) -> bool:
        """Whether the cached data for key is older than `staleness` (or doesn't exist)."""

    @abstractmethod
    def _recently_failed(self, key: str) -> bool:
        """Whether a fetch for key failed within `failure_retry_cooldown`."""

    @abstractmethod
    def _record_failure(self, key: str) -> None: ...

    @abstractmethod
    def _clear_failure(self, key: str) -> None: ...

    def fetch(self, key: str, force: bool = False, allow_fallback: bool = True) -> T:
        """
        Fetch `key`, preferring fresh cache, then network, then stale
        cache/seed data on network failure.

        Args:
            force: skip both the freshness check and the failure
                cooldown and always attempt the network.
            allow_fallback: if the network call fails, fall back to
                stale cache then bundled seed data instead of raising.
                Set False if the caller specifically needs to know the
                fetch failed rather than get quietly degraded data.
        """
        if not force and not self._is_stale(key):
            return self._load_cache(key)

        if not force and self._recently_failed(key):
            cached = self._load_cache(key)
            if cached:
                return cached

        try:
            data = self._fetch_live(key)
        except requests.RequestException:
            self._record_failure(key)
            if allow_fallback:
                cached = self._load_cache(key)
                if cached:
                    return cached
                seeded = self._load_seed(key)
                if seeded:
                    self._store_fallback(key, seeded)
                    return seeded
            raise

        self._clear_failure(key)
        self._save_cache(key, data)
        return data


class JsonFileFetcher(ResilientFetcher[T]):
    """
    ResilientFetcher with the storage hooks a plain JSON API client needs:
    one JSON file per client holding, per key, the last live payload and
    its cached_at / failed_at stamps. Subclasses implement only
    `_fetch_live`, plus `_encode`/`_decode` if T isn't JSON-native.
    """

    def __init__(
        self,
        cache_path: Path | str,
        staleness: timedelta,
        failure_retry_cooldown: timedelta = timedelta(minutes=2),
    ):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.staleness = staleness
        self.failure_retry_cooldown = failure_retry_cooldown

    def _encode(self, data: T) -> object:
        return data

    def _decode(self, raw: object) -> T:
        return raw  # type: ignore[return-value]

    # --- file helpers ------------------------------------------------------

    def _read_all(self) -> dict:
        try:
            return json.loads(self.cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _update_entry(self, key: str, **changes) -> None:
        payload = self._read_all()
        entry = payload.setdefault(key, {})
        for name, value in changes.items():
            if value is None:
                entry.pop(name, None)
            else:
                entry[name] = value
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self.cache_path)  # atomic: a crash mid-write never leaves a torn cache

    def _stamp(self, key: str, name: str) -> datetime | None:
        raw = self._read_all().get(key, {}).get(name)
        return datetime.fromisoformat(raw) if raw else None

    # --- ResilientFetcher hooks -------------------------------------------

    def _load_cache(self, key: str) -> T | None:
        entry = self._read_all().get(key)
        return self._decode(entry["data"]) if entry and "data" in entry else None

    def _save_cache(self, key: str, data: T) -> None:
        self._update_entry(key, data=self._encode(data), cached_at=datetime.now(timezone.utc).isoformat())

    def _store_fallback(self, key: str, data: T) -> None:
        self._update_entry(key, data=self._encode(data))

    def _load_seed(self, key: str) -> T | None:
        return None

    def _is_stale(self, key: str) -> bool:
        cached_at = self._stamp(key, "cached_at")
        return cached_at is None or datetime.now(timezone.utc) - cached_at > self.staleness

    def _recently_failed(self, key: str) -> bool:
        failed_at = self._stamp(key, "failed_at")
        return failed_at is not None and datetime.now(timezone.utc) - failed_at <= self.failure_retry_cooldown

    def _record_failure(self, key: str) -> None:
        self._update_entry(key, failed_at=datetime.now(timezone.utc).isoformat())

    def _clear_failure(self, key: str) -> None:
        self._update_entry(key, failed_at=None)
