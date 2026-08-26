"""
core/space_track_client.py — Space-Track.org authenticated TLE/GP client.

Space-Track.org (run by USSPACECOM) is the authoritative source CelesTrak
itself republishes from, and offers data CelesTrak's public mirror
doesn't (full catalog history, decay/reentry predictions, conjunction
data messages). Unlike CelesTrak, it requires a free registered account
and a session-cookie login flow rather than a bare keyless GET:

    1. POST username/password to /ajaxauth/login -> the response sets a
       session cookie on the client.
    2. Every subsequent request reuses that cookie (same requests.Session)
       until it expires (Space-Track sessions last ~2 hours of inactivity).

Fair-use rate limit (per Space-Track's own usage guidelines): no more
than ~1 request per 2 seconds, and no more than ~200-300 requests per
hour sustained. `_wait_for_rate_limit()` enforces the per-request spacing
side of that; this project's own request volume is far below the hourly
cap so no separate hourly counter is implemented.

Registration is required and is the user's own action (this project
cannot create the account) — see https://www.space-track.org. Until
SPACE_TRACK_USERNAME/SPACE_TRACK_PASSWORD are set (in .env or the
environment), `is_configured` is False and callers are expected to fall
back to the existing keyless CelesTrak path (see
core.tle_manager.TLEManager's optional `space_track_client` argument) —
this client never raises just because it's unconfigured; `is_configured`
is there precisely so callers can check first.

Group semantics note: Space-Track's `gp` (general perturbations) class
has no equivalent of CelesTrak's curated GROUP= parameter (CelesTrak's
"stations"/"visual"/"starlink" groups are its own hand-maintained
lists, not a Space-Track field) — the closest available approximation is
an OBJECT_NAME substring match, which is what `GROUP_NAME_PATTERNS`
below encodes for the handful of groups this project uses. It is a
best-effort approximation, not an identical dataset; groups without a
sensible name pattern (e.g. "active", the entire active catalog) are
left unmapped and simply always fall back to CelesTrak.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

BASE_URL = "https://www.space-track.org/"
LOGIN_PATH = "ajaxauth/login"

# Best-effort OBJECT_NAME substring approximations of CelesTrak's curated
# groups (see module docstring) — only groups with a reasonably unambiguous
# name pattern are included.
GROUP_NAME_PATTERNS: dict[str, str] = {
    "stations": "ISS,CSS,TIANGONG",
    "starlink": "STARLINK",
    "gps-ops": "NAVSTAR",
}

_PLACEHOLDER_VALUES = {"", "your_username", "your_password"}


class SpaceTrackClient:
    """Authenticated Space-Track.org client with a fair-use rate limiter."""

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        timeout: int = 30,
        min_request_interval: float = 2.0,
    ):
        self.username = username or os.environ.get("SPACE_TRACK_USERNAME", "")
        self.password = password or os.environ.get("SPACE_TRACK_PASSWORD", "")
        self.timeout = timeout
        self.min_request_interval = min_request_interval

        self._session = requests.Session()
        self._logged_in = False
        self._last_request_time = 0.0

    @property
    def is_configured(self) -> bool:
        """True if real (non-placeholder) credentials are available."""
        return (
            self.username.strip() not in _PLACEHOLDER_VALUES
            and self.password.strip() not in _PLACEHOLDER_VALUES
        )

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        self._wait_for_rate_limit()
        resp = self._session.request(method, BASE_URL + path, timeout=self.timeout, **kwargs)
        self._last_request_time = time.monotonic()
        return resp

    def login(self) -> None:
        """
        Authenticate and store the resulting session cookie for
        subsequent requests.

        Raises:
            RuntimeError: if not configured, or Space-Track rejects the
                credentials.
        """
        if not self.is_configured:
            raise RuntimeError(
                "Space-Track credentials not configured (SPACE_TRACK_USERNAME/"
                "SPACE_TRACK_PASSWORD). Register a free account at "
                "https://www.space-track.org and set them in .env."
            )

        resp = self._request(
            "POST", LOGIN_PATH, data={"identity": self.username, "password": self.password}
        )
        resp.raise_for_status()
        # A successful login returns an empty body; a rejected one returns
        # a JSON error payload (e.g. {"Login": "Failed"}).
        if resp.text.strip():
            raise RuntimeError(f"Space-Track login rejected: {resp.text.strip()}")

        self._logged_in = True

    def _ensure_logged_in(self) -> None:
        if not self._logged_in:
            self.login()

    def fetch_gp(
        self,
        norad_ids: list[int] | None = None,
        object_name_contains: str | None = None,
        format: str = "tle",
    ) -> str:
        """
        Query the `gp` (general perturbations, i.e. current TLE-equivalent
        elements) class, returning the raw response body in the requested
        format (default "tle", matching CelesTrak's own default so
        callers can reuse the same TLE-text parser either way).

        Args:
            norad_ids: filter to specific NORAD catalog IDs.
            object_name_contains: filter via a "~~PATTERN" OBJECT_NAME
                predicate (Space-Track's substring-match operator).
            format: one of Space-Track's supported response formats
                ("tle", "3le", "json", "xml", "csv", "html", "kvn").
        """
        self._ensure_logged_in()

        predicates = []
        if norad_ids:
            predicates.append(f"NORAD_CAT_ID/{','.join(str(n) for n in norad_ids)}")
        if object_name_contains:
            predicates.append(f"OBJECT_NAME/~~{object_name_contains}")
        predicates.append(f"format/{format}")

        path = "basicspacedata/query/class/gp/" + "/".join(predicates)
        resp = self._request("GET", path)
        resp.raise_for_status()
        return resp.text

    def fetch_group_tle(self, group: str) -> str:
        """
        Best-effort analog of CelesTrak's GROUP= query (see module
        docstring's group-semantics note): fetch raw TLE text for the
        OBJECT_NAME pattern this project maps the group to.

        Raises:
            ValueError: if `group` has no known name-pattern mapping —
                callers should catch this and fall back to CelesTrak,
                same as any other failure.
        """
        pattern = GROUP_NAME_PATTERNS.get(group)
        if pattern is None:
            raise ValueError(f"No Space-Track OBJECT_NAME mapping for group '{group}'; use CelesTrak for this group.")
        return self.fetch_gp(object_name_contains=pattern, format="tle")
