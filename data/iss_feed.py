"""
data/iss_feed.py — ISS live video feed metadata and current-position lookup.

This module does not implement video decoding/streaming itself (that
belongs to the CV pipeline in cv/ once we get there, e.g. an OpenCV
VideoCapture reading an HLS URL). It provides:

  1. Known-good ISS live video stream endpoints, for the dashboard to
     embed or for cv/preprocess.py to eventually pull frames from.
  2. A convenience wrapper around core.tle_manager + core.propagator to
     answer "where is the ISS right now / at time t" — useful context to
     display alongside the video feed (e.g. "currently over the Pacific").

Stream sources (per project spec):
  - NASA HDEV / ISS live views: https://eol.jsc.nasa.gov/ESRS/HDEV/
  - NASA live channel: https://www.nasa.gov/live/
  - Backup: NASA's official YouTube live channel

Note: NASA has changed its official ISS livestream hosting several times
over the years (HDEV hardware itself was retired in 2019; NASA now
generally points viewers at nasa.gov/live or the NASA YouTube channel for
the current external-camera feed). Treat these as the best-known public
entry points rather than a single stable API contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from core.propagator import Propagator, StateVector
from core.tle_manager import Satellite, TLEManager

ISS_NORAD_ID = 25544

STREAM_SOURCES = {
    "nasa_live": "https://www.nasa.gov/live/",
    "hdev_archive_info": "https://eol.jsc.nasa.gov/ESRS/HDEV/",
    "nasa_youtube": "https://www.youtube.com/@NASA/live",
}


@dataclass
class ISSStatus:
    """Current ISS state, paired with stream metadata for dashboard display."""

    state: StateVector
    satellite: Satellite
    stream_sources: dict[str, str]

    @property
    def summary(self) -> str:
        return (
            f"ISS (NORAD {self.satellite.norad_id}) at "
            f"lat={self.state.lat_deg:.2f}, lon={self.state.lon_deg:.2f}, "
            f"alt={self.state.alt_km:.1f} km"
        )


def _find_iss(satellites: list[Satellite]) -> Satellite:
    for sat in satellites:
        if sat.norad_id == ISS_NORAD_ID:
            return sat
    raise LookupError(f"ISS (NORAD {ISS_NORAD_ID}) not found in fetched 'stations' group")


def get_iss_status(tle_manager: TLEManager | None = None, at: datetime | None = None) -> ISSStatus:
    """
    Fetch the ISS's current (or specified-time) position alongside known
    live-video stream sources.

    Args:
        tle_manager: an existing TLEManager to reuse its cache; a fresh
            one (default DB path/staleness) is created if omitted.
        at: UTC datetime to evaluate position at; defaults to now.

    Returns:
        ISSStatus with propagated state, satellite metadata, and stream URLs.
    """
    mgr = tle_manager or TLEManager()
    satellites = mgr.fetch_group("stations")
    iss = _find_iss(satellites)

    when = at or datetime.now(timezone.utc)
    prop = Propagator(iss.line1, iss.line2, name=iss.name)
    state = prop.propagate(when)

    return ISSStatus(state=state, satellite=iss, stream_sources=dict(STREAM_SOURCES))
