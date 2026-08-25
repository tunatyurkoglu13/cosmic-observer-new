"""
viz/orbit_trails.py — Orbit path and ground-track sampling for the dashboard.

Two distinct products, both consumed by the 3D viewer:

  1. Orbit path (3D): the satellite's trajectory over one orbital period,
     in ECI space — drawn as a closed ring around the globe (a "trail").
  2. Ground track (2D): the corresponding sequence of (lat, lon) points,
     which is what you'd plot on a flat map — but which needs
     antimeridian-crossing handling, since a ground track that crosses
     +/-180 deg longitude must NOT be connected by a straight line across
     the whole map (it should wrap around instead). We split the track
     into segments at each crossing so the renderer draws separate
     polylines rather than one line slicing across the globe.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from core.kepler import orbital_period
from core.propagator import Propagator


def compute_orbit_path_teme(
    propagator: Propagator,
    epoch: datetime,
    semi_major_axis_km: float,
    num_points: int = 200,
) -> list[tuple[float, float, float]]:
    """
    Sample one full orbital period of the satellite's TEME position,
    starting at `epoch`, for drawing a 3D orbit trail.

    Args:
        propagator: Propagator wrapping the satellite's TLE.
        epoch: UTC datetime to start sampling from.
        semi_major_axis_km: used only to compute the orbital period via
            orbital_period() (Kepler's third law) — the actual positions
            still come from SGP4, not a Keplerian approximation.
        num_points: number of samples over the period.

    Returns:
        List of (x, y, z) TEME positions [km], num_points long, forming a
        closed loop (first and last points are one step apart, not
        duplicated).
    """
    period_s = orbital_period(semi_major_axis_km)
    step_s = period_s / num_points

    points = []
    for i in range(num_points):
        t = epoch + timedelta(seconds=i * step_s)
        try:
            state = propagator.propagate(t)
            points.append(tuple(state.r_teme))
        except RuntimeError:
            continue
    return points


def compute_ground_track(
    propagator: Propagator,
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
) -> list[list[tuple[float, float]]]:
    """
    Sample (lat, lon) ground-track points over a duration, split into
    segments at antimeridian crossings so each segment can be drawn as an
    independent polyline (avoiding a spurious line straight across the map).

    Args:
        propagator: Propagator wrapping the satellite's TLE.
        epoch: UTC start time.
        duration: total time span to sample.
        step: sampling interval.

    Returns:
        List of segments, each a list of (lat_deg, lon_deg) tuples.
    """
    n_samples = max(2, int(duration / step) + 1)

    segments: list[list[tuple[float, float]]] = [[]]
    prev_lon = None

    for i in range(n_samples):
        t = epoch + i * step
        try:
            state = propagator.propagate(t)
        except RuntimeError:
            continue

        lat, lon = state.lat_deg, state.lon_deg

        if prev_lon is not None and abs(lon - prev_lon) > 180.0:
            segments.append([])

        segments[-1].append((lat, lon))
        prev_lon = lon

    return [seg for seg in segments if seg]
