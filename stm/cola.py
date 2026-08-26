"""
stm/cola.py — Collision On Launch Assessment (COLA).

This is the feature real launch providers run before every rocket
launch: take the vehicle's planned ascent trajectory, wrap it in a
"safety bubble" that moves with it, and check whether any tracked
catalog object (active satellite, station, debris) will be inside that
bubble at the same moment the rocket is expected to be there. If nothing
gets that close throughout the whole ascent, the launch time is CLEAR;
if anything does, it's BLOCKED, and the "launch window" is really just
the search for a stretch of clear launch times, one minute apart, over a
search range (per project spec: "USSPACECOM tracks the catalog, computes
where everything will be hours/days ahead, and a fixed reference
trajectory is checked against that predicted traffic").

Two-stage design for performance (screening ~hundreds of catalog objects
against ~hundreds of candidate launch times, each with dozens of ascent
samples, would be extremely slow done naively with one SGP4 call per
object per sample):

  1. Precompute every catalog object's TEME position on ONE fine, shared
     time grid covering [search_start, search_end + ascent_duration],
     using Propagator.propagate_teme_array() (vectorized — one call per
     object across the whole grid, not one call per (object, time) pair).
  2. For each candidate launch time, build its ascent trajectory (Phase 4
     rocket_trajectory model) and, for each ascent sample, look up the
     *nearest* precomputed grid time for every catalog object and compute
     distances — this reuses the shared grid rather than re-propagating
     anything.

The "nearest grid sample" approximation is standard for this kind of
interactive screening tool: catalog objects move smoothly at orbital
velocity, so as long as the grid step is short relative to how fast
miss-distance can change (a few seconds to a few tens of seconds), the
nearest-sample distance is a good stand-in for the exact one. This is not
meant to replace a real COLA system's exact per-instant propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from core.launch_window import LaunchSite, launch_azimuth
from core.propagator import Propagator, jd_fr_array_from_datetimes
from core.rocket_trajectory import TrajectorySample, compute_ascent_trajectory
from core.tle_manager import Satellite

# Default "safety bubble" radius around the rocket at each instant, per
# the project's real-world reference point (a moving exclusion sphere,
# simpler than modeling a fixed cylindrical corridor in an inertial frame
# where the corridor itself would also need to rotate with Earth).
DEFAULT_BUBBLE_RADIUS_KM = 50.0


@dataclass
class CatalogGrid:
    """Precomputed TEME positions for a set of catalog objects over a shared fine time grid."""

    grid_times: list[datetime]
    jd_array: np.ndarray
    fr_array: np.ndarray
    satellites: list[Satellite]
    positions: np.ndarray  # shape (n_satellites, n_grid_times, 3), NaN where propagation failed


def build_catalog_grid(
    satellites: list[Satellite],
    grid_start: datetime,
    grid_end: datetime,
    grid_step: timedelta = timedelta(seconds=15),
) -> CatalogGrid:
    """
    Precompute every satellite's TEME position across a shared fine time
    grid, using the vectorized Propagator.propagate_teme_array() (one
    call per satellite across the whole grid).

    Args:
        satellites: catalog objects to include (a representative subset
            for interactive use — see stm/conjunction.py and
            viz/dashboard_3d.py for the same "subset of the full ~16,000
            object catalog" convention used elsewhere in this project).
        grid_start, grid_end: UTC time span the grid must cover (should
            span the whole launch-window search range PLUS the ascent
            duration, so every candidate launch's full ascent has grid
            coverage).
        grid_step: time resolution. Smaller = more accurate distance
            checks, more compute. 15 s is a reasonable default (at LEO
            relative speeds up to ~15 km/s, a satellite moves ~225 km in
            15 s — comparable to or smaller than the bubble radius scale,
            which is the right regime for this to be a meaningful screen).

    Returns:
        CatalogGrid ready to be passed to screen_launch_time().
    """
    n_steps = max(2, int((grid_end - grid_start) / grid_step) + 1)
    grid_times = [grid_start + i * grid_step for i in range(n_steps)]
    jd_array, fr_array = jd_fr_array_from_datetimes(grid_times)

    positions = np.full((len(satellites), n_steps, 3), np.nan)
    for i, sat in enumerate(satellites):
        try:
            prop = Propagator(sat.line1, sat.line2, name=sat.name)
        except Exception:
            continue
        error_array, r_array, _ = prop.propagate_teme_array(jd_array, fr_array)
        valid = error_array == 0
        positions[i, valid] = r_array[valid]

    return CatalogGrid(
        grid_times=grid_times, jd_array=jd_array, fr_array=fr_array,
        satellites=satellites, positions=positions,
    )


@dataclass
class ProximityEvent:
    """A single ascent-sample moment where a catalog object entered the safety bubble."""

    t_offset_s: float
    absolute_time: datetime
    satellite: Satellite
    distance_km: float


@dataclass
class COLAResult:
    """
    Outcome of screening one candidate launch time's ascent against the catalog.

    `trajectory` and `sample_min_distances_km` are included for
    visualization use (e.g. drawing the 3D ascent path colored by
    distance-to-nearest-object at each point) — they're the same
    per-sample data the screening loop already computes, just carried
    through rather than discarded, so a caller that wants to *draw* one
    specific candidate doesn't need a second, separate computation pass.
    """

    launch_time: datetime
    azimuth_deg: float
    clear: bool
    closest_approach_km: float
    violations: list[ProximityEvent] = field(default_factory=list)
    trajectory: list[TrajectorySample] = field(default_factory=list)
    sample_min_distances_km: list[float] = field(default_factory=list)


def _nearest_grid_index(grid_times: list[datetime], t: datetime) -> int:
    """Index of the grid_times entry closest to t (grid_times assumed sorted, evenly spaced)."""
    step_s = (grid_times[1] - grid_times[0]).total_seconds()
    offset_s = (t - grid_times[0]).total_seconds()
    idx = int(round(offset_s / step_s))
    return max(0, min(len(grid_times) - 1, idx))


def screen_launch_time(
    site: LaunchSite,
    target_inclination_deg: float,
    launch_time: datetime,
    catalog: CatalogGrid,
    target_altitude_km: float = 400.0,
    ascent_duration_s: float = 600.0,
    max_downrange_km: float = 2000.0,
    bubble_radius_km: float = DEFAULT_BUBBLE_RADIUS_KM,
    direction: str = "northeast",
) -> COLAResult:
    """
    Screen ONE candidate launch time: build its ascent trajectory, then
    at every ascent sample check every catalog object's distance (using
    the precomputed `catalog` grid) against `bubble_radius_km`.

    Returns:
        COLAResult — clear=True if no catalog object ever enters the
        bubble during the whole ascent; otherwise clear=False with every
        violating (sample, object, distance) recorded in `violations`.
    """
    azimuth = launch_azimuth(target_inclination_deg, site.lat_deg, direction=direction)
    trajectory = compute_ascent_trajectory(
        site, azimuth, launch_time, target_altitude_km,
        ascent_duration_s=ascent_duration_s, max_downrange_km=max_downrange_km,
    )

    violations = []
    closest_approach_km = float("inf")
    sample_min_distances_km = []

    for sample in trajectory:
        grid_idx = _nearest_grid_index(catalog.grid_times, sample.absolute_time)
        catalog_positions_now = catalog.positions[:, grid_idx, :]  # (n_satellites, 3)

        diffs = catalog_positions_now - sample.r_teme
        distances = np.linalg.norm(diffs, axis=1)  # NaN-safe: NaN propagates, never compares < threshold as True

        finite_mask = np.isfinite(distances)
        sample_min = float(np.min(distances[finite_mask])) if finite_mask.any() else float("inf")
        sample_min_distances_km.append(sample_min)
        closest_approach_km = min(closest_approach_km, sample_min)

        close_mask = finite_mask & (distances < bubble_radius_km)
        for idx in np.nonzero(close_mask)[0]:
            violations.append(
                ProximityEvent(
                    t_offset_s=sample.t_offset_s,
                    absolute_time=sample.absolute_time,
                    satellite=catalog.satellites[idx],
                    distance_km=float(distances[idx]),
                )
            )

    return COLAResult(
        launch_time=launch_time, azimuth_deg=azimuth, clear=(len(violations) == 0),
        closest_approach_km=closest_approach_km, violations=violations,
        trajectory=trajectory, sample_min_distances_km=sample_min_distances_km,
    )


def find_cola_launch_windows(
    site: LaunchSite,
    target_inclination_deg: float,
    satellites: list[Satellite],
    search_start: datetime,
    search_end: datetime,
    candidate_step: timedelta = timedelta(minutes=1),
    target_altitude_km: float = 400.0,
    ascent_duration_s: float = 600.0,
    max_downrange_km: float = 2000.0,
    bubble_radius_km: float = DEFAULT_BUBBLE_RADIUS_KM,
    grid_step: timedelta = timedelta(seconds=15),
    direction: str = "northeast",
) -> list[COLAResult]:
    """
    Full COLA scan: screen every candidate launch time from search_start
    to search_end (stepped by candidate_step) and return one COLAResult
    per candidate, in time order.

    This is the direct answer to "in the next N hours, which exact
    minutes can I launch in without hitting anything up there" — as
    opposed to core.launch_window.LaunchWindowCalculator, which only
    answers "when does Earth's rotation align the launch site with the
    target orbital plane" (a necessary condition, but not a traffic check).
    """
    n_candidates = max(1, int((search_end - search_start) / candidate_step) + 1)
    candidate_times = [search_start + i * candidate_step for i in range(n_candidates)]

    grid_end = search_end + timedelta(seconds=ascent_duration_s) + candidate_step
    catalog = build_catalog_grid(satellites, search_start, grid_end, grid_step=grid_step)

    results = []
    for launch_time in candidate_times:
        result = screen_launch_time(
            site, target_inclination_deg, launch_time, catalog,
            target_altitude_km=target_altitude_km, ascent_duration_s=ascent_duration_s,
            max_downrange_km=max_downrange_km, bubble_radius_km=bubble_radius_km, direction=direction,
        )
        results.append(result)

    return results
