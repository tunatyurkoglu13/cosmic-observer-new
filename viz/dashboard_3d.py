"""
viz/dashboard_3d.py — Snapshot builder for the 3D dashboard frontend.

Assembles everything the Three.js dashboard needs into one JSON-serializable
structure: a short time series of satellite ground positions (lat/lon/alt,
so the frontend just needs a standard lat/lon -> XYZ sphere-surface
formula and never has to run SGP4 itself), the subsolar point and
terminator per frame (for day/night shading), and satellite metadata
(classification, for marker coloring per viz/retro_ui.py's palette).

This module produces a *static* multi-frame snapshot rather than a live
feed — real-time WebSocket propagation (per the project's Phase 7 plan)
is a separate concern layered on top of the same core.propagator engine.
A snapshot is what a local dev preview of the dashboard needs; it also
demonstrates the exact data contract the future WebSocket messages should
follow (same per-satellite fields, just pushed one frame at a time).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from core.kepler import orbital_period
from core.propagator import Propagator
from core.tle_manager import Satellite
from viz.earth_render import subsolar_point, terminator_points
from viz.orbit_trails import compute_ground_track


@dataclass
class SatelliteTrack:
    """One satellite's classification/metadata plus its lat/lon/alt at each frame."""

    norad_id: int
    name: str
    classification: str
    track: list[tuple[float, float, float]]  # (lat_deg, lon_deg, alt_km) per frame


@dataclass
class DashboardSnapshot:
    """Everything the 3D dashboard needs to render a time-scrubbable scene."""

    frame_times_iso: list[str]
    subsolar_track: list[tuple[float, float]]
    terminator_tracks: list[list[tuple[float, float]]]
    satellites: list[SatelliteTrack]
    # norad_id -> list of (lat, lon) segments (already antimeridian-split)
    # tracing one full orbital period, for drawing a real orbit ring —
    # populated separately by add_ground_tracks(), typically only for a
    # small highlighted subset (e.g. stations), since it's one extra
    # SGP4 pass per object at fine time resolution.
    ground_tracks: dict[int, list[list[tuple[float, float]]]] = None

    def __post_init__(self):
        if self.ground_tracks is None:
            self.ground_tracks = {}


def build_snapshot(
    satellites: list[Satellite],
    epoch: datetime,
    n_frames: int = 24,
    frame_step: timedelta = timedelta(hours=1),
    terminator_points_per_frame: int = 72,
) -> DashboardSnapshot:
    """
    Propagate a satellite list across n_frames sample times and assemble
    a DashboardSnapshot.

    Args:
        satellites: catalog objects to include (mix classifications
            freely; each object's `.classification` is carried through
            for frontend marker coloring).
        epoch: UTC start time for the frame series.
        n_frames: number of time samples.
        frame_step: spacing between samples.
        terminator_points_per_frame: resolution of the day/night
            terminator polyline per frame (lower = cheaper JSON, coarser line).

    Returns:
        DashboardSnapshot. Satellites that error out (e.g. SGP4 failure)
        at every frame are dropped entirely; a satellite that fails at
        only some frames keeps a shorter-than-n_frames track (the
        frontend should index tracks by nearest available frame rather
        than assuming exact alignment for such objects — in practice
        this only affects decayed/near-decayed debris).
    """
    frame_times = [epoch + i * frame_step for i in range(n_frames)]

    subsolar_track = [subsolar_point(t) for t in frame_times]
    terminator_tracks = [terminator_points(t, n_points=terminator_points_per_frame) for t in frame_times]

    satellite_tracks = []
    for sat in satellites:
        try:
            prop = Propagator(sat.line1, sat.line2, name=sat.name)
        except Exception:
            continue

        track = []
        for t in frame_times:
            try:
                state = prop.propagate(t)
                track.append((state.lat_deg, state.lon_deg, state.alt_km))
            except RuntimeError:
                continue

        if track:
            satellite_tracks.append(
                SatelliteTrack(
                    norad_id=sat.norad_id, name=sat.name,
                    classification=sat.classification, track=track,
                )
            )

    return DashboardSnapshot(
        frame_times_iso=[t.isoformat() for t in frame_times],
        subsolar_track=subsolar_track,
        terminator_tracks=terminator_tracks,
        satellites=satellite_tracks,
    )


def add_ground_tracks(
    snapshot: DashboardSnapshot,
    satellites: list[Satellite],
    epoch: datetime,
    step: timedelta = timedelta(minutes=1),
) -> DashboardSnapshot:
    """
    Populate snapshot.ground_tracks with a full-orbital-period ground
    track (lat/lon, antimeridian-split into segments) for each given
    satellite, starting at `epoch`. Intended for a small highlighted
    subset (e.g. the "stations" group) rather than the whole catalog,
    since each object costs one dedicated fine-resolution SGP4 pass.

    Mutates and returns the same snapshot for convenient chaining.
    """
    for sat in satellites:
        try:
            prop = Propagator(sat.line1, sat.line2, name=sat.name)
        except Exception:
            continue

        period_s = orbital_period(sat.semi_major_axis_km)
        segments = compute_ground_track(
            prop, epoch, duration=timedelta(seconds=period_s), step=step
        )
        if segments:
            snapshot.ground_tracks[sat.norad_id] = segments

    return snapshot


def snapshot_to_dict(snapshot: DashboardSnapshot) -> dict:
    """Plain-dict/JSON-serializable form of a DashboardSnapshot."""
    return asdict(snapshot)


def write_snapshot_json(snapshot: DashboardSnapshot, path: Path | str) -> None:
    """Write a DashboardSnapshot to disk as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot_to_dict(snapshot), f)
