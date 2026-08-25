"""
stm/keepout.py — Keep-out zones around protected objects.

Two zone shapes are supported:

  - Sphere: a simple protection radius around a satellite (e.g. 1-25 km
    around an operational satellite/active spacecraft).
  - Box (in the RSW/Hill frame): an oriented rectangular corridor around
    a protected object, e.g. the ISS's real operational approach
    corridor, which NASA defines as an elongated box rather than a
    sphere because along-track (orbit-direction) miss distance is far
    less risky than radial/cross-track miss distance at the same
    magnitude (orbital mechanics makes along-track separations decay
    much more slowly and predictably than a sphere model would suggest).

RSW (Radial-Transverse-Normal / "Hill") frame, defined at the protected
(chief) object's instantaneous state:
    R_hat = r / |r|                    (radial, chief-to-center-of-Earth outward)
    W_hat = (r x v) / |r x v|          (orbit-normal / cross-track)
    S_hat = W_hat x R_hat              (along-track, completes the right-handed triad)

Any other object's position is checked by projecting its relative
position onto this frame and comparing to the box half-dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KeepOutZone:
    """A protected volume around a chief object."""

    name: str
    shape: str                                  # "sphere" | "box"
    radius_km: float | None = None              # for shape == "sphere"
    box_half_dims_km: tuple[float, float, float] | None = None  # (radial, along-track, cross-track) half-widths


# ISS's real operational approach corridor: a 50 km x 50 km x 4 km box
# (per project spec / NASA conjunction-assessment practice), centered on
# the ISS. Expressed here as half-dimensions in the RSW frame.
ISS_CORRIDOR = KeepOutZone(
    name="ISS Approach Corridor",
    shape="box",
    box_half_dims_km=(25.0, 25.0, 2.0),
)

# Generic active-satellite protection sphere (project spec: 1-25 km).
DEFAULT_ACTIVE_SATELLITE_KEEPOUT = KeepOutZone(
    name="Active Satellite Protection Sphere",
    shape="sphere",
    radius_km=5.0,
)


def rsw_frame(r_chief: np.ndarray, v_chief: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the orthonormal RSW basis {R_hat, S_hat, W_hat} at the chief's state.

    Returns:
        (R_hat, S_hat, W_hat), each a unit 3-vector in the same inertial
        frame as r_chief/v_chief (e.g. TEME).
    """
    R_hat = r_chief / np.linalg.norm(r_chief)
    h = np.cross(r_chief, v_chief)
    W_hat = h / np.linalg.norm(h)
    S_hat = np.cross(W_hat, R_hat)
    return R_hat, S_hat, W_hat


def relative_position_rsw(r_chief: np.ndarray, v_chief: np.ndarray, r_deputy: np.ndarray) -> np.ndarray:
    """
    Express (r_deputy - r_chief) in the chief's RSW frame.

    Returns:
        np.array([radial, along_track, cross_track]) [km].
    """
    R_hat, S_hat, W_hat = rsw_frame(r_chief, v_chief)
    rel = r_deputy - r_chief
    return np.array([np.dot(rel, R_hat), np.dot(rel, S_hat), np.dot(rel, W_hat)])


def check_keepout_violation(
    r_chief: np.ndarray,
    v_chief: np.ndarray,
    r_deputy: np.ndarray,
    zone: KeepOutZone = DEFAULT_ACTIVE_SATELLITE_KEEPOUT,
) -> bool:
    """
    Check whether a deputy object's position violates a keep-out zone
    defined around a chief (protected) object.

    Args:
        r_chief, v_chief: chief's position/velocity [km, km/s] in an
            inertial frame (e.g. TEME) at the epoch being checked.
        r_deputy: the other object's position [km], same frame/epoch.
        zone: the KeepOutZone to test against.

    Returns:
        True if r_deputy is inside the zone.
    """
    if zone.shape == "sphere":
        if zone.radius_km is None:
            raise ValueError(f"Zone '{zone.name}' is shape='sphere' but has no radius_km set")
        distance = np.linalg.norm(r_deputy - r_chief)
        return distance <= zone.radius_km

    if zone.shape == "box":
        if zone.box_half_dims_km is None:
            raise ValueError(f"Zone '{zone.name}' is shape='box' but has no box_half_dims_km set")
        radial, along_track, cross_track = relative_position_rsw(r_chief, v_chief, r_deputy)
        hr, hs, hw = zone.box_half_dims_km
        return abs(radial) <= hr and abs(along_track) <= hs and abs(cross_track) <= hw

    raise ValueError(f"Unknown keep-out zone shape '{zone.shape}'")


def min_distance_to_zone_boundary(
    r_chief: np.ndarray,
    v_chief: np.ndarray,
    r_deputy: np.ndarray,
    zone: KeepOutZone = DEFAULT_ACTIVE_SATELLITE_KEEPOUT,
) -> float:
    """
    Signed-ish margin to the zone boundary: negative means inside the
    zone (violation), positive means outside. For a sphere this is exact
    (distance - radius); for a box it's an approximation (the minimum
    per-axis slack), useful for ranking near-misses rather than for
    precise geometric distance to the box surface.
    """
    if zone.shape == "sphere":
        distance = np.linalg.norm(r_deputy - r_chief)
        return distance - zone.radius_km

    radial, along_track, cross_track = relative_position_rsw(r_chief, v_chief, r_deputy)
    hr, hs, hw = zone.box_half_dims_km
    slack_r = hr - abs(radial)
    slack_s = hs - abs(along_track)
    slack_w = hw - abs(cross_track)
    return -min(slack_r, slack_s, slack_w)
