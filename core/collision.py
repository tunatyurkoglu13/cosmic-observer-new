"""
core/collision.py — Conjunction screening and collision probability (Pc).

Implements the standard three-stage conjunction assessment funnel used
operationally (e.g. NASA CARA, SOCRATES):

    1. Coarse screen  — apogee/perigee altitude-band overlap (cheap, O(n)).
    2. Time search    — find Time of Closest Approach (TCA) between two
                         propagated trajectories over a search window.
    3. Risk metrics   — Mahalanobis distance as a fast reject filter, then a
                         full 2D B-plane collision-probability (Pc) integral
                         (Foster's method: numerical integration of the
                         combined-covariance Gaussian PDF over the
                         combined hard-body disk, in polar coordinates)
                         for anything that survives the filter.

Risk thresholds follow NASA CARA convention:
    Pc < 1e-7            -> green  (negligible)
    1e-7 <= Pc < 1e-4     -> yellow (monitor)
    Pc >= 1e-4            -> red    (maneuver should be considered)

References: Foster, J.L. & Estes, H.S., "A Parametric Analysis of Orbital
Debris Collision Probability" (NASA/JSC, 1992); Alfriend, K.T. et al.,
"Probability of Collision Error Analysis" (1999); NASA CARA handbook.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from scipy import integrate, optimize

from core.propagator import Propagator

# NASA CARA risk thresholds
PC_THRESHOLD_YELLOW = 1e-7
PC_THRESHOLD_RED = 1e-4

# Fast-reject Mahalanobis distance: beyond this, treat Pc as negligible
# without paying for the full 2D integral (~3-sigma-equivalent for a 2D
# combined covariance ellipse).
MAHALANOBIS_REJECT_DISTANCE = 8.0


@dataclass
class ConjunctionResult:
    """Outcome of a close-approach analysis between two objects."""

    tca: datetime                 # time of closest approach (UTC)
    miss_distance_km: float       # |r1 - r2| at TCA
    relative_speed_km_s: float    # |v1 - v2| at TCA
    mahalanobis_distance: float | None = None
    probability_of_collision: float | None = None

    @property
    def risk_level(self) -> str:
        if self.probability_of_collision is None:
            return "unscreened"
        if self.probability_of_collision >= PC_THRESHOLD_RED:
            return "red"
        if self.probability_of_collision >= PC_THRESHOLD_YELLOW:
            return "yellow"
        return "green"


def altitude_bands_overlap(perigee1: float, apogee1: float, perigee2: float, apogee2: float, margin_km: float = 50.0) -> bool:
    """
    Stage-1 coarse screen: two orbits can only produce a close approach if
    their altitude bands (perigee..apogee, expanded by a safety margin)
    overlap. This is a cheap O(1) test used to prune the O(n^2) catalog
    pair search down before any propagation is done.
    """
    lo1, hi1 = perigee1 - margin_km, apogee1 + margin_km
    lo2, hi2 = perigee2 - margin_km, apogee2 + margin_km
    return lo1 <= hi2 and lo2 <= hi1


def find_time_of_closest_approach(
    prop1: Propagator,
    prop2: Propagator,
    t_start: datetime,
    t_end: datetime,
    coarse_step: timedelta = timedelta(minutes=1),
) -> ConjunctionResult:
    """
    Stage-2: locate the Time of Closest Approach (TCA) between two objects
    over [t_start, t_end].

    Two-pass search:
      (a) Coarse pass — sample the separation distance every `coarse_step`
          across the window and find the minimum.
      (b) Refine — run a bounded 1D minimization (Brent's method) in the
          neighborhood of the coarse minimum to pinpoint TCA to sub-second
          precision, since orbital relative motion near TCA is locally
          smooth and unimodal.

    Returns:
        ConjunctionResult with tca, miss_distance_km, relative_speed_km_s
        populated (mahalanobis_distance / probability_of_collision are
        filled in later by mahalanobis_distance() / probability_of_collision()).
    """
    n_samples = max(2, int((t_end - t_start) / coarse_step) + 1)
    sample_times = [t_start + i * coarse_step for i in range(n_samples)]

    def separation(dt: datetime) -> float:
        s1 = prop1.propagate(dt)
        s2 = prop2.propagate(dt)
        return float(np.linalg.norm(s1.r_teme - s2.r_teme))

    distances = [separation(t) for t in sample_times]
    min_idx = int(np.argmin(distances))

    lo = sample_times[max(min_idx - 1, 0)]
    hi = sample_times[min(min_idx + 1, n_samples - 1)]

    def objective(offset_seconds: float) -> float:
        t = t_start + timedelta(seconds=offset_seconds)
        return separation(t)

    lo_s = (lo - t_start).total_seconds()
    hi_s = (hi - t_start).total_seconds()
    if hi_s <= lo_s:
        hi_s = lo_s + coarse_step.total_seconds()

    res = optimize.minimize_scalar(objective, bounds=(lo_s, hi_s), method="bounded")
    tca = t_start + timedelta(seconds=float(res.x))

    s1 = prop1.propagate(tca)
    s2 = prop2.propagate(tca)
    miss_distance = float(np.linalg.norm(s1.r_teme - s2.r_teme))
    rel_speed = float(np.linalg.norm(s1.v_teme - s2.v_teme))

    return ConjunctionResult(tca=tca, miss_distance_km=miss_distance, relative_speed_km_s=rel_speed)


def mahalanobis_distance(miss_vector: np.ndarray, combined_covariance: np.ndarray) -> float:
    """
    Mahalanobis distance of the miss vector under the combined position
    covariance C = C1 + C2:

        d_M = sqrt(r^T C^-1 r)

    Used as a cheap Stage-3 pre-filter: pairs with d_M above
    MAHALANOBIS_REJECT_DISTANCE have negligible Pc and can skip the full
    B-plane integral.
    """
    C_inv = np.linalg.inv(combined_covariance)
    return float(np.sqrt(miss_vector @ C_inv @ miss_vector))


def b_plane_frame(r_rel: np.ndarray, v_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the two in-plane B-plane basis vectors {e_x, e_y} orthogonal
    to the relative velocity at TCA. The B-plane is the plane through the
    secondary object, perpendicular to the relative velocity vector — the
    encounter geometry collapses to this 2D plane because, to first order,
    relative motion near TCA is a straight line through it.

    Returns:
        (e_x, e_y): orthonormal in-plane basis vectors (3-vectors each).
    """
    v_hat = v_rel / np.linalg.norm(v_rel)

    # Any vector not parallel to v_hat, then Gram-Schmidt it into the plane.
    helper = np.array([1.0, 0.0, 0.0]) if abs(v_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e_x = helper - np.dot(helper, v_hat) * v_hat
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(v_hat, e_x)

    return e_x, e_y


def probability_of_collision(
    miss_vector_eci: np.ndarray,
    v_rel_eci: np.ndarray,
    combined_covariance_eci: np.ndarray,
    hard_body_radius_km: float,
) -> float:
    """
    Stage-3 full risk metric: 2D collision probability via Foster's method.

    Procedure:
      1. Project the 3x3 combined position covariance and the miss vector
         onto the 2D B-plane (perpendicular to relative velocity at TCA).
      2. Integrate the resulting 2D Gaussian PDF over a disk of radius
         `hard_body_radius_km` (the sum of the two objects' combined
         collision radii) centered at the origin, offset by the miss
         vector, in polar coordinates:

            Pc = (1 / (2*pi*sqrt(|C|))) *
                 integral_0^{2pi} integral_0^{HBR}
                     exp(-0.5 * r_vec^T C^-1 r_vec) * rho d(rho) d(theta)

         where r_vec = miss_vector_2d + rho*(cos theta, sin theta).

    Args:
        miss_vector_eci: r1 - r2 at TCA, in ECI/TEME [km].
        v_rel_eci: v1 - v2 at TCA, in ECI/TEME [km/s] (defines the B-plane normal).
        combined_covariance_eci: C1 + C2, 3x3 position covariance [km^2].
        hard_body_radius_km: combined collision radius (sum of both objects' radii).

    Returns:
        Probability of collision, in [0, 1].
    """
    e_x, e_y = b_plane_frame(miss_vector_eci, v_rel_eci)
    B = np.vstack([e_x, e_y])  # 2x3 projection matrix

    miss_2d = B @ miss_vector_eci
    C_2d = B @ combined_covariance_eci @ B.T

    C_inv = np.linalg.inv(C_2d)
    det_C = np.linalg.det(C_2d)
    norm_factor = 1.0 / (2 * np.pi * np.sqrt(det_C))

    def integrand(rho: float, theta: float) -> float:
        dx = miss_2d[0] + rho * np.cos(theta)
        dy = miss_2d[1] + rho * np.sin(theta)
        r_vec = np.array([dx, dy])
        exponent = -0.5 * (r_vec @ C_inv @ r_vec)
        return norm_factor * np.exp(exponent) * rho

    pc, _ = integrate.dblquad(
        integrand,
        0.0, 2 * np.pi,
        lambda theta: 0.0, lambda theta: hard_body_radius_km,
    )
    return float(np.clip(pc, 0.0, 1.0))


def assess_conjunction(
    conjunction: ConjunctionResult,
    r1: np.ndarray,
    r2: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    covariance1_km2: np.ndarray,
    covariance2_km2: np.ndarray,
    hard_body_radius_km: float,
) -> ConjunctionResult:
    """
    Fill in mahalanobis_distance and probability_of_collision on an
    already-located ConjunctionResult, applying the Stage-3 fast-reject
    before paying for the full B-plane integral.
    """
    miss_vector = r1 - r2
    v_rel = v1 - v2
    combined_cov = covariance1_km2 + covariance2_km2

    d_m = mahalanobis_distance(miss_vector, combined_cov)
    conjunction.mahalanobis_distance = d_m

    if d_m > MAHALANOBIS_REJECT_DISTANCE:
        conjunction.probability_of_collision = 0.0
        return conjunction

    conjunction.probability_of_collision = probability_of_collision(
        miss_vector, v_rel, combined_cov, hard_body_radius_km
    )
    return conjunction
