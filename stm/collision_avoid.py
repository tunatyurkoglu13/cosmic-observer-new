"""
stm/collision_avoid.py — Collision-avoidance (COLA) maneuver planning.

Given a flagged conjunction (core.collision.ConjunctionResult / a
ConjunctionSummary from stm/conjunction.py), compute the manuever
(delta-v) needed to move the miss distance to a safe separation.

Two maneuver families are modeled:

  1. In-plane (along-track) separation burn, via the Clohessy-Wiltshire
     (Hill's) linearized relative-motion equations. This is the
     workhorse maneuver for real LEO collision avoidance: a small
     tangential burn well before TCA produces a *secular* along-track
     drift that grows over time, so even a tiny delta-v can open a large
     miss distance if applied early enough — which is why operators
     prefer to maneuver as early as possible after a conjunction is
     flagged rather than waiting.

  2. Cross-track (plane-change) burn, for cases where an inclination or
     RAAN nudge is preferred (e.g. to preserve semi-major axis / ground
     track), using the standard vis-viva-derived plane-change formula.

Clohessy-Wiltshire equations (Vallado Ch. 7 / Curtis Ch. 7), for a
deputy relative to a circular-orbit chief with mean motion n, given an
impulsive velocity change (dvx, dvy, dvz) applied at t=0 with zero
initial relative position:

    x(t) = (dvx/n) sin(nt) - (2 dvy/n)(cos(nt) - 1)
    y(t) = (2 dvx/n)(cos(nt) - 1) + (4 dvy/n) sin(nt) - 3 dvy t
    z(t) = (dvz/n) sin(nt)

x = radial, y = along-track, z = cross-track (orbit-normal).

The dominant term for a pure along-track burn (dvx = dvz = 0) at large t
is the secular -3*dvy*t in y(t): this is where the well-known "Δv burn
produces ~3*Δv*t of along-track drift" rule of thumb comes from.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from core.collision import ConjunctionResult

# CW equations are only valid while the relative separation stays small
# compared to the orbit radius (a few tens of km at most for LEO); beyond
# that, nonlinear terms (which we ignore) become significant.
CW_VALIDITY_LIMIT_KM = 50.0


@dataclass
class CollisionAvoidanceManeuver:
    """A planned single-impulse COLA burn."""

    burn_time: datetime
    time_to_tca_s: float
    maneuver_type: str              # "along-track" | "cross-track" | "combined"
    delta_v_km_s: float             # burn magnitude
    predicted_miss_distance_km: float


def clohessy_wiltshire_state(
    dv: np.ndarray, mean_motion_rad_s: float, t: float
) -> np.ndarray:
    """
    Relative position (x, y, z) at time t after an impulsive velocity
    change dv = (dvx, dvy, dvz) applied at t=0 from a zero initial
    relative offset, per the CW closed-form solution above.

    Args:
        dv: velocity impulse in the RSW/Hill frame [km/s]
            (x=radial, y=along-track, z=cross-track).
        mean_motion_rad_s: chief orbit's mean motion n [rad/s].
        t: elapsed time since the burn [s].

    Returns:
        Relative position (x, y, z) [km] in the same frame.
    """
    n = mean_motion_rad_s
    dvx, dvy, dvz = dv
    nt = n * t

    x = (dvx / n) * np.sin(nt) - (2 * dvy / n) * (np.cos(nt) - 1)
    y = (2 * dvx / n) * (np.cos(nt) - 1) + (4 * dvy / n) * np.sin(nt) - 3 * dvy * t
    z = (dvz / n) * np.sin(nt)

    return np.array([x, y, z])


def along_track_dv_for_target_miss_distance(
    current_miss_distance_km: float,
    target_miss_distance_km: float,
    time_to_tca_s: float,
) -> float:
    """
    Solve for the along-track delta-v (dvy) that shifts the miss distance
    from `current_miss_distance_km` to `target_miss_distance_km` by TCA,
    using only the CW secular term y(t) = -3*dvy*t (valid when the burn is
    applied well before TCA, so the oscillatory sin/cos terms have
    averaged out relative to the secular drift — the standard
    simplification used for early-warning COLA planning).

    Args:
        current_miss_distance_km: predicted miss distance with no maneuver.
        target_miss_distance_km: desired miss distance (should exceed the
            combined hard-body radius / your chosen safety margin).
        time_to_tca_s: seconds between the planned burn and TCA. Must be
            positive — the whole point of early maneuvering is that a
            smaller dv suffices the earlier it's applied.

    Returns:
        Signed delta-v [km/s] (sign indicates prograde vs retrograde;
        magnitude is what matters for propellant budgeting).
    """
    if time_to_tca_s <= 0:
        raise ValueError("time_to_tca_s must be positive — maneuver must precede TCA")

    delta_miss = target_miss_distance_km - current_miss_distance_km
    dvy = -delta_miss / (3 * time_to_tca_s)
    return dvy


def plan_along_track_maneuver(
    conjunction: ConjunctionResult,
    burn_time: datetime,
    mean_motion_rad_s: float,
    target_miss_distance_km: float = 5.0,
) -> CollisionAvoidanceManeuver:
    """
    Plan a minimum-effort in-plane separation burn for a flagged conjunction.

    Args:
        conjunction: the ConjunctionResult (must have .tca and
            .miss_distance_km populated, e.g. from
            core.collision.find_time_of_closest_approach()).
        burn_time: proposed UTC maneuver epoch (must be before conjunction.tca).
        mean_motion_rad_s: maneuvering object's mean motion n [rad/s].
        target_miss_distance_km: desired resulting miss distance.

    Returns:
        CollisionAvoidanceManeuver describing the required burn.
    """
    time_to_tca_s = (conjunction.tca - burn_time).total_seconds()
    dvy = along_track_dv_for_target_miss_distance(
        conjunction.miss_distance_km, target_miss_distance_km, time_to_tca_s
    )

    if target_miss_distance_km > CW_VALIDITY_LIMIT_KM:
        raise ValueError(
            f"target_miss_distance_km={target_miss_distance_km} exceeds the "
            f"CW linearization validity range (~{CW_VALIDITY_LIMIT_KM} km); "
            "use full nonlinear propagation to verify a maneuver this large."
        )

    return CollisionAvoidanceManeuver(
        burn_time=burn_time,
        time_to_tca_s=time_to_tca_s,
        maneuver_type="along-track",
        delta_v_km_s=abs(dvy),
        predicted_miss_distance_km=target_miss_distance_km,
    )


def plane_change_dv(v_km_s: float, delta_i_rad: float) -> float:
    """
    Pure inclination-change delta-v:

        dv = 2 * v * sin(delta_i / 2)

    Used when a cross-track nudge is preferred over an along-track burn
    (e.g. to avoid disturbing an operational ground-track or altitude).
    Note this is far more expensive in delta-v than an along-track burn
    for the same resulting miss-distance change at typical LEO speeds,
    which is why along-track maneuvers are the default COLA choice.
    """
    return 2 * v_km_s * np.sin(delta_i_rad / 2)


def combined_maneuver_dv(v1_km_s: float, v2_km_s: float, delta_i_rad: float) -> float:
    """
    Combined speed-and-plane-change delta-v (law of cosines on the
    velocity triangle):

        dv = sqrt(v1^2 + v2^2 - 2*v1*v2*cos(delta_i))

    Reduces to plane_change_dv() when v1 == v2.
    """
    return np.sqrt(v1_km_s**2 + v2_km_s**2 - 2 * v1_km_s * v2_km_s * np.cos(delta_i_rad))
