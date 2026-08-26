"""
core/orbit_determination.py — Extended Kalman Filter (EKF) orbit determination.

This is a small-scale version of what real orbit-determination centers
(USSPACECOM's 18th Space Defense Squadron, ESA's Space Debris Office,
commercial operators' own OD pipelines) do continuously: fuse a stream of
noisy position observations with a dynamical model to produce a state
estimate that's better than either the raw observations or the model
alone, plus a covariance describing how confident that estimate is.

Why "Extended" Kalman Filter and not a plain (linear) Kalman filter: the
dynamics here (two-body + J2 gravity, via core.numerical_propagator) are
nonlinear in the state, so the filter linearizes them at each step around
the current estimate — propagating the state through the true nonlinear
dynamics, but propagating the *covariance* through a linearized state
transition matrix Phi (the Jacobian of one integration step). Phi is
computed by numerical (finite-difference) differentiation of the RK4
step function rather than an analytically-derived Jacobian — consistent
with this project's general policy (see core.numerical_propagator's
docstring) of preferring a systematically-checkable numerical approach
over a hand-derived formula wherever the analytic route is easy to get
subtly wrong.

Measurement model: position-only (H = [I3, 0]) — the common case for
optical/radar tracking data that reports position (or range+angles
reducible to position) without a direct velocity measurement.

Honest scope note: this module demonstrates the EKF machinery against
*synthetic* noisy position observations generated from a known truth
trajectory (the standard way to validate a filter — you need ground
truth to measure whether it actually helped). A real deployment would
feed it an actual observation stream (e.g. a satellite's historical TLE
history from Space-Track, once available per this project's plan — see
core.space_track_client, a later phase); the filter code itself doesn't
change, only what's fed into update().

References: Vallado, "Fundamentals of Astrodynamics and Applications"
(4th Ed), Ch. 10 (orbit determination); Tapley, Schutz & Born,
"Statistical Orbit Determination", Ch. 4 (the EKF itself).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.numerical_propagator import DynamicsConfig, rk4_step


def numerical_jacobian(state: np.ndarray, dt: float, config: DynamicsConfig, epsilon: float = 1e-3) -> np.ndarray:
    """
    Finite-difference Jacobian of one RK4 step (state_{k+1} = f(state_k))
    with respect to state_k — the state transition matrix Phi the EKF
    needs to propagate covariance: P_pred = Phi @ P @ Phi.T + Q.

    Central differences (perturb +/- epsilon per component) rather than
    forward differences: O(epsilon^2) truncation error instead of
    O(epsilon), meaningfully more accurate for the same epsilon, at 2x
    the function evaluations (12 RK4 steps total for a 6-dim state —
    cheap relative to a real dynamics model, and this project already
    uses the same "central-difference over hand-derived closed form"
    trade-off in core.numerical_propagator's zonal_acceleration()).
    """
    n = len(state)
    jacobian = np.zeros((n, n))
    for i in range(n):
        state_plus, state_minus = state.copy(), state.copy()
        state_plus[i] += epsilon
        state_minus[i] -= epsilon
        f_plus = rk4_step(0.0, state_plus, dt, config)
        f_minus = rk4_step(0.0, state_minus, dt, config)
        jacobian[:, i] = (f_plus - f_minus) / (2 * epsilon)
    return jacobian


@dataclass
class EKFStep:
    """One filter step's output: the updated state estimate and its covariance."""

    state: np.ndarray       # [x, y, z, vx, vy, vz]
    covariance: np.ndarray  # 6x6

    @property
    def position(self) -> np.ndarray:
        return self.state[:3]

    @property
    def position_std(self) -> np.ndarray:
        """1-sigma position uncertainty per axis (sqrt of the covariance diagonal)."""
        return np.sqrt(np.diag(self.covariance)[:3])


class OrbitEKF:
    """
    Extended Kalman Filter for satellite orbit determination, using
    core.numerical_propagator's dynamics as the process model.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        dynamics_config: DynamicsConfig | None = None,
    ):
        """
        Args:
            initial_state: starting [r, v] guess, km and km/s.
            initial_covariance: 6x6 initial state uncertainty.
            process_noise: 6x6 process noise Q (uncertainty added per
                predict step, accounting for unmodeled dynamics —
                perturbations this project's simplified dynamics config
                doesn't include, e.g. higher-order harmonics, solar
                radiation pressure).
            measurement_noise: 3x3 R, position measurement uncertainty.
            dynamics_config: which perturbations the process model
                includes (defaults to J2-J4 + drag, matching
                core.numerical_propagator.DynamicsConfig's own default).
        """
        self.x = np.asarray(initial_state, dtype=float).copy()
        self.P = np.asarray(initial_covariance, dtype=float).copy()
        self.Q = np.asarray(process_noise, dtype=float)
        self.R = np.asarray(measurement_noise, dtype=float)
        self.config = dynamics_config or DynamicsConfig()

    def predict(self, dt: float) -> None:
        """Propagate state and covariance forward by dt using the nonlinear dynamics + linearized Phi."""
        phi = numerical_jacobian(self.x, dt, self.config)
        self.x = rk4_step(0.0, self.x, dt, self.config)
        self.P = phi @ self.P @ phi.T + self.Q

    def update(self, position_observation: np.ndarray) -> None:
        """Incorporate one position measurement via the standard Kalman update."""
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        innovation = position_observation - H @ self.x
        innovation_covariance = H @ self.P @ H.T + self.R
        kalman_gain = self.P @ H.T @ np.linalg.inv(innovation_covariance)

        self.x = self.x + kalman_gain @ innovation
        self.P = (np.eye(6) - kalman_gain @ H) @ self.P

    def step(self, dt: float, position_observation: np.ndarray | None = None) -> EKFStep:
        """Convenience: predict, then update if an observation is given, returning the resulting estimate."""
        self.predict(dt)
        if position_observation is not None:
            self.update(position_observation)
        return EKFStep(state=self.x.copy(), covariance=self.P.copy())
