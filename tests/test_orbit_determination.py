import numpy as np
import pytest

from core.constants import MU_EARTH, R_EARTH
from core.numerical_propagator import DynamicsConfig, rk4_step
from core.orbit_determination import OrbitEKF, numerical_jacobian


def _circular_orbit_state(altitude_km: float = 400.0) -> np.ndarray:
    a = R_EARTH + altitude_km
    v_circ = np.sqrt(MU_EARTH / a)
    return np.array([a, 0.0, 0.0, 0.0, v_circ, 0.0])


def test_numerical_jacobian_two_body_matches_known_linear_behavior():
    """
    Sanity check on the Jacobian machinery itself: for a short dt, the
    velocity block's derivative w.r.t. position should be small and the
    position block's derivative w.r.t. velocity should be close to dt*I
    (to first order, position_{k+1} ~ position_k + dt*velocity_k).
    """
    state = _circular_orbit_state()
    config = DynamicsConfig(j_coefficients={}, include_drag=False)
    dt = 1.0  # short step, so the first-order approximation is tight

    phi = numerical_jacobian(state, dt, config)
    # d(position_{k+1}) / d(velocity_k) should be close to dt * I3
    upper_right_block = phi[:3, 3:]
    assert np.allclose(upper_right_block, dt * np.eye(3), atol=1e-2)


def test_ekf_predict_only_matches_plain_rk4_propagation():
    """With no measurement updates, the EKF's state estimate must exactly track plain RK4 propagation."""
    state0 = _circular_orbit_state()
    config = DynamicsConfig(j_coefficients={}, include_drag=False)

    P0 = np.eye(6) * 0.01
    Q = np.eye(6) * 1e-9
    R = np.eye(3) * 1.0

    ekf = OrbitEKF(state0, P0, Q, R, dynamics_config=config)

    direct_state = state0.copy()
    dt = 30.0
    for _ in range(10):
        ekf.predict(dt)
        direct_state = rk4_step(0.0, direct_state, dt, config)

    assert np.allclose(ekf.x, direct_state, rtol=1e-9)


def test_ekf_reduces_error_relative_to_raw_noisy_observations():
    """
    The central demonstration: generate a truth trajectory, corrupt it
    with Gaussian position noise to simulate 'observations', and verify
    the EKF's filtered position estimate has lower average error against
    truth than the raw noisy observations do — the whole point of
    running a filter instead of just trusting each measurement directly.
    """
    rng = np.random.default_rng(0)
    config = DynamicsConfig(j_coefficients={}, include_drag=False)  # consistent process model == truth model (no mismatch)

    truth = _circular_orbit_state()
    dt = 30.0
    n_steps = 80
    noise_std = 2.0  # km

    truths = [truth.copy()]
    observations = []
    for _ in range(n_steps):
        truth = rk4_step(0.0, truth, dt, config)
        truths.append(truth.copy())
        observations.append(truth[:3] + rng.normal(0, noise_std, size=3))

    # Initialize with a deliberately imperfect guess (first noisy observation + a rough velocity).
    x0 = np.concatenate([observations[0], truths[0][3:]])
    P0 = np.diag([noise_std**2] * 3 + [0.05**2] * 3)
    Q = np.diag([1e-8] * 3 + [1e-10] * 3)
    R = np.eye(3) * noise_std**2

    ekf = OrbitEKF(x0, P0, Q, R, dynamics_config=config)

    filtered_positions = []
    for obs in observations[1:]:
        step_result = ekf.step(dt, position_observation=obs)
        filtered_positions.append(step_result.position)

    # Compare errors after an initial transient (letting the filter converge past its imperfect initial guess).
    burn_in = 15
    raw_errors = [np.linalg.norm(observations[i][:3] - truths[i + 1][:3]) for i in range(burn_in, len(filtered_positions))]
    filtered_errors = [np.linalg.norm(filtered_positions[i] - truths[i + 2][:3]) for i in range(burn_in, len(filtered_positions))]

    assert np.mean(filtered_errors) < np.mean(raw_errors)


def test_ekf_covariance_stays_positive_semidefinite():
    """A numerically healthy filter's covariance must remain symmetric positive-semidefinite throughout."""
    rng = np.random.default_rng(1)
    state0 = _circular_orbit_state()
    config = DynamicsConfig(j_coefficients={}, include_drag=False)

    ekf = OrbitEKF(
        state0, np.eye(6) * 1.0, np.eye(6) * 1e-8, np.eye(3) * 1.0, dynamics_config=config,
    )

    truth = state0.copy()
    for _ in range(20):
        truth = rk4_step(0.0, truth, 30.0, config)
        obs = truth[:3] + rng.normal(0, 1.0, size=3)
        ekf.step(30.0, position_observation=obs)

        eigenvalues = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigenvalues > -1e-8)  # allow tiny negative numerical noise
        assert np.allclose(ekf.P, ekf.P.T, atol=1e-6)


def test_ekf_position_std_shrinks_as_observations_accumulate():
    """Uncertainty should generally shrink as more measurements are incorporated (starting from a loose prior)."""
    rng = np.random.default_rng(2)
    state0 = _circular_orbit_state()
    config = DynamicsConfig(j_coefficients={}, include_drag=False)

    ekf = OrbitEKF(
        state0, np.eye(6) * 100.0, np.eye(6) * 1e-9, np.eye(3) * 1.0, dynamics_config=config,
    )

    truth = state0.copy()
    initial_std = ekf.step(30.0, position_observation=truth[:3]).position_std.mean()

    for _ in range(15):
        truth = rk4_step(0.0, truth, 30.0, config)
        obs = truth[:3] + rng.normal(0, 1.0, size=3)
        result = ekf.step(30.0, position_observation=obs)

    assert result.position_std.mean() < initial_std
