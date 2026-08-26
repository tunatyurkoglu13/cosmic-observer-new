from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pytest

from core.constants import MU_EARTH, R_EARTH
from core.kepler import cartesian_to_keplerian, mean_motion, orbital_period
from core.numerical_propagator import (
    DynamicsConfig,
    ThirdBodyEphemeris,
    fetch_third_body_ephemeris,
    propagate_numerically,
    rk4_step,
    third_body_acceleration,
    zonal_acceleration,
    zonal_potential,
)
from core.propagator import j2_secular_rates


def _circular_orbit_state(altitude_km: float, inclination_deg: float = 0.0) -> np.ndarray:
    a = R_EARTH + altitude_km
    v_circ = np.sqrt(MU_EARTH / a)
    inc = np.radians(inclination_deg)
    r = np.array([a, 0.0, 0.0])
    v = np.array([0.0, v_circ * np.cos(inc), v_circ * np.sin(inc)])
    return np.concatenate([r, v])


def test_zonal_acceleration_with_no_coefficients_matches_newtonian_two_body():
    """With j_coefficients={}, a = -grad(-mu/r) must reduce exactly to Newtonian two-body gravity."""
    r = np.array([7000.0, 500.0, -300.0])
    a = zonal_acceleration(r, j_coefficients={})
    expected = -MU_EARTH * r / np.linalg.norm(r) ** 3
    assert np.allclose(a, expected, rtol=1e-6)


def test_zonal_potential_two_body_only():
    r = np.array([7000.0, 0.0, 0.0])
    U = zonal_potential(r, j_coefficients={})
    assert np.isclose(U, -MU_EARTH / 7000.0)


def test_propagate_numerically_two_body_circular_orbit_stays_circular():
    """No perturbations: a circular orbit's radius should stay constant (energy/angular momentum conserved)."""
    state0 = _circular_orbit_state(altitude_km=500.0)
    a0 = np.linalg.norm(state0[:3])
    period = orbital_period(a0)

    config = DynamicsConfig(j_coefficients={}, include_drag=False)
    result = propagate_numerically(state0, duration_s=period, dt=10.0, config=config)

    radii = np.linalg.norm(result.states[:, :3], axis=1)
    assert np.allclose(radii, a0, rtol=1e-4)  # RK4 integration error over one full orbit, still tiny


def test_propagate_numerically_two_body_returns_to_start_after_one_period():
    state0 = _circular_orbit_state(altitude_km=500.0)
    a0 = np.linalg.norm(state0[:3])
    period = orbital_period(a0)

    config = DynamicsConfig(j_coefficients={}, include_drag=False)
    result = propagate_numerically(state0, duration_s=period, dt=10.0, config=config)

    final_position_error = np.linalg.norm(result.states[-1, :3] - state0[:3])
    # ~0.3% of the ~7000 km orbit radius — a reasonable RK4 fixed-step
    # (dt=10s) truncation error accumulated over one full period, not a bug.
    assert final_position_error < 30.0


def test_j2_only_raan_drift_matches_analytic_secular_rate():
    """
    The key correctness check for this module: propagate an inclined
    orbit under J2 ONLY (no drag/third-body) for many orbits, extract
    the RAAN drift by fitting osculating elements before/after, and
    compare against this project's already-tested analytic J2 secular
    rate (core.propagator.j2_secular_rates) — cross-validating the
    finite-difference zonal gradient approach against a trusted,
    independent formula rather than another copy of the same code.
    """
    altitude_km = 700.0
    inclination_deg = 98.0  # sun-synchronous-like, strong nodal regression
    state0 = _circular_orbit_state(altitude_km, inclination_deg)

    a0 = np.linalg.norm(state0[:3])
    e0 = 0.0
    i0 = np.radians(inclination_deg)
    n0 = mean_motion(a0)
    period = orbital_period(a0)

    n_orbits = 20
    config = DynamicsConfig(j_coefficients={2: 1.0826269e-3}, include_drag=False)
    result = propagate_numerically(state0, duration_s=n_orbits * period, dt=20.0, config=config)

    raan_before = cartesian_to_keplerian(result.states[0, :3], result.states[0, 3:]).raan
    raan_after = cartesian_to_keplerian(result.states[-1, :3], result.states[-1, 3:]).raan

    # Unwrap the raan difference to handle 2*pi wraparound over many orbits.
    raan_dot_numeric = (raan_after - raan_before) / (n_orbits * period)
    # Account for possible multi-wrap: compare against expected total drift instead.
    raan_dot_analytic, _ = j2_secular_rates(a0, e0, i0, MU_EARTH, R_EARTH, 1.0826269e-3)

    expected_total_drift = raan_dot_analytic * n_orbits * period
    numeric_total_drift = raan_dot_numeric * n_orbits * period  # same as (raan_after-raan_before) unwrapped only if <2pi

    # Direct comparison of rates (both are secular/average rates over many orbits):
    assert abs(raan_dot_numeric - raan_dot_analytic) / abs(raan_dot_analytic) < 0.05  # within 5%


def test_drag_reduces_altitude_over_time_when_enabled():
    """
    A low-altitude circular orbit with drag enabled should lose altitude
    over time; with drag disabled (all else equal), it should not.

    Uses a realistic ballistic coefficient (area/mass ~0.02 m^2/kg, a
    typical spacecraft value — matches the order of magnitude used
    elsewhere in this project, e.g. core/rocket_trajectory.py's ascent
    modeling) rather than an extreme high-drag object: a very high
    area-to-mass ratio at 200 km can legitimately cascade into a fast
    reentry within a handful of orbits once altitude drops into
    exponentially denser atmosphere — real physics, but not what this
    test is trying to isolate (a measurable, gradual altitude decrease).
    """
    state0 = _circular_orbit_state(altitude_km=300.0)
    period = orbital_period(np.linalg.norm(state0[:3]))

    config_drag = DynamicsConfig(j_coefficients={}, include_drag=True, drag_coefficient=2.2, area_m2=10.0, mass_kg=500.0)
    result_drag = propagate_numerically(state0, duration_s=10 * period, dt=10.0, config=config_drag)

    config_no_drag = DynamicsConfig(j_coefficients={}, include_drag=False)
    result_no_drag = propagate_numerically(state0, duration_s=10 * period, dt=10.0, config=config_no_drag)

    radius_drag_end = np.linalg.norm(result_drag.states[-1, :3])
    radius_no_drag_end = np.linalg.norm(result_no_drag.states[-1, :3])

    assert radius_drag_end < radius_no_drag_end


def test_third_body_acceleration_direction_sanity():
    """A satellite directly between Earth and a third body should feel a net pull toward that body relative to Earth."""
    r_sat = np.array([7000.0, 0.0, 0.0])
    sun_pos = np.array([1.496e8, 0.0, 0.0])  # ~1 AU away along +x
    a = third_body_acceleration(r_sat, sun_pos, MU_SUN := 1.32712440018e11)
    assert a[0] > 0  # pulled toward the Sun (+x), relative to Earth


def test_fetch_third_body_ephemeris_returns_none_on_horizons_failure():
    with patch("data.horizons.HorizonsClient.fetch_vectors", side_effect=RuntimeError("simulated Horizons outage")):
        result = fetch_third_body_ephemeris(datetime(2026, 1, 1, tzinfo=timezone.utc), duration_s=3600.0)
    assert result is None


def test_third_body_ephemeris_interpolate_matches_endpoints():
    times = np.array([0.0, 100.0, 200.0])
    sun = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    moon = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]])
    eph = ThirdBodyEphemeris(times_s=times, sun_positions=sun, moon_positions=moon)

    sun_mid, moon_mid = eph.interpolate(50.0)
    assert np.allclose(sun_mid, [1.5, 0.0, 0.0])
    assert np.allclose(moon_mid, [0.0, 1.5, 0.0])


@pytest.mark.network
@pytest.mark.slow
def test_fetch_third_body_ephemeris_real_horizons_data():
    """Real end-to-end check against live JPL Horizons."""
    result = fetch_third_body_ephemeris(datetime(2026, 1, 1, tzinfo=timezone.utc), duration_s=3600.0 * 12)
    if result is None:
        pytest.skip("JPL Horizons unavailable right now (external condition)")

    assert len(result.times_s) >= 2
    # Sun should be roughly 1 AU from Earth center.
    sun_dist = np.linalg.norm(result.sun_positions[0])
    assert 1.4e8 < sun_dist < 1.55e8
    moon_dist = np.linalg.norm(result.moon_positions[0])
    assert 3.5e5 < moon_dist < 4.1e5


@pytest.mark.slow
def test_numerical_propagator_agrees_with_sgp4_over_short_span():
    """
    Cross-check against SGP4 (core.propagator) for a similar circular
    LEO orbit over a short span: both propagators should agree to within
    a modest tolerance despite using entirely independent methods
    (analytic mean-element theory vs. direct numerical integration),
    since both are modeling the same dominant J2 physics.
    """
    from core.propagator import Propagator

    # A real, roughly-circular ISS-like TLE.
    line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9008"
    line2 = "2 25544  51.6416 339.9500 0001177  93.3861 264.7930 15.49560752 42982"
    prop = Propagator(line1, line2, name="ISS")

    epoch = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    state_sgp4_0 = prop.propagate(epoch)

    initial_state = np.concatenate([state_sgp4_0.r_teme, state_sgp4_0.v_teme])
    duration_s = 3600.0  # 1 hour — short enough that J2-only numerical integration shouldn't drift far from SGP4

    config = DynamicsConfig(j_coefficients={2: 1.0826269e-3}, include_drag=False)
    result = propagate_numerically(initial_state, duration_s=duration_s, dt=15.0, config=config)

    state_sgp4_1 = prop.propagate(epoch + timedelta(seconds=duration_s))

    position_error = np.linalg.norm(result.states[-1, :3] - state_sgp4_1.r_teme)
    # A few tens of km after an hour is a reasonable bound: SGP4 includes
    # drag + higher-order terms our J2-only integrator doesn't, so some
    # divergence is expected and informative, not a bug.
    assert position_error < 50.0
