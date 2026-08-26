"""
core/numerical_propagator.py — Numerical (Cowell-method) orbit propagator.

SGP4 (core/propagator.py) is an *analytic* propagator: it works from a
closed-form perturbation theory fitted to average out short-period
effects, which is exactly why it's fast enough to propagate a 20,000+
object catalog but also why it's only as accurate as the perturbation
model NORAD built it around. Cowell's method is the opposite approach:
integrate Newton's law directly,

    d^2r/dt^2 = a_two-body(r) + a_J2..J4(r) + a_drag(r, v) + a_third-body(r, t)

numerically (RK4 here), with every perturbing acceleration added
explicitly and independently. It's slower and needs a well-characterized
initial state (not a TLE's mean elements), but it's the direct "ground
truth" numerical-integration approach used inside real precision orbit
determination systems, and comparing it against SGP4 for the same
initial conditions is a genuinely instructive way to see where/how much
an analytic propagator's simplifications start to matter.

Zonal gravity harmonics (J2, J3, J4, ...) are implemented via the
general geopotential expansion

    U(r, phi) = -(mu/r) * [1 - sum_{n=2}^{N} J_n (Re/r)^n P_n(sin(phi))]

(phi = geocentric latitude, P_n = ordinary Legendre polynomial of degree
n — see Vallado Ch. 8 or Montenbruck & Gill Ch. 3) with the acceleration
computed as a central-difference numerical gradient, a = -grad(U),
*rather than* hand-transcribed per-degree closed-form Cartesian
expressions. This is a deliberate choice: it only requires the scalar
J_n coefficient values (low-risk, easily cross-checked numbers) rather
than trusting a memorized/derived closed-form acceleration formula per
degree, and it's independently verified in this module's tests by
confirming the J2-only numerical gradient matches this project's
already-tested analytic J2 secular-rate derivation (core.propagator.j2_secular_rates).

Default J3/J4 values (JGM-3-derived, cross-checked against a second,
independent source — MathWorks Aerospace Blockset's zonal harmonic
gravity model — during development; both agree to 4 significant figures):
    J3 = -2.5323e-6
    J4 = -1.6204e-6

Atmospheric drag reuses data.space_weather.drag_acceleration() (the same
exponential atmosphere model used elsewhere in this project) rather than
a second implementation.

Third-body (Sun/Moon) perturbation is optional and degrades gracefully:
positions are fetched once per propagation call via data.horizons
(JPL Horizons) and linearly interpolated between samples (adequate,
since the Sun/Moon move negligibly on an integrator sub-step timescale)
— if Horizons is unreachable, third-body terms are simply omitted rather
than failing the whole propagation.

References: Vallado, "Fundamentals of Astrodynamics and Applications"
(4th Ed), Ch. 8-9; Montenbruck & Gill, "Satellite Orbits", Ch. 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from scipy.special import eval_legendre

from core.constants import MU_EARTH, R_EARTH
from data.space_weather import drag_acceleration

# Default zonal harmonic coefficients (see module docstring for sourcing/verification).
DEFAULT_J_COEFFICIENTS = {
    2: 1.0826269e-3,     # matches core.constants.J2 to 5 significant figures
    3: -2.5323e-6,
    4: -1.6204e-6,
}

MU_SUN = 1.32712440018e11    # km^3/s^2 (standard gravitational parameter of the Sun)
MU_MOON = 4902.800066        # km^3/s^2


def zonal_potential(r_vec: np.ndarray, j_coefficients: dict[int, float], mu: float = MU_EARTH, r_earth: float = R_EARTH) -> float:
    """
    Earth's gravitational potential at position r_vec, including the
    requested zonal harmonics (see module docstring for the formula).
    """
    r = np.linalg.norm(r_vec)
    sin_phi = np.clip(r_vec[2] / r, -1.0, 1.0)

    correction = sum(Jn * (r_earth / r) ** n * eval_legendre(n, sin_phi) for n, Jn in j_coefficients.items())
    return -(mu / r) * (1.0 - correction)


def zonal_acceleration(
    r_vec: np.ndarray,
    j_coefficients: dict[int, float] = DEFAULT_J_COEFFICIENTS,
    mu: float = MU_EARTH,
    r_earth: float = R_EARTH,
    step: float = 1e-3,
) -> np.ndarray:
    """
    Acceleration from Earth's zonal gravity harmonics, a = -grad(U),
    computed via central finite differences on zonal_potential(). With
    j_coefficients={} this reduces to plain two-body gravity (verify:
    grad(-mu/r) = mu*r_vec/r^3, so a = -mu*r_vec/r^3 — attractive,
    inverse-square, exactly Newtonian gravity).
    """
    grad = np.zeros(3)
    for i in range(3):
        r_plus, r_minus = r_vec.copy(), r_vec.copy()
        r_plus[i] += step
        r_minus[i] -= step
        grad[i] = (zonal_potential(r_plus, j_coefficients, mu, r_earth) - zonal_potential(r_minus, j_coefficients, mu, r_earth)) / (2 * step)
    return -grad


def third_body_acceleration(r_vec: np.ndarray, body_position: np.ndarray, mu_body: float) -> np.ndarray:
    """
    Third-body perturbing acceleration (Sun or Moon) on a satellite at
    r_vec, given the perturbing body's position `body_position` relative
    to Earth's center at the same epoch:

        a = mu_body * [ (s - r)/|s - r|^3 - s/|s|^3 ]

    The second term is the indirect effect (Earth itself is also pulled
    by the third body, so what matters for *relative* satellite motion
    is the differential attraction, not the raw force on the satellite).
    """
    s = body_position
    rel = s - r_vec
    return mu_body * (rel / np.linalg.norm(rel) ** 3 - s / np.linalg.norm(s) ** 3)


@dataclass
class ThirdBodyEphemeris:
    """Sparse-sampled Sun/Moon positions (km, Earth-centered) for linear interpolation during integration."""

    times_s: np.ndarray       # seconds since propagation start
    sun_positions: np.ndarray  # (N, 3)
    moon_positions: np.ndarray  # (N, 3)

    def interpolate(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        sun = np.array([np.interp(t_s, self.times_s, self.sun_positions[:, i]) for i in range(3)])
        moon = np.array([np.interp(t_s, self.times_s, self.moon_positions[:, i]) for i in range(3)])
        return sun, moon


def fetch_third_body_ephemeris(start: datetime, duration_s: float, sample_step: timedelta = timedelta(hours=6)) -> ThirdBodyEphemeris | None:
    """
    Fetch Sun and Moon positions (Earth-centered, km) at a coarse cadence
    spanning a propagation window, for third-body perturbation via
    ThirdBodyEphemeris.interpolate(). Returns None (rather than raising)
    if JPL Horizons is unreachable — callers should treat that as
    "proceed without third-body perturbation," not a hard failure; this
    is a genuinely optional refinement, not a fundamental force like
    two-body gravity or J2.
    """
    from data.horizons import HorizonsClient

    n_samples = max(2, int(duration_s / sample_step.total_seconds()) + 2)
    end = start + n_samples * sample_step

    # Horizons' date parser expects "YYYY-MM-DD HH:MM" (or similar
    # offset-free formats) — datetime.isoformat()'s "+00:00" UTC suffix
    # makes it fail with "Cannot interpret date" (confirmed directly
    # against the live API during development).
    start_str = start.strftime("%Y-%m-%d %H:%M")
    end_str = end.strftime("%Y-%m-%d %H:%M")

    try:
        client = HorizonsClient()
        sun_samples = client.fetch_vectors("sun", start_str, end_str, step_size=f"{int(sample_step.total_seconds() // 60)}m", center="500@399")
        moon_samples = client.fetch_vectors("moon", start_str, end_str, step_size=f"{int(sample_step.total_seconds() // 60)}m", center="500@399")
    except Exception:
        return None

    if not sun_samples or not moon_samples:
        return None

    times_s = np.array([(s.jd_tdb - sun_samples[0].jd_tdb) * 86400.0 for s in sun_samples])
    sun_positions = np.array([s.r_km for s in sun_samples])
    moon_positions = np.array([m.r_km for m in moon_samples[: len(sun_samples)]])

    return ThirdBodyEphemeris(times_s=times_s, sun_positions=sun_positions, moon_positions=moon_positions)


@dataclass
class DynamicsConfig:
    """Which perturbations to include in the equations of motion."""

    j_coefficients: dict[int, float] = field(default_factory=lambda: dict(DEFAULT_J_COEFFICIENTS))
    include_drag: bool = True
    drag_coefficient: float = 2.2
    area_m2: float = 10.0
    mass_kg: float = 500.0
    third_body_ephemeris: ThirdBodyEphemeris | None = None


def equations_of_motion(t_s: float, state: np.ndarray, config: DynamicsConfig) -> np.ndarray:
    """
    dx/dt for the 6-vector state [r, v]: velocity, then total acceleration
    (two-body + zonal harmonics + optional drag + optional third-body).
    """
    r, v = state[:3], state[3:]

    a = zonal_acceleration(r, config.j_coefficients)

    if config.include_drag:
        altitude_km = np.linalg.norm(r) - R_EARTH
        a = a + drag_acceleration(v, altitude_km, config.drag_coefficient, config.area_m2, config.mass_kg)

    if config.third_body_ephemeris is not None:
        sun_pos, moon_pos = config.third_body_ephemeris.interpolate(t_s)
        a = a + third_body_acceleration(r, sun_pos, MU_SUN)
        a = a + third_body_acceleration(r, moon_pos, MU_MOON)

    return np.concatenate([v, a])


def rk4_step(t_s: float, state: np.ndarray, dt: float, config: DynamicsConfig) -> np.ndarray:
    """One fixed-step 4th-order Runge-Kutta integration step."""
    k1 = equations_of_motion(t_s, state, config)
    k2 = equations_of_motion(t_s + dt / 2, state + dt / 2 * k1, config)
    k3 = equations_of_motion(t_s + dt / 2, state + dt / 2 * k2, config)
    k4 = equations_of_motion(t_s + dt, state + dt * k3, config)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@dataclass
class NumericalPropagationResult:
    times_s: np.ndarray
    states: np.ndarray  # (N, 6): [x,y,z,vx,vy,vz] per row, TEME-equivalent inertial frame


def propagate_numerically(
    initial_state: np.ndarray,
    duration_s: float,
    dt: float = 30.0,
    config: DynamicsConfig | None = None,
) -> NumericalPropagationResult:
    """
    Integrate a 6-vector initial state [r, v] (km, km/s, in an inertial
    frame — e.g. TEME, to be directly comparable with SGP4 output)
    forward by `duration_s` using fixed-step RK4.

    Args:
        initial_state: [x, y, z, vx, vy, vz], km and km/s.
        duration_s: total propagation duration.
        dt: fixed integration step [s]. Smaller = more accurate, slower;
            30s is a reasonable default for LEO (period ~90 min), giving
            ~180 steps/orbit.
        config: which perturbations to include; defaults to J2-J4 +
            drag, no third-body (third-body needs an explicit
            ThirdBodyEphemeris — see fetch_third_body_ephemeris()).

    Returns:
        NumericalPropagationResult with the full state history.
    """
    config = config or DynamicsConfig()
    n_steps = max(1, int(round(duration_s / dt)))

    times = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, 6))
    states[0] = initial_state

    for i in range(n_steps):
        states[i + 1] = rk4_step(times[i], states[i], dt, config)
        times[i + 1] = times[i] + dt

    return NumericalPropagationResult(times_s=times, states=states)
