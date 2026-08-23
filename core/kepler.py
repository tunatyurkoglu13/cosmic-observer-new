"""
core/kepler.py — Keplerian orbital elements and coordinate conversions.

This module is the "textbook" layer of the propagation stack: it works with
the classical two-body (Keplerian) orbit model, ignoring perturbations. It is
used for (a) teaching / visualizing orbit geometry, (b) converting between
representations (Keplerian <-> Cartesian), and (c) as a cross-check for the
SGP4 propagator in propagator.py, which includes drag and J2 secular terms.

Equations follow Vallado, "Fundamentals of Astrodynamics and Applications"
(4th Ed.), Chapter 2.

Classical (osculating) elements used throughout:
    a     semi-major axis            [km]
    e     eccentricity               [-]
    i     inclination                [rad]
    raan  right ascension of the
          ascending node (Omega)     [rad]
    argp  argument of perigee (omega)[rad]
    nu    true anomaly               [rad]
    M     mean anomaly               [rad]
    E     eccentric anomaly          [rad]
"""

from dataclasses import dataclass

import numpy as np

from core.constants import MU_EARTH


@dataclass
class KeplerianElements:
    """Classical orbital elements at a single epoch."""

    a: float      # semi-major axis [km]
    e: float      # eccentricity [-]
    i: float      # inclination [rad]
    raan: float   # right ascension of ascending node [rad]
    argp: float   # argument of perigee [rad]
    nu: float     # true anomaly [rad]


def mean_motion(a: float, mu: float = MU_EARTH) -> float:
    """
    Mean motion n = sqrt(mu / a^3)  [rad/s].

    This is the constant angular rate a body would sweep through its orbit
    at if it moved uniformly (it only equals the *true* angular rate at
    perigee/apogee for an eccentric orbit).
    """
    return np.sqrt(mu / a**3)


def semi_major_axis_from_mean_motion(n: float, mu: float = MU_EARTH) -> float:
    """
    Inverse of mean_motion(): a = (mu / n^2)^(1/3)  [km].

    Used to recover the semi-major axis from a TLE's mean motion field
    (which is given in revolutions/day and must be converted to rad/s
    before calling this function).
    """
    return (mu / n**2) ** (1.0 / 3.0)


def orbital_period(a: float, mu: float = MU_EARTH) -> float:
    """Orbital period T = 2*pi*sqrt(a^3 / mu)  [s]."""
    return 2 * np.pi * np.sqrt(a**3 / mu)


def solve_kepler_equation(M: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """
    Solve Kepler's equation M = E - e*sin(E) for the eccentric anomaly E,
    given mean anomaly M and eccentricity e, via Newton-Raphson iteration.

    f(E)  = E - e*sin(E) - M
    f'(E) = 1 - e*cos(E)
    E_{k+1} = E_k - f(E_k) / f'(E_k)

    Initial guess: E_0 = M for low eccentricity, else E_0 = pi (standard
    practice per Vallado to guarantee convergence for e close to 1).

    Args:
        M: mean anomaly [rad], any real value (will be wrapped internally).
        e: eccentricity, 0 <= e < 1.
        tol: convergence tolerance on |f(E)| [rad].
        max_iter: safety cap on iterations.

    Returns:
        Eccentric anomaly E [rad], wrapped to [0, 2*pi).
    """
    if not (0.0 <= e < 1.0):
        raise ValueError("solve_kepler_equation only supports elliptical orbits (0 <= e < 1)")

    M = M % (2 * np.pi)
    E = M if e < 0.8 else np.pi

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        d = f / fp
        E -= d
        if abs(d) < tol:
            break

    return E % (2 * np.pi)


def true_anomaly_from_eccentric(E: float, e: float) -> float:
    """
    True anomaly from eccentric anomaly:
        tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)

    Implemented via atan2 on both half-angle components so the correct
    quadrant is preserved for E anywhere in [0, 2*pi).
    """
    sqrt_term = np.sqrt((1 + e) / (1 - e))
    nu = 2 * np.arctan2(sqrt_term * np.sin(E / 2), np.cos(E / 2))
    return nu % (2 * np.pi)


def eccentric_anomaly_from_true(nu: float, e: float) -> float:
    """Inverse of true_anomaly_from_eccentric(): E from nu via the same half-angle identity."""
    sqrt_term = np.sqrt((1 - e) / (1 + e))
    E = 2 * np.arctan2(sqrt_term * np.sin(nu / 2), np.cos(nu / 2))
    return E % (2 * np.pi)


def mean_anomaly_from_eccentric(E: float, e: float) -> float:
    """Kepler's equation, forward direction: M = E - e*sin(E)."""
    return (E - e * np.sin(E)) % (2 * np.pi)


def keplerian_to_cartesian(elements: KeplerianElements, mu: float = MU_EARTH) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert classical orbital elements to a Cartesian state vector (r, v) in
    the same inertial frame the elements are defined in (e.g. ECI/TEME).

    Method (Vallado Alg. 10):
      1. Compute position/velocity in the perifocal (PQW) frame using the
         orbit equation r = p / (1 + e*cos(nu)) and the vis-viva-consistent
         velocity components.
      2. Rotate PQW -> ECI via the 3-1-3 Euler rotation R3(-raan) R1(-i) R3(-argp).

    Returns:
        (r, v): position [km] and velocity [km/s] as 3-vectors.
    """
    a, e, i, raan, argp, nu = (
        elements.a, elements.e, elements.i,
        elements.raan, elements.argp, elements.nu,
    )
    p = a * (1 - e**2)
    r_mag = p / (1 + e * np.cos(nu))

    # Perifocal frame (PQW): P toward perigee, Q 90 deg ahead in orbit plane
    r_pqw = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_argp, sin_argp = np.cos(argp), np.sin(argp)

    # Combined rotation matrix R3(-raan) . R1(-i) . R3(-argp)
    R = np.array([
        [cos_raan * cos_argp - sin_raan * sin_argp * cos_i,
         -cos_raan * sin_argp - sin_raan * cos_argp * cos_i,
         sin_raan * sin_i],
        [sin_raan * cos_argp + cos_raan * sin_argp * cos_i,
         -sin_raan * sin_argp + cos_raan * cos_argp * cos_i,
         -cos_raan * sin_i],
        [sin_argp * sin_i,
         cos_argp * sin_i,
         cos_i],
    ])

    r_eci = R @ r_pqw
    v_eci = R @ v_pqw
    return r_eci, v_eci


def cartesian_to_keplerian(r: np.ndarray, v: np.ndarray, mu: float = MU_EARTH) -> KeplerianElements:
    """
    Convert a Cartesian state vector (r, v) to classical orbital elements.
    Inverse of keplerian_to_cartesian(). Method: Vallado Alg. 9 (rv2coe).

    Degenerate cases (equatorial and/or circular orbits, where raan and/or
    argp are undefined) are NOT specially handled here — for those orbits
    the returned raan/argp are computed from ill-conditioned expressions and
    should be treated as unreliable. This is acceptable for this project
    since real catalog objects are essentially never exactly circular or
    exactly equatorial.
    """
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    h = np.cross(r, v)                      # specific angular momentum
    h_mag = np.linalg.norm(h)

    n = np.cross(np.array([0.0, 0.0, 1.0]), h)  # node vector
    n_mag = np.linalg.norm(n)

    # Eccentricity vector: e_vec = ((v^2 - mu/r) r - (r.v) v) / mu
    e_vec = ((v_mag**2 - mu / r_mag) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)

    energy = v_mag**2 / 2 - mu / r_mag
    a = -mu / (2 * energy)

    i = np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0))

    raan = np.arccos(np.clip(n[0] / n_mag, -1.0, 1.0))
    if n[1] < 0:
        raan = 2 * np.pi - raan

    argp = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * e), -1.0, 1.0))
    if e_vec[2] < 0:
        argp = 2 * np.pi - argp

    nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1.0, 1.0))
    if np.dot(r, v) < 0:
        nu = 2 * np.pi - nu

    return KeplerianElements(a=a, e=e, i=i, raan=raan, argp=argp, nu=nu)
