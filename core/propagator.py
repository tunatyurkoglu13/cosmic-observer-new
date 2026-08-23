"""
core/propagator.py — SGP4 propagation and coordinate-frame conversions.

SGP4 (Simplified General Perturbations 4) is the standard analytic
propagator for near-Earth catalog objects (period < 225 min). It bakes in
J2/J3/J4 zonal harmonics and a simplified atmospheric drag model tuned
against the same data NORAD uses to produce TLEs — so TLEs and SGP4 must
always be used together (propagating a TLE with a numerical integrator
instead of SGP4 introduces systematic error, since the TLE's mean elements
already have SGP4's own perturbation model baked out of them).

We use the `sgp4` package's C++-backed implementation (Vallado's 2006
"AIAA" revision), which is the reference implementation used by
Space-Track and CelesTrak.

Frame chain implemented here:
    TLE --SGP4-->  TEME (True Equator, Mean Equinox), km & km/s
         --GMST--> ECEF (Earth-Centered, Earth-Fixed)
         --geodetic--> (lat, lon, alt) on WGS-72

References: Vallado & Crawford, "SGP4 Orbit Determination" (AIAA 2006-6753);
Vallado, "Fundamentals of Astrodynamics and Applications" (4th Ed), Ch. 3.
"""

from dataclasses import dataclass

import numpy as np
from sgp4.api import Satrec, jday

from core.constants import EARTH_FLATTENING, OMEGA_EARTH, R_EARTH


@dataclass
class StateVector:
    """A single propagated state: position/velocity plus derived geodetic coords."""

    epoch_jd: float          # Julian date (UTC) of this state
    r_teme: np.ndarray       # position in TEME frame [km]
    v_teme: np.ndarray       # velocity in TEME frame [km/s]
    r_ecef: np.ndarray       # position in ECEF frame [km]
    lat_deg: float           # geodetic latitude [deg]
    lon_deg: float           # geodetic longitude [deg]
    alt_km: float            # altitude above WGS-72 ellipsoid [km]


class Propagator:
    """Wraps a single TLE and propagates it to arbitrary epochs via SGP4."""

    def __init__(self, line1: str, line2: str, name: str | None = None):
        self.name = name
        self.line1 = line1
        self.line2 = line2
        self._sat = Satrec.twoline2rv(line1, line2)

    def propagate(self, dt) -> StateVector:
        """
        Propagate to a Python datetime `dt` (must be timezone-aware UTC, or
        naive and assumed to already be UTC).

        Returns a StateVector with TEME, ECEF, and geodetic representations.
        Raises RuntimeError if SGP4 reports an error code (e.g. decayed orbit).
        """
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)
        error_code, r_teme, v_teme = self._sat.sgp4(jd, fr)
        if error_code != 0:
            raise RuntimeError(f"SGP4 error code {error_code} for {self.name or 'satellite'} at JD {jd + fr}")

        r_teme = np.array(r_teme)
        v_teme = np.array(v_teme)

        epoch_jd = jd + fr
        gmst = gmst_from_jd(epoch_jd)
        r_ecef = teme_to_ecef(r_teme, gmst)
        lat_deg, lon_deg, alt_km = ecef_to_geodetic(r_ecef)

        return StateVector(
            epoch_jd=epoch_jd,
            r_teme=r_teme,
            v_teme=v_teme,
            r_ecef=r_ecef,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            alt_km=alt_km,
        )

    def propagate_batch(self, datetimes) -> list[StateVector]:
        """Convenience wrapper: propagate to a sequence of datetimes."""
        return [self.propagate(dt) for dt in datetimes]


def gmst_from_jd(jd: float) -> float:
    """
    Greenwich Mean Sidereal Time, IAU-82 approximation.

        theta_GMST = 280.46061837 + 360.98564736629 * (JD - 2451545.0)  [deg]

    This is the low-precision formula (good to ~0.1 arcsec/century, ample
    for TLE-accuracy work) — not the full IAU-2006 series. Result is
    wrapped to [0, 360) degrees and returned in radians.

    Args:
        jd: full Julian date (UT1 ~ UTC for this precision level).

    Returns:
        GMST angle in radians.
    """
    T = jd - 2451545.0
    gmst_deg = (280.46061837 + 360.98564736629 * T) % 360.0
    return np.radians(gmst_deg)


def teme_to_ecef(r_teme: np.ndarray, gmst_rad: float) -> np.ndarray:
    """
    Rotate a TEME position vector into ECEF by the (negative) GMST angle
    about the Z axis. This ignores polar motion and the small TEME/PEF
    offset (sub-arcsecond effects) which are negligible at TLE accuracy
    (TLE position error is typically ~1 km at epoch, growing with time).

        [x_ecef]   [ cos(theta)  sin(theta)  0] [x_teme]
        [y_ecef] = [-sin(theta)  cos(theta)  0] [y_teme]
        [z_ecef]   [     0            0      1] [z_teme]
    """
    c, s = np.cos(gmst_rad), np.sin(gmst_rad)
    R = np.array([
        [c, s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return R @ r_teme


def ecef_to_geodetic(r_ecef: np.ndarray) -> tuple[float, float, float]:
    """
    Convert an ECEF position to geodetic (lat, lon, alt) on the WGS-72
    ellipsoid, using Bowring's iterative method (converges in ~2-3
    iterations for Earth's small flattening).

    Returns:
        (lat_deg, lon_deg, alt_km)
    """
    x, y, z = r_ecef
    f = EARTH_FLATTENING
    a = R_EARTH
    e2 = 2 * f - f**2  # first eccentricity squared

    lon = np.arctan2(y, x)

    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))  # initial guess

    for _ in range(5):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + alt)))

    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    alt = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), alt


def j2_secular_rates(a: float, e: float, i: float, mu: float, r_earth: float, j2: float) -> tuple[float, float]:
    """
    J2 secular drift rates for RAAN and argument of perigee.

    RAAN drift:
        d(Omega)/dt = -(3/2) * n * J2 * (R_E / p)^2 * cos(i)

    Argument-of-perigee drift:
        d(omega)/dt = (3/4) * n * J2 * (R_E / p)^2 * (5*cos^2(i) - 1)

    where n = sqrt(mu/a^3) is the mean motion and p = a*(1-e^2) is the
    semi-latus rectum. These are *secular* (average) rates from first-order
    perturbation theory (Vallado Ch. 9) — SGP4 already includes an
    equivalent correction internally; this function exists for standalone
    analysis (e.g. explaining why a sun-synchronous orbit's RAAN tracks
    the Sun) rather than for use inside the SGP4 propagation path.

    Returns:
        (raan_dot, argp_dot) in rad/s.
    """
    n = np.sqrt(mu / a**3)
    p = a * (1 - e**2)
    factor = n * j2 * (r_earth / p) ** 2

    raan_dot = -1.5 * factor * np.cos(i)
    argp_dot = 0.75 * factor * (5 * np.cos(i) ** 2 - 1)
    return raan_dot, argp_dot
