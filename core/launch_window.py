"""
core/launch_window.py — Launch site database, launch azimuth, and launch
window optimization.

This module answers three related questions:
  1. From a given launch site, which orbital inclinations are directly
     reachable, and at what compass azimuth? (launch_azimuth /
     achievable_inclination_range)
  2. Given a target orbital plane (inclination + RAAN, or a sun-synchronous
     constraint), *when* does Earth's rotation bring the launch site under
     that plane? (next_launch_window_time / sun_synchronous_inclination)
  3. What does it cost (delta-v) to get from the initial insertion orbit
     to the final mission orbit? (hohmann_transfer)

References: Vallado, "Fundamentals of Astrodynamics and Applications"
(4th Ed), Ch. 6 (orbital maneuvers) & Ch. 11 (launch, "site restrictions
on launch azimuth"); Wertz, "Space Mission Analysis and Design" (SMAD).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from sgp4.api import jday

from core.constants import J2, MU_EARTH, OMEGA_EARTH, R_EARTH
from core.kepler import mean_motion
from core.propagator import gmst_from_jd

# Earth's mean angular rate around the Sun (deg/day) — the RAAN drift
# rate a sun-synchronous orbit's inclination must produce so its orbital
# plane keeps pace with the Sun (fixed Local Time of Ascending Node).
SUN_SYNCHRONOUS_RAAN_DRIFT_DEG_PER_DAY = 360.0 / 365.2422


@dataclass
class LaunchSite:
    """A launch site/pad, with its allowable launch-azimuth corridor."""

    name: str
    country: str
    lat_deg: float
    lon_deg: float
    azimuth_min_deg: float
    azimuth_max_deg: float


# Preset launch sites (project spec table).
LAUNCH_SITES: dict[str, LaunchSite] = {
    "cape_canaveral": LaunchSite("Cape Canaveral SLC-40", "USA", 28.5621, -80.5773, 35.0, 120.0),
    "kennedy": LaunchSite("Kennedy LC-39A", "USA", 28.6084, -80.6043, 35.0, 120.0),
    "vandenberg": LaunchSite("Vandenberg SLC-4E", "USA", 34.6328, -120.6105, 147.0, 201.0),
    "baikonur": LaunchSite("Baikonur Site 31/6", "Kazakhstan/Russia", 45.9960, 63.5640, 48.0, 135.0),
    "plesetsk": LaunchSite("Plesetsk Site 43", "Russia", 62.9273, 40.4500, 30.0, 180.0),
    "kourou": LaunchSite("Kourou ELA-3", "ESA/France", 5.2360, -52.7687, -105.0, 105.0),
    "wenchang": LaunchSite("Wenchang LC-2", "China", 19.6180, 110.9510, 80.0, 180.0),
    "jiuquan": LaunchSite("Jiuquan SLS-2", "China", 40.9606, 100.2980, 90.0, 180.0),
    "tanegashima": LaunchSite("Tanegashima LP-1", "Japan", 30.4000, 130.9700, 80.0, 180.0),
    "satish_dhawan": LaunchSite("Satish Dhawan SLP", "India", 13.7200, 80.2300, 90.0, 180.0),
}


def achievable_inclination_range(site_lat_deg: float) -> tuple[float, float]:
    """
    Minimum/maximum orbital inclination directly reachable (no dogleg
    maneuver) from a site at the given latitude:

        i_min = |latitude|
        i_max = 180 - |latitude|

    Derived from cos(i) = cos(phi)*sin(Az): since |sin(Az)| <= 1, we need
    |cos(i)| <= cos(phi), i.e. i in [|phi|, 180-|phi|]. Physically: you
    cannot launch directly into an orbital plane whose inclination is
    lower than your launch latitude — the rocket would need a plane-change
    "dogleg" maneuver (which costs extra delta-v) to reach it.
    """
    i_min = abs(site_lat_deg)
    return i_min, 180.0 - i_min


def launch_azimuth(target_inclination_deg: float, site_lat_deg: float, direction: str = "northeast") -> float:
    """
    Launch azimuth for a direct ascent to a target inclination:

        cos(i) = cos(phi) * sin(Az)  =>  Az = arcsin(cos(i) / cos(phi))

    Args:
        target_inclination_deg: desired orbital inclination [deg].
        site_lat_deg: launch site latitude [deg].
        direction: "northeast" for the ascending-node solution (Az in
            [0, 90] for prograde low-inclination orbits), "southeast" for
            the complementary solution (180 - Az_northeast) — both put
            the vehicle on the same inclination but launch toward
            different initial headings / with different downrange
            geography, which is why a site's azimuth corridor
            (LaunchSite.azimuth_min/max_deg) usually only permits one of
            the two.

    Returns:
        Azimuth in degrees, measured clockwise from true north, in [0, 360).

    Raises:
        ValueError: if target_inclination_deg is not directly reachable
            from this latitude (see achievable_inclination_range()).
    """
    i_min, i_max = achievable_inclination_range(site_lat_deg)
    if not (i_min - 1e-9 <= target_inclination_deg <= i_max + 1e-9):
        raise ValueError(
            f"Inclination {target_inclination_deg} deg is not directly reachable from "
            f"latitude {site_lat_deg} deg (achievable range: [{i_min:.2f}, {i_max:.2f}] deg); "
            "a dogleg (plane-change) maneuver would be required."
        )

    cos_phi = np.cos(np.radians(site_lat_deg))
    cos_i = np.cos(np.radians(target_inclination_deg))
    sin_az = np.clip(cos_i / cos_phi, -1.0, 1.0)
    az_northeast = np.degrees(np.arcsin(sin_az))

    if direction == "northeast":
        az = az_northeast
    elif direction == "southeast":
        az = 180.0 - az_northeast
    else:
        raise ValueError("direction must be 'northeast' or 'southeast'")

    return az % 360.0


def _datetime_to_jd(dt: datetime) -> float:
    """UTC datetime -> full Julian date, via the same sgp4 jday() helper used in propagator.py."""
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)
    return jd + fr


def ascending_node_longitude_deg(raan_deg: float, epoch: datetime) -> float:
    """
    Earth-fixed (ECEF) longitude currently under the target orbital
    plane's ascending node, at a given epoch:

        lon_node = RAAN - GMST(epoch)   (mod 360)

    RAAN is measured from the inertial vernal equinox direction; GMST is
    the angle from that same direction to the Greenwich meridian, so
    their difference gives the node's Earth-fixed longitude.
    """
    jd = _datetime_to_jd(epoch)
    gmst_deg = np.degrees(gmst_from_jd(jd))
    return (raan_deg - gmst_deg) % 360.0


def next_launch_window_time(
    site_lon_deg: float,
    raan_deg: float,
    reference_epoch: datetime,
    orbit_count: int = 0,
) -> datetime:
    """
    Time at which Earth's rotation next brings the launch site's
    longitude under the target orbital plane's ascending node:

        t_window = Delta_lambda / omega_E + n * (2*pi / omega_E)

    where Delta_lambda is the (eastward) angular distance the site must
    rotate through to reach the node's current Earth-fixed longitude, and
    n = orbit_count selects the same alignment on a later Earth rotation
    (roughly one solar day later — the node itself also slowly regresses
    due to J2, which is why this is treated as an approximate window
    rather than an exact one many days out).

    Args:
        site_lon_deg: launch site longitude [deg].
        raan_deg: target orbital plane's RAAN [deg].
        reference_epoch: UTC datetime to search forward from.
        orbit_count: which subsequent daily alignment to return (0 = the
            very next one).

    Returns:
        UTC datetime of the launch window.
    """
    node_lon_deg = ascending_node_longitude_deg(raan_deg, reference_epoch)
    delta_lambda_deg = (node_lon_deg - site_lon_deg) % 360.0

    t_wait_s = np.radians(delta_lambda_deg) / OMEGA_EARTH + orbit_count * (2 * np.pi / OMEGA_EARTH)
    return reference_epoch + timedelta(seconds=float(t_wait_s))


def sun_synchronous_inclination(altitude_km: float, eccentricity: float = 0.0) -> float:
    """
    Inclination required for a sun-synchronous orbit (SSO) at a given
    altitude: the inclination whose J2-induced RAAN drift exactly matches
    Earth's ~0.9856 deg/day motion around the Sun, so the orbital plane's
    orientation relative to the Sun (and hence Local Time of Ascending
    Node) stays fixed.

    Inverts the RAAN secular-drift formula (core.propagator.j2_secular_rates):

        Omega_dot = -(3/2) * n * J2 * (R_E/p)^2 * cos(i)
        => cos(i) = Omega_dot_target / (-(3/2) * n * J2 * (R_E/p)^2)

    Since Omega_dot_target is positive (eastward, prograde node
    regression matching the Sun) and the coefficient is negative, cos(i)
    comes out negative — meaning SSOs are always retrograde (i > 90 deg),
    consistent with the well-known ~98 deg inclination used by real
    Earth-observation SSO missions.

    Args:
        altitude_km: circular-equivalent altitude [km].
        eccentricity: orbital eccentricity (0.0 for circular).

    Returns:
        Required inclination [deg].

    Raises:
        ValueError: if no sun-synchronous solution exists at this altitude
            (the required |cos(i)| would exceed 1).
    """
    a = R_EARTH + altitude_km
    n = mean_motion(a, mu=MU_EARTH)
    p = a * (1 - eccentricity**2)

    raan_dot_target = np.radians(SUN_SYNCHRONOUS_RAAN_DRIFT_DEG_PER_DAY) / 86400.0  # rad/s
    coefficient = -1.5 * n * J2 * (R_EARTH / p) ** 2

    cos_i = raan_dot_target / coefficient
    if abs(cos_i) > 1.0:
        raise ValueError(f"No sun-synchronous solution exists at altitude {altitude_km} km")

    return float(np.degrees(np.arccos(cos_i)))


def phasing_wait_time(omega_target_rad_s: float, omega_chaser_rad_s: float, k: int = 1) -> float:
    """
    Rendezvous phasing wait time: how long the chaser (launch vehicle /
    newly inserted spacecraft) must wait in its parking orbit before the
    relative geometry with a target (e.g. a space station) repeats,
    allowing a rendezvous transfer to begin:

        Delta_T_phase = 2*pi*k / (omega_target - omega_chaser),  k in Z

    Args:
        omega_target_rad_s: target's angular rate (mean motion) [rad/s].
        omega_chaser_rad_s: chaser's angular rate (mean motion) [rad/s].
        k: which repeat cycle to use; the sign/magnitude of k should be
            chosen so the returned wait time is positive (e.g. if the
            chaser is faster than the target, omega_chaser > omega_target
            and you'll need a negative k, or equivalently swap which body
            you call "target").

    Returns:
        Wait time [s] (may be negative if k has the wrong sign for this
        configuration — callers should check the sign and adjust k).

    Raises:
        ValueError: if the two angular rates are equal (no relative
            drift, so no future alignment beyond the current one).
    """
    diff = omega_target_rad_s - omega_chaser_rad_s
    if np.isclose(diff, 0.0):
        raise ValueError("Target and chaser have equal angular rates — no phasing drift occurs")
    return 2 * np.pi * k / diff


@dataclass
class HohmannTransfer:
    """Delta-v budget and duration for a two-impulse Hohmann transfer between circular orbits."""

    delta_v1_km_s: float
    delta_v2_km_s: float
    delta_v_total_km_s: float
    transfer_time_s: float


def hohmann_transfer(r1_km: float, r2_km: float, mu: float = MU_EARTH) -> HohmannTransfer:
    """
    Two-impulse Hohmann transfer between two circular orbits of radii r1
    (initial) and r2 (final):

        dv1 = sqrt(mu/r1) * (sqrt(2*r2/(r1+r2)) - 1)
        dv2 = sqrt(mu/r2) * (1 - sqrt(2*r1/(r1+r2)))
        transfer_time = pi * sqrt(a_transfer^3 / mu),  a_transfer = (r1+r2)/2

    This is the minimum-delta-v two-impulse transfer between coplanar
    circular orbits (Hohmann, 1925) — used here for the launch-insertion
    to final-mission-orbit raise (e.g. LEO parking orbit -> GEO transfer
    orbit -> GEO).

    Args:
        r1_km, r2_km: initial/final orbital radii (from Earth's center,
            i.e. altitude + R_EARTH) [km].
        mu: gravitational parameter [km^3/s^2].

    Returns:
        HohmannTransfer with both burn magnitudes, their sum, and the
        one-way transfer duration.
    """
    dv1 = np.sqrt(mu / r1_km) * (np.sqrt(2 * r2_km / (r1_km + r2_km)) - 1)
    dv2 = np.sqrt(mu / r2_km) * (1 - np.sqrt(2 * r1_km / (r1_km + r2_km)))

    a_transfer = (r1_km + r2_km) / 2
    transfer_time = np.pi * np.sqrt(a_transfer**3 / mu)

    return HohmannTransfer(
        delta_v1_km_s=float(dv1),
        delta_v2_km_s=float(dv2),
        delta_v_total_km_s=float(abs(dv1) + abs(dv2)),
        transfer_time_s=float(transfer_time),
    )


@dataclass
class LaunchTarget:
    """Desired mission orbit for launch-window planning."""

    altitude_km: float
    inclination_deg: float
    raan_deg: float | None = None       # None => no specific RAAN constraint (e.g. non-rendezvous mission)
    sun_synchronous: bool = False


@dataclass
class LaunchWindowResult:
    """One candidate launch opportunity."""

    launch_time: datetime
    azimuth_deg: float
    feasible: bool
    notes: str = ""


class LaunchWindowCalculator:
    """Ties together azimuth feasibility and Earth-rotation timing for a given launch site."""

    def __init__(self, site: LaunchSite):
        self.site = site

    def compute_azimuth(self, target_inclination_deg: float, direction: str = "northeast") -> float:
        """Azimuth for a direct ascent to the target inclination from this site."""
        return launch_azimuth(target_inclination_deg, self.site.lat_deg, direction=direction)

    def find_windows(
        self,
        target: LaunchTarget,
        start: datetime,
        end: datetime,
        direction: str = "northeast",
    ) -> list[LaunchWindowResult]:
        """
        Find candidate launch windows for a target orbit within [start, end].

        If target.sun_synchronous is True, the required inclination is
        derived from target.altitude_km (overriding target.inclination_deg)
        via sun_synchronous_inclination().

        If target.raan_deg is None, no specific Earth-rotation alignment
        is required (any time works, subject only to azimuth feasibility)
        — a single window is returned at `start` with a note to that effect.

        Otherwise, one window per site-longitude/node alignment (roughly
        daily) is returned across the search range.
        """
        inclination_deg = (
            sun_synchronous_inclination(target.altitude_km) if target.sun_synchronous else target.inclination_deg
        )

        try:
            azimuth = self.compute_azimuth(inclination_deg, direction=direction)
            feasible = self.site.azimuth_min_deg <= azimuth <= self.site.azimuth_max_deg
            notes = "" if feasible else (
                f"Azimuth {azimuth:.1f} deg falls outside site corridor "
                f"[{self.site.azimuth_min_deg}, {self.site.azimuth_max_deg}] deg"
            )
        except ValueError as e:
            return [LaunchWindowResult(launch_time=start, azimuth_deg=float("nan"), feasible=False, notes=str(e))]

        if target.raan_deg is None:
            return [
                LaunchWindowResult(
                    launch_time=start, azimuth_deg=azimuth, feasible=feasible,
                    notes=notes or "No RAAN constraint specified — any time in range is a valid window.",
                )
            ]

        windows = []
        orbit_count = 0
        while True:
            candidate = next_launch_window_time(self.site.lon_deg, target.raan_deg, start, orbit_count=orbit_count)
            if candidate > end:
                break
            windows.append(
                LaunchWindowResult(launch_time=candidate, azimuth_deg=azimuth, feasible=feasible, notes=notes)
            )
            orbit_count += 1

        return windows
