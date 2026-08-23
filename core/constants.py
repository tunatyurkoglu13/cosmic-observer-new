"""
Physical and geodetic constants shared across the orbital mechanics core.

Values follow WGS-72 (the reference ellipsoid SGP4 itself was built around)
so that propagator.py stays numerically consistent with the sgp4 library.
Source: Vallado, "Fundamentals of Astrodynamics and Applications" (4th Ed),
Appendix B; Hoots & Roehrich, Spacetrack Report #3.
"""

import numpy as np

# Earth gravitational parameter, mu = G * M_earth  [km^3 / s^2]
MU_EARTH = 398600.4418

# Earth equatorial radius (WGS-72)  [km]
R_EARTH = 6378.137

# Earth flattening (WGS-72)
EARTH_FLATTENING = 1 / 298.26

# J2 zonal harmonic (oblateness term), dimensionless
J2 = 1.08262668e-3

# Earth's sidereal rotation rate  [rad / s]
OMEGA_EARTH = 7.2921159e-5

# Speed of light  [km / s]
SPEED_OF_LIGHT = 299792.458

# Julian date of the J2000.0 epoch
JD_J2000 = 2451545.0
