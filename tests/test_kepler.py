import numpy as np

from core.constants import MU_EARTH
from core.kepler import (
    KeplerianElements,
    cartesian_to_keplerian,
    keplerian_to_cartesian,
    mean_anomaly_from_eccentric,
    mean_motion,
    orbital_period,
    solve_kepler_equation,
    true_anomaly_from_eccentric,
)


def test_kepler_equation_circular_orbit():
    # For e=0, E should equal M exactly.
    M = 1.234
    E = solve_kepler_equation(M, e=0.0)
    assert np.isclose(E, M, atol=1e-9)


def test_kepler_equation_roundtrip():
    for M in np.linspace(0.1, 2 * np.pi - 0.1, 10):
        for e in [0.01, 0.1, 0.5, 0.85]:
            E = solve_kepler_equation(M, e)
            M_recovered = mean_anomaly_from_eccentric(E, e)
            assert np.isclose(M_recovered, M, atol=1e-9)


def test_true_anomaly_roundtrip():
    from core.kepler import eccentric_anomaly_from_true

    for nu in np.linspace(0.1, 2 * np.pi - 0.1, 10):
        for e in [0.01, 0.3, 0.7]:
            E = eccentric_anomaly_from_true(nu, e)
            nu_recovered = true_anomaly_from_eccentric(E, e)
            assert np.isclose(nu_recovered, nu, atol=1e-9)


def test_mean_motion_period_consistency():
    a = 7000.0  # km, LEO-ish
    n = mean_motion(a)
    T = orbital_period(a)
    assert np.isclose(n * T, 2 * np.pi, atol=1e-9)


def test_keplerian_cartesian_roundtrip():
    elements = KeplerianElements(
        a=7000.0, e=0.01, i=np.radians(51.6),
        raan=np.radians(120.0), argp=np.radians(45.0), nu=np.radians(30.0),
    )
    r, v = keplerian_to_cartesian(elements)
    recovered = cartesian_to_keplerian(r, v)

    assert np.isclose(recovered.a, elements.a, rtol=1e-6)
    assert np.isclose(recovered.e, elements.e, atol=1e-6)
    assert np.isclose(recovered.i, elements.i, atol=1e-6)
    assert np.isclose(recovered.raan, elements.raan, atol=1e-6)
    assert np.isclose(recovered.argp, elements.argp, atol=1e-6)
    assert np.isclose(recovered.nu, elements.nu, atol=1e-6)


def test_vis_viva_speed_at_perigee():
    # At perigee of a circular orbit, speed should equal sqrt(mu/a).
    a = 7000.0
    elements = KeplerianElements(a=a, e=0.0, i=0.0, raan=0.0, argp=0.0, nu=0.0)
    r, v = keplerian_to_cartesian(elements)
    expected_speed = np.sqrt(MU_EARTH / a)
    assert np.isclose(np.linalg.norm(v), expected_speed, rtol=1e-9)
    assert np.isclose(np.linalg.norm(r), a, rtol=1e-9)
