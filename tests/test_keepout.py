import numpy as np

from stm.keepout import (
    DEFAULT_ACTIVE_SATELLITE_KEEPOUT,
    ISS_CORRIDOR,
    KeepOutZone,
    check_keepout_violation,
    min_distance_to_zone_boundary,
    relative_position_rsw,
    rsw_frame,
)


def test_rsw_frame_is_orthonormal():
    r = np.array([6800.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.5])
    R_hat, S_hat, W_hat = rsw_frame(r, v)

    for vec in (R_hat, S_hat, W_hat):
        assert np.isclose(np.linalg.norm(vec), 1.0)
    assert np.isclose(np.dot(R_hat, S_hat), 0.0, atol=1e-9)
    assert np.isclose(np.dot(R_hat, W_hat), 0.0, atol=1e-9)
    assert np.isclose(np.dot(S_hat, W_hat), 0.0, atol=1e-9)


def test_relative_position_rsw_pure_along_track_offset():
    r_chief = np.array([6800.0, 0.0, 0.0])
    v_chief = np.array([0.0, 7.5, 0.0])  # circular, orbit in xy-plane
    # A deputy purely ahead along-track (+y direction from chief) with no radial/cross-track offset.
    r_deputy = r_chief + np.array([0.0, 2.0, 0.0])

    rel = relative_position_rsw(r_chief, v_chief, r_deputy)
    assert np.isclose(rel[0], 0.0, atol=1e-9)   # radial
    assert np.isclose(abs(rel[1]), 2.0, atol=1e-9)  # along-track
    assert np.isclose(rel[2], 0.0, atol=1e-9)   # cross-track


def test_sphere_zone_violation_inside_and_outside():
    r_chief = np.array([6800.0, 0.0, 0.0])
    v_chief = np.array([0.0, 7.5, 0.0])
    zone = KeepOutZone(name="test-sphere", shape="sphere", radius_km=5.0)

    r_inside = r_chief + np.array([1.0, 0.0, 0.0])
    r_outside = r_chief + np.array([10.0, 0.0, 0.0])

    assert check_keepout_violation(r_chief, v_chief, r_inside, zone)
    assert not check_keepout_violation(r_chief, v_chief, r_outside, zone)


def test_box_zone_iss_corridor_violation():
    r_chief = np.array([6800.0, 0.0, 0.0])
    v_chief = np.array([0.0, 7.5, 0.0])

    # Well within the 25 km / 25 km / 2 km half-dims of ISS_CORRIDOR.
    r_inside = r_chief + np.array([1.0, 3.0, 0.5])
    # Cross-track offset (5 km) exceeds the 2 km half-dim -> outside the box.
    r_outside = r_chief + np.array([0.0, 0.0, 5.0])

    assert check_keepout_violation(r_chief, v_chief, r_inside, ISS_CORRIDOR)
    assert not check_keepout_violation(r_chief, v_chief, r_outside, ISS_CORRIDOR)


def test_min_distance_to_zone_boundary_sign_conventions():
    r_chief = np.array([6800.0, 0.0, 0.0])
    v_chief = np.array([0.0, 7.5, 0.0])
    zone = KeepOutZone(name="test-sphere", shape="sphere", radius_km=5.0)

    r_inside = r_chief + np.array([1.0, 0.0, 0.0])
    r_outside = r_chief + np.array([10.0, 0.0, 0.0])

    assert min_distance_to_zone_boundary(r_chief, v_chief, r_inside, zone) < 0
    assert min_distance_to_zone_boundary(r_chief, v_chief, r_outside, zone) > 0


def test_default_active_satellite_keepout_is_sphere():
    assert DEFAULT_ACTIVE_SATELLITE_KEEPOUT.shape == "sphere"
    assert DEFAULT_ACTIVE_SATELLITE_KEEPOUT.radius_km == 5.0
