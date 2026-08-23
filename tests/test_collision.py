import numpy as np

from core.collision import (
    altitude_bands_overlap,
    b_plane_frame,
    mahalanobis_distance,
    probability_of_collision,
)


def test_altitude_bands_overlap_true_for_leo_pair():
    assert altitude_bands_overlap(perigee1=400, apogee1=420, perigee2=410, apogee2=430)


def test_altitude_bands_overlap_false_for_leo_vs_geo():
    assert not altitude_bands_overlap(perigee1=400, apogee1=420, perigee2=35786, apogee2=35786, margin_km=50)


def test_mahalanobis_distance_zero_miss_is_zero():
    C = np.eye(3) * 0.01  # km^2
    miss = np.zeros(3)
    assert mahalanobis_distance(miss, C) == 0.0


def test_mahalanobis_distance_scales_with_miss_magnitude():
    C = np.eye(3) * 0.01
    small_miss = np.array([0.1, 0.0, 0.0])
    large_miss = np.array([1.0, 0.0, 0.0])
    d_small = mahalanobis_distance(small_miss, C)
    d_large = mahalanobis_distance(large_miss, C)
    assert d_large > d_small


def test_b_plane_frame_orthogonal_to_velocity_and_each_other():
    r_rel = np.array([1.0, 2.0, 3.0])
    v_rel = np.array([7.5, 0.0, 0.0])
    e_x, e_y = b_plane_frame(r_rel, v_rel)

    v_hat = v_rel / np.linalg.norm(v_rel)
    assert np.isclose(np.dot(e_x, v_hat), 0.0, atol=1e-9)
    assert np.isclose(np.dot(e_y, v_hat), 0.0, atol=1e-9)
    assert np.isclose(np.dot(e_x, e_y), 0.0, atol=1e-9)
    assert np.isclose(np.linalg.norm(e_x), 1.0, atol=1e-9)
    assert np.isclose(np.linalg.norm(e_y), 1.0, atol=1e-9)


def test_probability_of_collision_direct_hit_near_one():
    # Zero miss distance, tiny covariance (tight knowledge), generous HBR -> Pc should approach 1.
    miss = np.zeros(3)
    v_rel = np.array([7.5, 0.0, 0.0])
    cov = np.eye(3) * 1e-6  # km^2, i.e. ~mm-scale position uncertainty
    pc = probability_of_collision(miss, v_rel, cov, hard_body_radius_km=0.1)
    assert pc > 0.99


def test_probability_of_collision_large_miss_near_zero():
    # Miss distance transverse to v_rel (i.e. an actual B-plane offset) far
    # outside a small hard-body radius, with tight covariance -> Pc ~ 0.
    # (A miss vector purely *along* v_rel projects to ~0 in the B-plane,
    # since only the transverse offset matters for a close approach.)
    miss = np.array([0.0, 100.0, 0.0])  # 100 km transverse miss
    v_rel = np.array([7.5, 0.0, 0.0])
    cov = np.eye(3) * 0.01
    pc = probability_of_collision(miss, v_rel, cov, hard_body_radius_km=0.02)
    assert pc < 1e-10


def test_probability_of_collision_bounded_in_unit_interval():
    miss = np.array([0.05, 0.02, 0.0])
    v_rel = np.array([6.0, 1.0, 0.0])
    cov = np.array([[0.02, 0.001, 0.0], [0.001, 0.015, 0.0], [0.0, 0.0, 0.03]])
    pc = probability_of_collision(miss, v_rel, cov, hard_body_radius_km=0.02)
    assert 0.0 <= pc <= 1.0
