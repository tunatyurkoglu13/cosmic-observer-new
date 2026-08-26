import numpy as np

from core.collision_monte_carlo import (
    MonteCarloPcResult,
    monte_carlo_collision_probability,
    validate_pc,
)


def test_monte_carlo_matches_analytic_for_moderate_pc_scenario():
    """
    A scenario with Pc in a range Monte Carlo can estimate precisely with
    a modest sample count (not a rare-event tail probability, which would
    need an impractically large N for a fast test) — the two independent
    methods should agree within a handful of standard errors.
    """
    miss = np.array([0.02, 0.01, 0.0])
    v_rel = np.array([7.5, 0.0, 0.0])
    covariance = np.eye(3) * 0.01
    hbr = 0.05

    report = validate_pc(miss, v_rel, covariance, hbr, n_samples=300_000, random_state=42)

    assert report.monte_carlo.pc_estimate > 0.01  # not a rare-event regime for this test
    assert report.difference_in_standard_errors < 5.0  # ~5-sigma agreement bound


def test_monte_carlo_direct_hit_near_one():
    miss = np.zeros(3)
    v_rel = np.array([7.5, 0.0, 0.0])
    covariance = np.eye(3) * 1e-6
    result = monte_carlo_collision_probability(miss, v_rel, covariance, hard_body_radius_km=0.1, n_samples=50_000, random_state=1)
    assert result.pc_estimate > 0.99


def test_monte_carlo_large_miss_near_zero():
    miss = np.array([0.0, 100.0, 0.0])
    v_rel = np.array([7.5, 0.0, 0.0])
    covariance = np.eye(3) * 0.01
    result = monte_carlo_collision_probability(miss, v_rel, covariance, hard_body_radius_km=0.02, n_samples=50_000, random_state=2)
    assert result.pc_estimate == 0.0
    assert result.n_hits == 0


def test_standard_error_shrinks_with_more_samples():
    """The Monte Carlo standard error must shrink as 1/sqrt(N) — a basic sanity check on the estimator itself."""
    miss = np.array([0.02, 0.0, 0.0])
    v_rel = np.array([7.5, 0.0, 0.0])
    covariance = np.eye(3) * 0.01
    hbr = 0.05

    small_n = monte_carlo_collision_probability(miss, v_rel, covariance, hbr, n_samples=1_000, random_state=7)
    large_n = monte_carlo_collision_probability(miss, v_rel, covariance, hbr, n_samples=100_000, random_state=7)

    assert large_n.standard_error < small_n.standard_error
    # 100x more samples -> ~10x smaller standard error (1/sqrt(100)=0.1); allow generous slack.
    assert large_n.standard_error < small_n.standard_error * 0.3


def test_confidence_interval_contains_analytic_value_in_typical_case():
    miss = np.array([0.015, 0.0, 0.0])
    v_rel = np.array([7.5, 0.0, 0.0])
    covariance = np.eye(3) * 0.008
    hbr = 0.04

    report = validate_pc(miss, v_rel, covariance, hbr, n_samples=300_000, random_state=99)
    lo, hi = report.monte_carlo.confidence_interval_95
    assert lo <= report.analytic_pc <= hi


def test_monte_carlo_result_hits_consistent_with_estimate():
    miss = np.array([0.02, 0.0, 0.0])
    v_rel = np.array([7.5, 0.0, 0.0])
    covariance = np.eye(3) * 0.01
    result = monte_carlo_collision_probability(miss, v_rel, covariance, 0.05, n_samples=10_000, random_state=3)
    assert np.isclose(result.pc_estimate, result.n_hits / result.n_samples)
