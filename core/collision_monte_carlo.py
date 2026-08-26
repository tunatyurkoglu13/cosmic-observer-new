"""
core/collision_monte_carlo.py — Monte Carlo cross-validation of the
analytic collision-probability (Pc) integral.

core.collision.probability_of_collision() computes Pc via numerical
quadrature (scipy.integrate.dblquad) of the B-plane 2D Gaussian density
over the combined hard-body disk. This module answers the question every
computational scientist should ask of an integral like that: does a
completely independent method — brute-force random sampling — agree?

Method: draw N samples from the same 2D Gaussian (mean = the B-plane
miss vector, covariance = the same projected combined covariance the
analytic integral uses), and count what fraction land inside the
hard-body disk. This is a direct restatement of the same probability
as a sampling problem rather than a quadrature problem, so agreement
between the two independently-computed values is a genuine check on the
quadrature code's correctness — not a test of the B-plane linearization
modeling assumption itself (encounter geometry as an instantaneous 2D
slice — see core.collision's docstring), which is a separate, already-
documented choice this module doesn't re-litigate.

The Monte Carlo estimator's own uncertainty is standard for a binomial
proportion: for N samples with true probability p, the standard error is
sqrt(p(1-p)/N) — so this module also reports that, letting a caller
judge whether the two methods agree within the sampling noise expected
at a given N (this is the "count photons, compute sqrt(N) and see if
your systematic matches" habit computational physics runs on).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.collision import b_plane_frame, probability_of_collision


@dataclass
class MonteCarloPcResult:
    """A Monte Carlo Pc estimate alongside its own sampling uncertainty."""

    pc_estimate: float
    standard_error: float
    n_samples: int
    n_hits: int

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """Approximate 95% CI (normal approximation — valid when n_hits isn't tiny; see module docstring)."""
        margin = 1.96 * self.standard_error
        return max(0.0, self.pc_estimate - margin), min(1.0, self.pc_estimate + margin)


def monte_carlo_collision_probability(
    miss_vector_eci: np.ndarray,
    v_rel_eci: np.ndarray,
    combined_covariance_eci: np.ndarray,
    hard_body_radius_km: float,
    n_samples: int = 200_000,
    random_state: int | None = None,
) -> MonteCarloPcResult:
    """
    Estimate Pc by direct sampling in the B-plane, using the exact same
    projection (core.collision.b_plane_frame) the analytic integral uses,
    so this validates the quadrature itself rather than a different
    physical model.

    Args:
        miss_vector_eci, v_rel_eci, combined_covariance_eci,
        hard_body_radius_km: identical meaning/units to
            core.collision.probability_of_collision()'s arguments.
        n_samples: number of Monte Carlo draws. Standard error shrinks as
            1/sqrt(n_samples) — quadrupling n_samples halves it.
        random_state: seed for reproducibility (None = nondeterministic).

    Returns:
        MonteCarloPcResult with the point estimate and its standard error.
    """
    e_x, e_y = b_plane_frame(miss_vector_eci, v_rel_eci)
    B = np.vstack([e_x, e_y])

    miss_2d = B @ miss_vector_eci
    covariance_2d = B @ combined_covariance_eci @ B.T

    rng = np.random.default_rng(random_state)
    samples = rng.multivariate_normal(mean=miss_2d, cov=covariance_2d, size=n_samples)

    distances = np.linalg.norm(samples, axis=1)
    n_hits = int(np.sum(distances <= hard_body_radius_km))
    pc_estimate = n_hits / n_samples

    standard_error = float(np.sqrt(pc_estimate * (1 - pc_estimate) / n_samples))

    return MonteCarloPcResult(
        pc_estimate=pc_estimate, standard_error=standard_error,
        n_samples=n_samples, n_hits=n_hits,
    )


@dataclass
class PcValidationReport:
    """Side-by-side comparison of the analytic (quadrature) and Monte Carlo Pc estimates."""

    analytic_pc: float
    monte_carlo: MonteCarloPcResult

    @property
    def absolute_difference(self) -> float:
        return abs(self.analytic_pc - self.monte_carlo.pc_estimate)

    @property
    def difference_in_standard_errors(self) -> float:
        """How many Monte Carlo standard errors apart the two estimates are — the natural agreement metric."""
        if self.monte_carlo.standard_error == 0:
            return float("inf") if self.absolute_difference > 0 else 0.0
        return self.absolute_difference / self.monte_carlo.standard_error


def validate_pc(
    miss_vector_eci: np.ndarray,
    v_rel_eci: np.ndarray,
    combined_covariance_eci: np.ndarray,
    hard_body_radius_km: float,
    n_samples: int = 200_000,
    random_state: int | None = None,
) -> PcValidationReport:
    """Compute both the analytic and Monte Carlo Pc for the same encounter geometry and compare them."""
    analytic_pc = probability_of_collision(miss_vector_eci, v_rel_eci, combined_covariance_eci, hard_body_radius_km)
    mc_result = monte_carlo_collision_probability(
        miss_vector_eci, v_rel_eci, combined_covariance_eci, hard_body_radius_km,
        n_samples=n_samples, random_state=random_state,
    )
    return PcValidationReport(analytic_pc=analytic_pc, monte_carlo=mc_result)
