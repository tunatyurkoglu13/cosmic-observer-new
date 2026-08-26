"""
core/telemetry_anomaly.py — Maneuver/anomaly detection from TLE residuals.

Bridges the project's two halves — cv/anomaly.py's autoencoder learned
"what does normal look like, flag what doesn't" pattern, and the
orbital-mechanics core — by applying the same idea to orbital state
residuals instead of images.

Method: for a satellite with a *sequence* of real TLEs (successive
CelesTrak/Space-Track updates over time), propagate each TLE forward to
the epoch of the NEXT TLE using SGP4, and compare that prediction
against what the next TLE itself says the position actually was. Under
routine, un-maneuvered flight, this residual is small and consistent
(driven only by SGP4's own approximation error and unmodeled forces
between updates). A station-keeping burn, collision-avoidance maneuver,
or genuine anomaly shows up as a residual sharply larger than the
established baseline — exactly the "reconstruction error" concept in
cv/anomaly.py's autoencoder, applied to orbital state instead of pixels.

This is a real, simple technique operational analysts actually use as a
first-pass maneuver detector (a big, sudden jump in "predicted vs. next
TLE" residual is one of the most direct signals that something happened
between two catalog updates) — not a sophisticated learned model, but a
genuine, useful one that requires no training data.

Detection rule: flag a residual as anomalous if it exceeds
`threshold_sigma` standard deviations above the *median* (not mean —
robust to the very anomalies we're trying to detect skewing the
baseline) of the object's own residual history, in the spirit of
cv.anomaly's contamination-based Isolation Forest threshold, but simple
enough to need no separate ML dependency for this one-dimensional signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from core.propagator import Propagator
from core.tle_manager import Satellite


@dataclass
class ResidualPoint:
    """One TLE-to-TLE residual: how far SGP4's prediction from the earlier TLE missed the later TLE's own position."""

    epoch: datetime           # epoch of the *later* TLE (when the residual is "observed")
    residual_km: float
    predicted_from_epoch: datetime  # epoch of the earlier TLE used to predict


@dataclass
class AnomalyReport:
    """Residual history for one object, with anomalous points flagged."""

    residuals: list[ResidualPoint]
    threshold_km: float
    anomalous_indices: list[int]

    @property
    def anomalous_points(self) -> list[ResidualPoint]:
        return [self.residuals[i] for i in self.anomalous_indices]


def compute_tle_residuals(tle_sequence: list[Satellite]) -> list[ResidualPoint]:
    """
    For a time-ordered sequence of TLEs for the SAME object, compute the
    position residual between each TLE's own SGP4-propagated position at
    the NEXT TLE's epoch, and where that next TLE says the object
    actually was (via ITS OWN SGP4 propagation at its own epoch — a TLE's
    stated position at its own epoch is, by construction, what SGP4
    seeded with that TLE returns at time zero).

    Args:
        tle_sequence: Satellite objects for one object, sorted by epoch
            ascending (out-of-order input is sorted defensively).

    Returns:
        List of ResidualPoint, one per consecutive TLE pair (length
        len(tle_sequence) - 1). Pairs where propagation fails (e.g. a
        corrupt TLE) are skipped rather than raising.
    """
    ordered = sorted(tle_sequence, key=lambda s: s.epoch)

    residuals = []
    for earlier, later in zip(ordered, ordered[1:]):
        try:
            earlier_prop = Propagator(earlier.line1, earlier.line2, name=earlier.name)
            later_prop = Propagator(later.line1, later.line2, name=later.name)

            predicted_state = earlier_prop.propagate(later.epoch)
            actual_state = later_prop.propagate(later.epoch)
        except RuntimeError:
            continue  # SGP4 error (e.g. decayed orbit) for this pair — skip, don't abort the whole sequence

        residual_km = float(np.linalg.norm(predicted_state.r_teme - actual_state.r_teme))
        residuals.append(ResidualPoint(epoch=later.epoch, residual_km=residual_km, predicted_from_epoch=earlier.epoch))

    return residuals


def detect_anomalies(residuals: list[ResidualPoint], threshold_sigma: float = 3.0, min_history: int = 4) -> AnomalyReport:
    """
    Flag residuals that stand out from an object's own established
    baseline, using a robust (median + MAD-based) threshold rather than
    mean + standard deviation, which a single large maneuver residual
    would otherwise skew.

    MAD (median absolute deviation) is converted to an equivalent
    "robust sigma" via the standard 1.4826 scale factor (the constant
    that makes MAD-based sigma match ordinary standard deviation for
    normally-distributed data — see Ruppert, "Statistics and Data
    Analysis for Financial Engineering", or any robust-statistics text).

    Args:
        residuals: from compute_tle_residuals().
        threshold_sigma: how many robust-sigma above the median counts as anomalous.
        min_history: minimum number of residuals needed to establish a
            baseline at all; with fewer, everything is reported as
            non-anomalous (too little history to judge "normal" yet).

    Returns:
        AnomalyReport with a robust threshold and the flagged indices.
    """
    if len(residuals) < min_history:
        return AnomalyReport(residuals=residuals, threshold_km=float("inf"), anomalous_indices=[])

    values = np.array([r.residual_km for r in residuals])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad

    if robust_sigma == 0:
        # Degenerate case: a perfectly uniform residual history (e.g. all
        # zero in a synthetic test) — fall back to flagging anything
        # strictly greater than the (zero-spread) baseline.
        threshold_km = median
        anomalous_indices = [i for i, v in enumerate(values) if v > threshold_km]
    else:
        threshold_km = median + threshold_sigma * robust_sigma
        anomalous_indices = [i for i, v in enumerate(values) if v > threshold_km]

    return AnomalyReport(residuals=residuals, threshold_km=threshold_km, anomalous_indices=anomalous_indices)
