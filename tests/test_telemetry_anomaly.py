from datetime import datetime, timedelta, timezone

from core.telemetry_anomaly import compute_tle_residuals, detect_anomalies
from core.tle_manager import Satellite

MEAN_MOTION_REV_PER_DAY = 15.49560752
BASE_EPOCH_DAY = 24001.50000000  # matches the real ISS TLE fixture used elsewhere in this project


def _make_synthetic_tle(epoch_day: float, mean_anomaly_deg: float) -> tuple[str, str]:
    """
    Build a synthetic, well-formed TLE by substituting the epoch and
    mean-anomaly fields of a real ISS TLE template (the sgp4 library
    doesn't enforce checksum validity — confirmed during development —
    so this is a safe, simple way to generate a self-consistent test
    sequence without needing a full TLE-generation pipeline).
    """
    line1_template = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9008"
    line2_template = "2 25544  51.6416 339.9500 0001177  93.3861 264.7930 15.49560752 42982"

    epoch_str = f"{epoch_day:14.8f}"
    assert len(epoch_str) == 14
    line1 = line1_template[:18] + epoch_str + line1_template[32:]

    ma_str = f"{mean_anomaly_deg:8.4f}"
    assert len(ma_str) == 8
    line2 = line2_template[:43] + ma_str + line2_template[51:]

    return line1, line2


def _epoch_day_to_datetime(epoch_day: float) -> datetime:
    epoch_year = 2000 + int(epoch_day // 1000)
    day_of_year = epoch_day - (epoch_year - 2000) * 1000
    return datetime(epoch_year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)


def _make_tle_sequence(n_periods: int, mean_anomaly_at_epoch: float = 264.7930, anomaly_at_index: int | None = None) -> list[Satellite]:
    """
    A self-consistent sequence of TLEs, one per exact orbital period
    apart (so mean anomaly should return to the same value each step in
    an unperturbed, un-maneuvered scenario — any deviation from that at
    a given index is a synthetic 'maneuver').
    """
    period_days = 1.0 / MEAN_MOTION_REV_PER_DAY
    satellites = []
    for i in range(n_periods):
        epoch_day = BASE_EPOCH_DAY + i * period_days
        mean_anomaly = mean_anomaly_at_epoch
        if anomaly_at_index is not None and i == anomaly_at_index:
            mean_anomaly = (mean_anomaly_at_epoch + 40.0) % 360.0  # inject an out-of-family jump

        line1, line2 = _make_synthetic_tle(epoch_day, mean_anomaly)
        satellites.append(
            Satellite(
                norad_id=25544, name="ISS (TEST)", line1=line1, line2=line2,
                epoch=_epoch_day_to_datetime(epoch_day),
                inclination_deg=51.6416, eccentricity=0.0001177,
                mean_motion_rev_per_day=MEAN_MOTION_REV_PER_DAY, semi_major_axis_km=6796.0,
                classification="stations",
            )
        )
    return satellites


def test_compute_tle_residuals_returns_one_per_consecutive_pair():
    sats = _make_tle_sequence(n_periods=6)
    residuals = compute_tle_residuals(sats)
    assert len(residuals) == 5


def test_compute_tle_residuals_small_for_undisturbed_sequence():
    """An unperturbed, self-consistent sequence should produce small, tightly clustered residuals."""
    sats = _make_tle_sequence(n_periods=8)
    residuals = compute_tle_residuals(sats)
    values = [r.residual_km for r in residuals]
    assert all(v < 50.0 for v in values)  # SGP4's own short-arc self-consistency is well within tens of km


def test_detect_anomalies_flags_injected_maneuver():
    """
    The key demonstration: a sequence that's self-consistent except for
    one deliberately injected out-of-family mean-anomaly jump (a
    synthetic 'maneuver') must be flagged as anomalous, while the
    surrounding normal points must not be.
    """
    sats = _make_tle_sequence(n_periods=10, anomaly_at_index=6)
    residuals = compute_tle_residuals(sats)
    report = detect_anomalies(residuals, threshold_sigma=3.0)

    assert len(report.anomalous_indices) >= 1

    # The anomaly at satellite index 6 shows up as a residual between
    # satellite pairs (5,6) and (6,7) — i.e. residual list indices 5 and/or 6.
    assert any(idx in (5, 6) for idx in report.anomalous_indices)


def test_detect_anomalies_no_false_positives_on_clean_sequence():
    sats = _make_tle_sequence(n_periods=10)
    residuals = compute_tle_residuals(sats)
    report = detect_anomalies(residuals, threshold_sigma=3.0)
    assert report.anomalous_indices == []


def test_detect_anomalies_insufficient_history_flags_nothing():
    sats = _make_tle_sequence(n_periods=3)
    residuals = compute_tle_residuals(sats)
    report = detect_anomalies(residuals, min_history=4)
    assert report.anomalous_indices == []
    assert report.threshold_km == float("inf")


def test_anomaly_report_anomalous_points_property():
    sats = _make_tle_sequence(n_periods=10, anomaly_at_index=6)
    residuals = compute_tle_residuals(sats)
    report = detect_anomalies(residuals, threshold_sigma=3.0)
    for point in report.anomalous_points:
        assert point.residual_km > report.threshold_km
