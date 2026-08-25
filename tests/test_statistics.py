from datetime import datetime, timezone

from core.tle_manager import Satellite
from reports.statistics import (
    REGIME_BANDS,
    altitude_inclination_histogram_summary,
    classify_regime,
    compute_population_statistics,
)


def _make_sat(norad_id, incl, ecc, a, classification="active"):
    return Satellite(
        norad_id=norad_id, name=f"SAT-{norad_id}", line1="l1", line2="l2",
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        inclination_deg=incl, eccentricity=ecc, mean_motion_rev_per_day=15.0,
        semi_major_axis_km=a, classification=classification,
    )


def test_classify_regime_bands():
    assert classify_regime(500.0) == "LEO"
    assert classify_regime(20000.0) == "MEO"
    assert classify_regime(35786.0) == "GEO"
    assert classify_regime(50000.0) == "HEO"


def test_compute_population_statistics_empty_list():
    stats = compute_population_statistics([])
    assert stats.total_count == 0
    assert stats.count_by_classification == {}
    assert stats.count_by_regime == {}


def test_compute_population_statistics_basic_counts():
    sats = [
        _make_sat(1, 51.6, 0.0001, 6793.0, classification="active"),   # ~415 km LEO
        _make_sat(2, 98.0, 0.0001, 7178.0, classification="debris"),   # ~800 km LEO
        _make_sat(3, 0.05, 0.0001, 42164.0, classification="stations"),  # GEO
    ]
    stats = compute_population_statistics(sats)
    assert stats.total_count == 3
    assert stats.count_by_classification == {"active": 1, "debris": 1, "stations": 1}
    assert stats.count_by_regime["LEO"] == 2
    assert stats.count_by_regime["GEO"] == 1
    assert stats.altitude_min_km < stats.altitude_max_km


def test_compute_population_statistics_inclination_mean():
    sats = [_make_sat(1, 30.0, 0.0, 7000.0), _make_sat(2, 60.0, 0.0, 7000.0)]
    stats = compute_population_statistics(sats)
    assert abs(stats.inclination_mean_deg - 45.0) < 1e-9


def test_altitude_inclination_histogram_summary_is_json_shaped():
    sats = [_make_sat(1, 51.6, 0.0001, 6793.0), _make_sat(2, 98.0, 0.0001, 7178.0)]
    summary = altitude_inclination_histogram_summary(sats)
    assert "counts" in summary
    assert "altitude_edges_km" in summary
    assert "inclination_edges_deg" in summary
    assert sum(sum(row) for row in summary["counts"]) == 2
