from datetime import datetime, timedelta, timezone

from core.tle_manager import Satellite
from reports.risk_report import generate_risk_report, render_text_summary

# ISS and its docked NAUKA module: essentially co-located, guaranteed to
# be flagged as an "emergency"-level conjunction over any short window.
ISS_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9008"
ISS_LINE2 = "2 25544  51.6416 339.9500 0001177  93.3861 264.7930 15.49560752 42982"
NAUKA_LINE1 = "1 49044U 21066A   24001.51000000  .00016000  00000-0  98000-4 0  9001"
NAUKA_LINE2 = "2 49044  51.6410 339.0000 0001200  90.0000 270.0000 15.49500000 12345"


def _make_sat(norad_id, name, line1, line2):
    return Satellite(
        norad_id=norad_id, name=name, line1=line1, line2=line2,
        epoch=datetime(2024, 1, 1, tzinfo=timezone.utc),
        inclination_deg=51.64, eccentricity=0.0001, mean_motion_rev_per_day=15.495,
        semi_major_axis_km=6796.0, classification="stations",
    )


def test_generate_risk_report_structure():
    iss = _make_sat(25544, "ISS", ISS_LINE1, ISS_LINE2)
    nauka = _make_sat(49044, "NAUKA", NAUKA_LINE1, NAUKA_LINE2)

    start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)

    report = generate_risk_report([iss, nauka], start, end)
    assert report.objects_screened == 2
    assert isinstance(report.entries, list)
    for entry in report.entries:
        assert entry.summary.alert_level in ("emergency", "high", "medium", "info")
        if entry.summary.alert_level in ("emergency", "high"):
            # A maneuver suggestion should generally be produced for high-risk entries
            # (unless the CW linearization validity check rejected the target distance).
            assert entry.suggested_maneuver is None or entry.suggested_maneuver.delta_v_km_s > 0


def test_risk_report_counts_by_level_matches_entries():
    iss = _make_sat(25544, "ISS", ISS_LINE1, ISS_LINE2)
    nauka = _make_sat(49044, "NAUKA", NAUKA_LINE1, NAUKA_LINE2)
    start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)

    report = generate_risk_report([iss, nauka], start, end)
    total_from_counts = sum(report.counts_by_level.values())
    assert total_from_counts == len(report.entries)


def test_render_text_summary_contains_key_fields():
    iss = _make_sat(25544, "ISS", ISS_LINE1, ISS_LINE2)
    nauka = _make_sat(49044, "NAUKA", NAUKA_LINE1, NAUKA_LINE2)
    start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)

    report = generate_risk_report([iss, nauka], start, end)
    text = render_text_summary(report)
    assert "COLLISION RISK REPORT" in text
    assert "Objects screened: 2" in text


def test_generate_risk_report_no_conjunctions_for_unrelated_satellites():
    # Two satellites in wildly different altitude bands should screen out at Stage 1.
    leo = Satellite(
        norad_id=1, name="LEO-SAT", line1=ISS_LINE1, line2=ISS_LINE2,
        epoch=datetime(2024, 1, 1, tzinfo=timezone.utc),
        inclination_deg=51.6, eccentricity=0.0001, mean_motion_rev_per_day=15.5,
        semi_major_axis_km=6793.0, classification="active",
    )
    geo = Satellite(
        norad_id=2, name="GEO-SAT", line1="1 00000U 00000A   24001.50000000 0.0 00000-0 00000-0 0 0000",
        line2="2 00000   0.0500 100.0000 0001000  90.0000 270.0000  1.00270000000001",
        epoch=datetime(2024, 1, 1, tzinfo=timezone.utc),
        inclination_deg=0.05, eccentricity=0.0001, mean_motion_rev_per_day=1.0027,
        semi_major_axis_km=42164.0, classification="active",
    )
    start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    report = generate_risk_report([leo, geo], start, end)
    assert report.entries == []
