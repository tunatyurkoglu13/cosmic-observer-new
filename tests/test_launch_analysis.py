from datetime import datetime, timedelta, timezone

from core.launch_window import LAUNCH_SITES, LaunchTarget
from reports.launch_analysis import (
    KP_CAUTION_THRESHOLD,
    generate_launch_analysis,
    render_text_summary,
)


def test_generate_launch_analysis_without_space_weather_check():
    site = LAUNCH_SITES["cape_canaveral"]
    target = LaunchTarget(altitude_km=400.0, inclination_deg=51.6)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    report = generate_launch_analysis(site, target, start, end, check_space_weather=False)
    assert report.space_weather_kp is None
    assert len(report.assessments) >= 1
    for a in report.assessments:
        assert a.window.feasible  # 51.6 deg from Cape Canaveral is a known-feasible case
        assert a.caution is False


def test_generate_launch_analysis_flags_infeasible_azimuth():
    site = LAUNCH_SITES["vandenberg"]
    target = LaunchTarget(altitude_km=500.0, inclination_deg=30.0)  # below Vandenberg's own latitude
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    report = generate_launch_analysis(site, target, start, end, check_space_weather=False)
    assert len(report.assessments) == 1
    assert report.assessments[0].caution is True
    assert report.assessments[0].caution_reason != ""


def test_render_text_summary_contains_key_fields():
    site = LAUNCH_SITES["cape_canaveral"]
    target = LaunchTarget(altitude_km=400.0, inclination_deg=51.6)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    report = generate_launch_analysis(site, target, start, end, check_space_weather=False)
    text = render_text_summary(report)
    assert "LAUNCH ANALYSIS REPORT" in text
    assert site.name in text


def test_kp_caution_threshold_is_g1_storm_level():
    assert KP_CAUTION_THRESHOLD == 5.0
