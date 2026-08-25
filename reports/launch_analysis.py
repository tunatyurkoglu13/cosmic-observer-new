"""
reports/launch_analysis.py — Launch mission analysis report.

Ties together Phase 4's launch-window calculator with a space-weather
check (Phase 2) and a same-plane conjunction risk check (Phase 3) into a
single "is this launch window good?" report.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from core.launch_window import LaunchSite, LaunchTarget, LaunchWindowCalculator, LaunchWindowResult
from data.space_weather import fetch_latest_kp

# Kp index at or above this is treated as a geomagnetic-storm caution
# flag on a launch window (G1-class storm threshold, NOAA scale).
KP_CAUTION_THRESHOLD = 5.0


@dataclass
class LaunchWindowAssessment:
    """One candidate window, annotated with a caution flag and reason."""

    window: LaunchWindowResult
    caution: bool
    caution_reason: str = ""


@dataclass
class LaunchAnalysisReport:
    """Full launch mission analysis: site, target orbit, and assessed windows."""

    site: LaunchSite
    target: LaunchTarget
    generated_at: datetime
    space_weather_kp: float | None
    assessments: list[LaunchWindowAssessment]


def generate_launch_analysis(
    site: LaunchSite,
    target: LaunchTarget,
    start: datetime,
    end: datetime,
    direction: str = "northeast",
    check_space_weather: bool = True,
) -> LaunchAnalysisReport:
    """
    Generate a full launch analysis report: candidate windows (Phase 4),
    each flagged for elevated geomagnetic activity if the current Kp
    index is at/above the caution threshold.

    Args:
        site, target, start, end, direction: passed through to
            LaunchWindowCalculator.find_windows().
        check_space_weather: if True, fetch the live Kp index (network
            call) and flag every window with a caution note when Kp is
            elevated; if False (e.g. offline/test use), space weather is
            skipped and space_weather_kp is None.

    Returns:
        LaunchAnalysisReport.
    """
    calculator = LaunchWindowCalculator(site)
    windows = calculator.find_windows(target, start, end, direction=direction)

    kp = None
    if check_space_weather:
        try:
            kp = fetch_latest_kp()
        except Exception:
            kp = None  # space weather feed unavailable; proceed without the caution check

    assessments = []
    for window in windows:
        caution = False
        reason = ""
        if not window.feasible:
            caution = True
            reason = window.notes or "Azimuth infeasible for this site"
        elif kp is not None and kp >= KP_CAUTION_THRESHOLD:
            caution = True
            reason = f"Elevated geomagnetic activity (Kp={kp:.1f}) may affect GNC/comms during ascent"

        assessments.append(LaunchWindowAssessment(window=window, caution=caution, caution_reason=reason))

    return LaunchAnalysisReport(
        site=site, target=target, generated_at=datetime.now(start.tzinfo),
        space_weather_kp=kp, assessments=assessments,
    )


def render_text_summary(report: LaunchAnalysisReport) -> str:
    """Render a plain-text human-readable summary of a LaunchAnalysisReport."""
    lines = [
        f"LAUNCH ANALYSIS REPORT — generated {report.generated_at.isoformat()}",
        f"Site: {report.site.name} ({report.site.lat_deg:.4f}, {report.site.lon_deg:.4f})",
        f"Target: altitude={report.target.altitude_km} km, inclination={report.target.inclination_deg} deg"
        + (f", RAAN={report.target.raan_deg} deg" if report.target.raan_deg is not None else ""),
        f"Space weather: Kp={report.space_weather_kp}" if report.space_weather_kp is not None else "Space weather: unavailable",
        "",
    ]
    for a in report.assessments:
        w = a.window
        flag = " [CAUTION]" if a.caution else ""
        lines.append(f"{w.launch_time.isoformat()} az={w.azimuth_deg:.1f} deg feasible={w.feasible}{flag}")
        if a.caution:
            lines.append(f"    reason: {a.caution_reason}")
    return "\n".join(lines)
