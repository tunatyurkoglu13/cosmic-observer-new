"""
reports/risk_report.py — Collision-risk report generation.

Runs the Phase 3 conjunction screening funnel over a catalog and packages
the results into a report: which conjunctions were flagged, at what risk
level, and — for anything at "high" or "emergency" alert level — a
suggested along-track avoidance maneuver (Phase 3's collision_avoid),
computed against a conservative lead time and target miss distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from core.collision import ConjunctionResult
from core.kepler import mean_motion
from core.tle_manager import Satellite
from stm.collision_avoid import CollisionAvoidanceManeuver, plan_along_track_maneuver
from stm.conjunction import ConjunctionSummary, screen_catalog

# Alert levels that warrant an automatically-suggested avoidance maneuver
# in the report (lower-risk "medium"/"info" conjunctions are reported but
# not accompanied by a maneuver suggestion).
MANEUVER_SUGGESTED_LEVELS = ("emergency", "high")

DEFAULT_MANEUVER_LEAD_TIME = timedelta(hours=6)
DEFAULT_TARGET_MISS_DISTANCE_KM = 5.0


@dataclass
class ConjunctionReportEntry:
    """One flagged conjunction, plus an optional suggested maneuver."""

    summary: ConjunctionSummary
    suggested_maneuver: CollisionAvoidanceManeuver | None = None


@dataclass
class RiskReport:
    """A full collision-risk assessment over a catalog and time window."""

    generated_at: datetime
    window_start: datetime
    window_end: datetime
    objects_screened: int
    entries: list[ConjunctionReportEntry]

    @property
    def counts_by_level(self) -> dict[str, int]:
        counts = {"emergency": 0, "high": 0, "medium": 0, "info": 0}
        for entry in self.entries:
            counts[entry.summary.alert_level] += 1
        return counts


def generate_risk_report(
    satellites: list[Satellite],
    window_start: datetime,
    window_end: datetime,
    maneuver_lead_time: timedelta = DEFAULT_MANEUVER_LEAD_TIME,
    target_miss_distance_km: float = DEFAULT_TARGET_MISS_DISTANCE_KM,
) -> RiskReport:
    """
    Screen a catalog for conjunctions over [window_start, window_end] and
    assemble a RiskReport, including suggested avoidance maneuvers for
    high/emergency-level conjunctions where a maneuver lead time before
    TCA is actually available.

    Args:
        satellites: catalog objects to screen (all-vs-all).
        window_start, window_end: UTC screening window.
        maneuver_lead_time: how far before TCA the suggested burn is
            planned for (must leave positive time-to-TCA to be valid;
            entries where TCA is sooner than this are reported without a
            maneuver suggestion rather than raising).
        target_miss_distance_km: desired post-maneuver miss distance.

    Returns:
        RiskReport with one ConjunctionReportEntry per flagged conjunction.
    """
    summaries = screen_catalog(satellites, window_start, window_end)

    entries = []
    for summary in summaries:
        maneuver = None
        if summary.alert_level in MANEUVER_SUGGESTED_LEVELS:
            burn_time = summary.tca - maneuver_lead_time
            n = mean_motion(summary.sat1.semi_major_axis_km)
            conjunction = ConjunctionResult(
                tca=summary.tca,
                miss_distance_km=summary.miss_distance_km,
                relative_speed_km_s=summary.relative_speed_km_s,
            )
            try:
                maneuver = plan_along_track_maneuver(
                    conjunction, burn_time, mean_motion_rad_s=n,
                    target_miss_distance_km=target_miss_distance_km,
                )
            except ValueError:
                maneuver = None  # target beyond CW linearization validity; report without one

        entries.append(ConjunctionReportEntry(summary=summary, suggested_maneuver=maneuver))

    return RiskReport(
        generated_at=datetime.now(window_start.tzinfo),
        window_start=window_start,
        window_end=window_end,
        objects_screened=len(satellites),
        entries=entries,
    )


def render_text_summary(report: RiskReport) -> str:
    """Render a plain-text human-readable summary of a RiskReport."""
    lines = [
        f"COLLISION RISK REPORT — generated {report.generated_at.isoformat()}",
        f"Screening window: {report.window_start.isoformat()} -> {report.window_end.isoformat()}",
        f"Objects screened: {report.objects_screened}",
        f"Conjunctions flagged: {len(report.entries)}",
        f"  By level: {report.counts_by_level}",
        "",
    ]
    for entry in report.entries:
        s = entry.summary
        lines.append(
            f"[{s.alert_level.upper()}] {s.sat1.name} (NORAD {s.sat1.norad_id}) <-> "
            f"{s.sat2.name} (NORAD {s.sat2.norad_id}): "
            f"{s.miss_distance_km:.2f} km at {s.tca.isoformat()}, "
            f"rel. speed {s.relative_speed_km_s:.2f} km/s"
        )
        if entry.suggested_maneuver:
            m = entry.suggested_maneuver
            lines.append(
                f"    -> Suggested burn at {m.burn_time.isoformat()}: "
                f"{m.delta_v_km_s * 1000:.2f} m/s along-track "
                f"(-> {m.predicted_miss_distance_km:.1f} km predicted miss distance)"
            )
    return "\n".join(lines)
