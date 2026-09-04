"""
core/alert_store.py — Persistent store for real SSA alert events
(conjunctions, CV anomalies, NEO risk/close approaches) — the active
notification layer, as opposed to every other module in this project,
which only ever answers "what is the state right now."

Real, not synthetic: every alert this store ever receives is written by
core.conjunction_watch (real SGP4-propagated catalog screening), app.py's
live /ws/cv anomaly detector, or data.nasa_cneos/data.neows (real
NASA/JPL feeds) — this module itself never invents an event, only
records/dedupes/serves ones handed to it.

Deduplication: `record()` takes a `dedup_key` identifying "the same
underlying condition" (e.g. one specific conjunction pair, one specific
NEO) and a `cooldown_minutes` — if that key already fired within the
cooldown window, the new call is silently suppressed (returns None)
rather than spamming a fresh row every detection-loop tick for a
condition that's still ongoing.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cache" / "alerts.sqlite3"

SEVERITIES = ("info", "warning", "critical")


@dataclass
class AlertEvent:
    id: int | None
    category: str            # "conjunction" | "anomaly" | "neo_close_approach" | "neo_risk"
    severity: str              # "info" | "warning" | "critical"
    title: str
    description: str
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
    dedup_key: str = ""
    acknowledged: bool = False


class AlertStore:
    """Records, dedupes, and serves real alert events."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    dedup_key TEXT NOT NULL,
                    acknowledged INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_dedup ON alerts (dedup_key, timestamp)")

    def record(
        self,
        category: str,
        severity: str,
        title: str,
        description: str,
        metadata: dict | None = None,
        dedup_key: str | None = None,
        cooldown_minutes: float = 60.0,
    ) -> AlertEvent | None:
        """
        Insert a new alert unless one with the same dedup_key fired
        within the last `cooldown_minutes` (in which case this returns
        None — the caller should treat that as "suppressed, not an error").
        """
        if severity not in SEVERITIES:
            raise ValueError(f"Unknown severity '{severity}'. Must be one of {SEVERITIES}")

        now = datetime.now(timezone.utc)
        dedup_key = dedup_key or f"{category}:{title}"

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT timestamp FROM alerts WHERE dedup_key = ? ORDER BY timestamp DESC LIMIT 1",
                (dedup_key,),
            ).fetchone()
            if row is not None:
                last_ts = datetime.fromisoformat(row[0])
                if (now - last_ts).total_seconds() < cooldown_minutes * 60:
                    return None

            cursor = conn.execute(
                "INSERT INTO alerts (category, severity, title, description, timestamp, metadata, dedup_key, acknowledged) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (category, severity, title, description, now.isoformat(), json.dumps(metadata or {}), dedup_key),
            )
            new_id = cursor.lastrowid

        return AlertEvent(
            id=new_id, category=category, severity=severity, title=title, description=description,
            timestamp=now, metadata=metadata or {}, dedup_key=dedup_key, acknowledged=False,
        )

    def query(self, limit: int = 100, category: str | None = None, unacknowledged_only: bool = False) -> list[AlertEvent]:
        sql = "SELECT id, category, severity, title, description, timestamp, metadata, dedup_key, acknowledged FROM alerts"
        conditions = []
        params: list = []
        if category:
            conditions.append("category = ?")
            params.append(category)
        if unacknowledged_only:
            conditions.append("acknowledged = 0")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get(self, alert_id: int) -> AlertEvent | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, category, severity, title, description, timestamp, metadata, dedup_key, acknowledged "
                "FROM alerts WHERE id = ?",
                (alert_id,),
            ).fetchone()
        return self._row_to_event(row) if row else None

    def _row_to_event(self, row) -> AlertEvent:
        return AlertEvent(
            id=row[0], category=row[1], severity=row[2], title=row[3], description=row[4],
            timestamp=datetime.fromisoformat(row[5]), metadata=json.loads(row[6]) if row[6] else {},
            dedup_key=row[7], acknowledged=bool(row[8]),
        )

    def acknowledge(self, alert_id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
            return cursor.rowcount > 0

    def count_unacknowledged(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged = 0").fetchone()
        return row[0] if row else 0
