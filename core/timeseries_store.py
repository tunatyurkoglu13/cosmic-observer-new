"""
core/timeseries_store.py — Generic SQLite-backed time-series storage for
real, periodically-sampled metrics (Kp index, DSN link activity, NEO risk
count, CV anomaly score, ...).

Every other data client in this project answers "what is X right now?".
This is the one piece that remembers what X *was* a moment ago — a
single small, generic samples table (metric, timestamp, value, metadata)
any part of the app can write real observations into and query real
history back out of, so the dashboard can show genuine trend lines
instead of only ever a live snapshot.

Not a general-purpose time-series database (no downsampling/rollups, no
distributed anything) — a single local SQLite file is exactly right for
this project's actual scale: a handful of metrics, sampled every 1-5
minutes, queried by one dashboard. See app.py's background sampler loop
for what actually gets recorded and how often.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cache" / "timeseries.sqlite3"


@dataclass
class Sample:
    metric: str
    timestamp: datetime
    value: float
    metadata: dict | None = None


class TimeSeriesStore:
    """Records and queries real-valued metric samples over time."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH, retention_days: int = 30):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    value REAL NOT NULL,
                    metadata TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_metric_ts ON samples (metric, timestamp)")

    def record(self, metric: str, value: float, metadata: dict | None = None, timestamp: datetime | None = None) -> None:
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO samples (metric, timestamp, value, metadata) VALUES (?, ?, ?, ?)",
                (metric, ts, value, json.dumps(metadata) if metadata else None),
            )

    def query(self, metric: str, since: datetime | None = None, limit: int = 2000) -> list[Sample]:
        query_sql = "SELECT metric, timestamp, value, metadata FROM samples WHERE metric = ?"
        params: list = [metric]
        if since is not None:
            query_sql += " AND timestamp >= ?"
            params.append(since.isoformat())
        query_sql += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query_sql, params).fetchall()

        return [
            Sample(
                metric=r[0], timestamp=datetime.fromisoformat(r[1]), value=r[2],
                metadata=json.loads(r[3]) if r[3] else None,
            )
            for r in rows
        ]

    def latest(self, metric: str) -> Sample | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT metric, timestamp, value, metadata FROM samples WHERE metric = ? ORDER BY timestamp DESC LIMIT 1",
                (metric,),
            ).fetchone()
        if row is None:
            return None
        return Sample(
            metric=row[0], timestamp=datetime.fromisoformat(row[1]), value=row[2],
            metadata=json.loads(row[3]) if row[3] else None,
        )

    def list_metrics(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT DISTINCT metric FROM samples ORDER BY metric").fetchall()
        return [r[0] for r in rows]

    def prune(self) -> int:
        """Delete samples older than retention_days. Returns the number of rows deleted."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM samples WHERE timestamp < ?", (cutoff,))
            return cursor.rowcount
