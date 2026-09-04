from datetime import datetime, timedelta, timezone

from core.timeseries_store import TimeSeriesStore


def test_record_and_query_roundtrip(tmp_path):
    store = TimeSeriesStore(db_path=tmp_path / "ts.sqlite3")
    now = datetime.now(timezone.utc)
    store.record("kp_index", 3.5, timestamp=now)
    store.record("kp_index", 4.0, timestamp=now + timedelta(minutes=1))

    samples = store.query("kp_index")
    assert len(samples) == 2
    assert samples[0].value == 3.5
    assert samples[1].value == 4.0
    assert samples[0].timestamp <= samples[1].timestamp  # ascending order


def test_query_filters_by_since(tmp_path):
    store = TimeSeriesStore(db_path=tmp_path / "ts.sqlite3")
    now = datetime.now(timezone.utc)
    store.record("kp_index", 1.0, timestamp=now - timedelta(hours=2))
    store.record("kp_index", 2.0, timestamp=now - timedelta(minutes=1))

    recent = store.query("kp_index", since=now - timedelta(hours=1))
    assert len(recent) == 1
    assert recent[0].value == 2.0


def test_query_respects_limit(tmp_path):
    store = TimeSeriesStore(db_path=tmp_path / "ts.sqlite3")
    now = datetime.now(timezone.utc)
    for i in range(10):
        store.record("kp_index", float(i), timestamp=now + timedelta(seconds=i))

    samples = store.query("kp_index", limit=3)
    assert len(samples) == 3


def test_query_different_metrics_independent(tmp_path):
    store = TimeSeriesStore(db_path=tmp_path / "ts.sqlite3")
    store.record("kp_index", 3.0)
    store.record("dsn_active_spacecraft", 7.0)

    assert len(store.query("kp_index")) == 1
    assert len(store.query("dsn_active_spacecraft")) == 1
    assert store.query("nonexistent_metric") == []


def test_latest_returns_most_recent_sample(tmp_path):
    store = TimeSeriesStore(db_path=tmp_path / "ts.sqlite3")
    now = datetime.now(timezone.utc)
    store.record("kp_index", 1.0, timestamp=now - timedelta(hours=1))
    store.record("kp_index", 5.0, timestamp=now)

    latest = store.latest("kp_index")
    assert latest.value == 5.0


def test_latest_returns_none_for_unknown_metric(tmp_path):
    store = TimeSeriesStore(db_path=tmp_path / "ts.sqlite3")
    assert store.latest("nonexistent") is None


def test_list_metrics_returns_distinct_sorted_names(tmp_path):
    store = TimeSeriesStore(db_path=tmp_path / "ts.sqlite3")
    store.record("kp_index", 1.0)
    store.record("dsn_active_spacecraft", 2.0)
    store.record("kp_index", 3.0)

    assert store.list_metrics() == ["dsn_active_spacecraft", "kp_index"]


def test_metadata_roundtrip(tmp_path):
    store = TimeSeriesStore(db_path=tmp_path / "ts.sqlite3")
    store.record("neo_max_torino", 2.0, metadata={"designation": "2024 AB"})
    sample = store.latest("neo_max_torino")
    assert sample.metadata == {"designation": "2024 AB"}


def test_prune_deletes_old_samples(tmp_path):
    store = TimeSeriesStore(db_path=tmp_path / "ts.sqlite3", retention_days=7)
    now = datetime.now(timezone.utc)
    store.record("kp_index", 1.0, timestamp=now - timedelta(days=30))
    store.record("kp_index", 2.0, timestamp=now)

    deleted = store.prune()
    assert deleted == 1
    remaining = store.query("kp_index")
    assert len(remaining) == 1
    assert remaining[0].value == 2.0
