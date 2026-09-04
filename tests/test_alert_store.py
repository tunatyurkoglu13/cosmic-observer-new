import pytest

from core.alert_store import AlertStore


def test_record_and_query_roundtrip(tmp_path):
    store = AlertStore(db_path=tmp_path / "alerts.sqlite3")
    event = store.record(
        category="conjunction", severity="critical", title="Test conjunction",
        description="A test event", metadata={"foo": "bar"}, dedup_key="test:1",
    )
    assert event is not None
    assert event.id is not None

    alerts = store.query()
    assert len(alerts) == 1
    assert alerts[0].title == "Test conjunction"
    assert alerts[0].metadata == {"foo": "bar"}
    assert alerts[0].acknowledged is False


def test_record_rejects_unknown_severity(tmp_path):
    store = AlertStore(db_path=tmp_path / "alerts.sqlite3")
    with pytest.raises(ValueError, match="Unknown severity"):
        store.record(category="conjunction", severity="apocalyptic", title="x", description="x")


def test_record_dedup_suppresses_within_cooldown(tmp_path):
    store = AlertStore(db_path=tmp_path / "alerts.sqlite3")
    first = store.record(
        category="conjunction", severity="warning", title="Same event",
        description="first", dedup_key="dup:1", cooldown_minutes=60,
    )
    second = store.record(
        category="conjunction", severity="warning", title="Same event",
        description="second (should be suppressed)", dedup_key="dup:1", cooldown_minutes=60,
    )
    assert first is not None
    assert second is None
    assert len(store.query()) == 1


def test_record_allows_new_event_after_cooldown_key_differs(tmp_path):
    store = AlertStore(db_path=tmp_path / "alerts.sqlite3")
    store.record(category="conjunction", severity="warning", title="A", description="a", dedup_key="dup:A", cooldown_minutes=60)
    second = store.record(category="conjunction", severity="warning", title="B", description="b", dedup_key="dup:B", cooldown_minutes=60)
    assert second is not None
    assert len(store.query()) == 2


def test_record_allows_new_event_with_zero_cooldown(tmp_path):
    store = AlertStore(db_path=tmp_path / "alerts.sqlite3")
    store.record(category="anomaly", severity="info", title="A", description="a", dedup_key="same", cooldown_minutes=0)
    second = store.record(category="anomaly", severity="info", title="A", description="a again", dedup_key="same", cooldown_minutes=0)
    assert second is not None
    assert len(store.query()) == 2


def test_query_filters_by_category(tmp_path):
    store = AlertStore(db_path=tmp_path / "alerts.sqlite3")
    store.record(category="conjunction", severity="info", title="C", description="c", dedup_key="c1")
    store.record(category="anomaly", severity="info", title="A", description="a", dedup_key="a1")

    conjunctions = store.query(category="conjunction")
    assert len(conjunctions) == 1
    assert conjunctions[0].category == "conjunction"


def test_query_filters_unacknowledged_only(tmp_path):
    store = AlertStore(db_path=tmp_path / "alerts.sqlite3")
    e1 = store.record(category="anomaly", severity="info", title="A", description="a", dedup_key="a1")
    store.record(category="anomaly", severity="info", title="B", description="b", dedup_key="b1")
    store.acknowledge(e1.id)

    unacked = store.query(unacknowledged_only=True)
    assert len(unacked) == 1
    assert unacked[0].title == "B"


def test_acknowledge_returns_false_for_unknown_id(tmp_path):
    store = AlertStore(db_path=tmp_path / "alerts.sqlite3")
    assert store.acknowledge(999) is False


def test_acknowledge_marks_alert_acknowledged(tmp_path):
    store = AlertStore(db_path=tmp_path / "alerts.sqlite3")
    event = store.record(category="anomaly", severity="info", title="A", description="a", dedup_key="a1")
    assert store.acknowledge(event.id) is True

    fetched = store.get(event.id)
    assert fetched.acknowledged is True


def test_count_unacknowledged(tmp_path):
    store = AlertStore(db_path=tmp_path / "alerts.sqlite3")
    e1 = store.record(category="anomaly", severity="info", title="A", description="a", dedup_key="a1")
    store.record(category="anomaly", severity="info", title="B", description="b", dedup_key="b1")
    assert store.count_unacknowledged() == 2

    store.acknowledge(e1.id)
    assert store.count_unacknowledged() == 1


def test_get_returns_none_for_unknown_id(tmp_path):
    store = AlertStore(db_path=tmp_path / "alerts.sqlite3")
    assert store.get(999) is None
