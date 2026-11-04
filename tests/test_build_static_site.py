import base64
import hashlib
import re
from datetime import datetime, timedelta, timezone

import pytest

from core.timeseries_store import TimeSeriesStore
from scripts import build_static_site as build


def test_trajectory_key_matches_adapter_rule():
    # pages/static-adapter.js derives the same key with .replace(/\D/g, '').
    assert build.trajectory_key("2026-11-04T17:14:41.333706+00:00") == "202611041714413337060000"


def test_extend_history_only_adds_points_outside_stored_range(tmp_path):
    store = TimeSeriesStore(tmp_path / "ts.db")
    t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
    hour = timedelta(hours=1)
    store.record("kp_index", 2.0, timestamp=t0 + 5 * hour)
    store.record("kp_index", 3.0, timestamp=t0 + 10 * hour)

    points = [(t0 + i * hour, float(i)) for i in range(0, 13, 2)]  # 0, 2, ..., 12 h
    added = build._extend_history(store, "kp_index", points)

    stamps = [s.timestamp for s in store.query("kp_index")]
    assert added == 4  # 0, 2, 4 before the stored range; 12 after; 6, 8, 10 overlap
    assert stamps == sorted(stamps)
    assert len(stamps) == 6


def test_build_pages_injects_adapter_csp_and_rewrites_links(tmp_path):
    build.build_pages(tmp_path, "https://example.github.io/repo/", "https://github.com/example/repo")

    index = (tmp_path / "index.html").read_text()
    assert '<script src="static-adapter.js"></script>' in index
    assert '<meta name="co-source" content="https://github.com/example/repo" />' in index
    assert 'content="https://example.github.io/repo/og-image.png"' in index
    assert 'href="cv.html"' in index and 'href="/cv"' not in index

    # The CSP must allow exactly the inline import map, by hash.
    importmap = re.search(r'<script type="importmap">(.*?)</script>', index, re.S).group(1)
    digest = base64.b64encode(hashlib.sha256(importmap.encode()).digest()).decode()
    assert f"'sha256-{digest}'" in index
    assert "'unsafe-inline' https://unpkg.com" not in index  # scripts are never blanket-allowed

    assert "SAMPLED EVERY 2 MIN" not in (tmp_path / "timeline.html").read_text()
    assert '<base href="/repo/" />' in (tmp_path / "404.html").read_text()


def test_assert_no_secrets_blocks_leaked_key(tmp_path, monkeypatch):
    monkeypatch.setenv("NASA_API_KEY", "s3cr3t-key-value")
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "ok.json").write_text('{"img": "https://epic.gsfc.nasa.gov/x.png"}')
    build.assert_no_secrets(tmp_path)

    (tmp_path / "api" / "leak.json").write_text('{"img": "https://api.nasa.gov/x?api_key=s3cr3t-key-value"}')
    with pytest.raises(SystemExit, match="NASA_API_KEY"):
        build.assert_no_secrets(tmp_path)


def test_assert_no_secrets_ignores_demo_key(tmp_path, monkeypatch):
    monkeypatch.setenv("NASA_API_KEY", "DEMO_KEY")
    (tmp_path / "x.json").write_text("DEMO_KEY")
    build.assert_no_secrets(tmp_path)
