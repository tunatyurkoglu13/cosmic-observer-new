import io
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests
from PIL import Image

from core.resilient_fetch import JsonFileFetcher
from data import donki, eonet, nasa_imagery, nasa_library


def _jpeg(pixels: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(buf, format="PNG")  # lossless, so thresholds are exact
    return buf.getvalue()


def _response(payload=None, content=None):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.content = content if content is not None else b"[...]"
    return resp


# --- NASA imagery (GIBS / Trek) ----------------------------------------------

def test_measure_coverage_ignores_polar_night_and_weights_by_area():
    img = np.full((180, 360, 3), 60)
    img[:20] = 0             # lat > 70 N: polar night, outside the judged band
    img[90:, :180] = 0       # southern half of the western hemisphere missing
    coverage = nasa_imagery.measure_coverage(_jpeg(img))
    assert coverage == pytest.approx(0.75, abs=0.01)


def test_measure_coverage_counts_dark_ocean_as_data():
    img = np.full((90, 180, 3), [10, 20, 50])  # VIIRS true-colour ocean
    assert nasa_imagery.measure_coverage(_jpeg(img)) == pytest.approx(1.0)


def test_pick_true_color_date_takes_most_recent_complete_day():
    full = _jpeg(np.full((128, 256, 3), 80))
    half = np.full((128, 256, 3), 80)
    half[:, :128] = 0
    coverages = {1: _jpeg(np.zeros((128, 256, 3))), 2: _jpeg(half), 3: full}
    today = date(2026, 11, 4)

    def fake_get(url, timeout):
        offset = next(o for o in coverages if f"TIME={(today - timedelta(days=o)).isoformat()}" in url)
        return _response(content=coverages[offset])

    with patch("data.nasa_imagery.requests.get", side_effect=fake_get):
        day, coverage = nasa_imagery.pick_true_color_date(today, lookback_days=3)
    assert day == date(2026, 11, 1)
    assert coverage == pytest.approx(1.0)


def test_pick_true_color_date_falls_back_to_best_partial_day():
    today = date(2026, 11, 4)
    images = [np.zeros((128, 256, 3)), np.full((128, 256, 3), 80), np.zeros((128, 256, 3))]
    images[1][:, :64] = 0  # ~75% coverage

    with patch("data.nasa_imagery.requests.get", side_effect=[_response(content=_jpeg(i)) for i in images]):
        day, coverage = nasa_imagery.pick_true_color_date(today, lookback_days=3)
    assert day == date(2026, 11, 2)
    assert coverage == pytest.approx(0.75, abs=0.01)


def test_gibs_url_uses_lon_lat_bbox_order():
    url = nasa_imagery.gibs_map_url(nasa_imagery.NIGHT_LAYER, 4096, 2048)
    assert "VERSION=1.1.1" in url and "BBOX=-180%2C-90%2C180%2C90" in url and "TIME" not in url


def test_trek_surface_describes_a_whole_equirectangular_level():
    moon = nasa_imagery.trek_surface("moon")
    assert (moon["columns"], moon["rows"]) == (8, 4)
    assert moon["tile_url"].endswith("/{z}/{y}/{x}.jpg") and "/Moon/EQ/" in moon["tile_url"]


# --- EONET -----------------------------------------------------------------

def test_eonet_normalizes_storm_track_and_latest_magnitude():
    raw = {
        "id": "EONET_1", "title": "Tropical Cyclone X", "link": "https://eonet/1",
        "categories": [{"id": "severeStorms", "title": "Severe Storms"}],
        "sources": [{"id": "JTWC", "url": "https://jtwc"}],
        "geometry": [
            {"date": "2026-11-03T06:00:00Z", "type": "Point", "coordinates": [80.2, 15.8], "magnitudeValue": 65.0, "magnitudeUnit": "kts"},
            {"date": "2026-11-02T06:00:00Z", "type": "Point", "coordinates": [82.0, 14.0], "magnitudeValue": 45.0, "magnitudeUnit": "kts"},
        ],
    }
    event = eonet.normalize_event(raw)
    assert (event["lat"], event["lon"]) == (15.8, 80.2)
    assert [p["date"][:10] for p in event["track"]] == ["2026-11-02", "2026-11-03"]
    assert (event["magnitude"], event["magnitude_unit"]) == (65.0, "kts")


def test_eonet_polygon_uses_centroid_and_skips_empty_geometry():
    poly = {"id": "E2", "categories": [{"id": "wildfires"}], "geometry": [
        {"date": "2026-11-01T00:00:00Z", "type": "Polygon", "coordinates": [[[10, 0], [12, 0], [12, 2], [10, 2]]]},
    ]}
    event = eonet.normalize_event(poly)
    assert (event["lat"], event["lon"]) == (1.0, 11.0)
    assert eonet.normalize_event({"id": "E3", "geometry": []}) is None


# --- DONKI -----------------------------------------------------------------

@pytest.mark.parametrize("cls, rank", [("X1.0", 4.1), ("M2.4", 3.24), ("C9.9", 2.99), (None, 0.0), ("?", 0.0)])
def test_flare_rank_orders_by_class_then_magnitude(cls, rank):
    assert donki.flare_rank(cls) == pytest.approx(rank)


def test_donki_alert_thresholds():
    assert donki.assess_flare({"class": "X2.1"}) == "critical"
    assert donki.assess_flare({"class": "M1.0"}) == "warning"
    assert donki.assess_flare({"class": "C5.0"}) is None
    assert donki.assess_cme({"earth_arrival": None, "speed_km_s": 2000}) is None
    assert donki.assess_cme({"earth_arrival": "2026-11-05T12:00Z", "speed_km_s": 600}) == "warning"
    assert donki.assess_cme({"earth_arrival": "2026-11-05T12:00Z", "speed_km_s": 1200}) == "critical"
    assert donki.assess_storm({"kp_max": 7.33}) == "critical"
    assert donki.assess_storm({"kp_max": 5.0}) == "warning"
    assert donki.assess_storm({"kp_max": 4.67}) is None


def test_normalize_cme_uses_most_accurate_analysis_and_earliest_arrival():
    raw = {
        "activityID": "2026-11-01T10:00:00-CME-001", "startTime": "2026-11-01T10:00Z", "note": "x" * 400,
        "cmeAnalyses": [
            {"isMostAccurate": False, "speed": 500, "enlilList": [{"estimatedShockArrivalTime": "2026-11-04T09:00Z"}]},
            {"isMostAccurate": True, "speed": 1100, "halfAngle": 40, "type": "O",
             "enlilList": [{"estimatedShockArrivalTime": "2026-11-03T18:00Z"}, {"estimatedShockArrivalTime": None}]},
        ],
    }
    cme = donki.normalize_cme(raw)
    assert cme["speed_km_s"] == 1100 and cme["type"] == "O"
    assert cme["earth_arrival"] == "2026-11-03T18:00Z"
    assert len(cme["note"]) == donki.NOTE_MAX_CHARS + 1 and cme["note"].endswith("…")


def test_donki_sends_key_in_header_never_in_url(monkeypatch):
    monkeypatch.setenv("NASA_API_KEY", "header-only-key")
    with patch("data.donki.requests.get", return_value=_response([], content=b"")) as get:
        assert donki.donki_get("FLR", date(2026, 11, 1), date(2026, 11, 4)) == []
    args, kwargs = get.call_args
    assert kwargs["headers"] == {"X-Api-Key": "header-only-key"}
    assert "header-only-key" not in args[0] and "api_key" not in kwargs["params"]


def test_donki_client_summarizes_events(tmp_path):
    payloads = {
        "FLR": [{"flrID": "f1", "classType": "M1.2", "peakTime": "2026-11-02T01:00Z"},
                {"flrID": "f2", "classType": "X1.1", "peakTime": "2026-11-03T01:00Z"}],
        "CME": [{"activityID": "c1", "startTime": "2026-11-02T00:00Z",
                 "cmeAnalyses": [{"isMostAccurate": True, "speed": 900,
                                  "enlilList": [{"estimatedShockArrivalTime": "2026-11-04T00:00Z"}]}]}],
        "GST": [{"gstID": "g1", "startTime": "2026-11-03T00:00Z", "allKpIndex": [{"kpIndex": 5.33}, {"kpIndex": 6.0}]}],
    }
    with patch("data.donki.donki_get", side_effect=lambda ep, *a: payloads[ep]):
        result = donki.DONKIClient(cache_path=tmp_path / "d.json").fetch_recent(7)
    assert result["summary"] == {
        "flares": 2, "flares_by_class": {"C": 0, "M": 1, "X": 1}, "strongest_flare": "X1.1",
        "cmes": 1, "earth_directed_cmes": 1, "storms": 1, "max_kp": 6.0,
    }
    assert result["flares"][0]["id"] == "f2"  # most recent first


# --- JsonFileFetcher -------------------------------------------------------

class _CountingFetcher(JsonFileFetcher[dict]):
    def __init__(self, path, fail=False):
        super().__init__(path, staleness=timedelta(hours=1))
        self.calls = 0
        self.fail = fail

    def _fetch_live(self, key):
        self.calls += 1
        if self.fail:
            raise requests.ConnectionError("down")
        return {"key": key, "n": self.calls}


def test_json_file_fetcher_caches_then_serves_stale_on_outage(tmp_path):
    path = tmp_path / "cache.json"
    fetcher = _CountingFetcher(path)
    assert fetcher.fetch("a") == {"key": "a", "n": 1}
    assert fetcher.fetch("a") == {"key": "a", "n": 1}  # fresh cache, no second call
    assert fetcher.calls == 1

    down = _CountingFetcher(path, fail=True)
    assert down.fetch("a", force=True) == {"key": "a", "n": 1}  # network down -> last good copy
    assert down._recently_failed("a")
    with pytest.raises(requests.ConnectionError):
        down.fetch("never-cached", force=True)


# --- NASA footage ------------------------------------------------------------

def test_footage_capture_walks_segments_and_rewinds(monkeypatch):
    from cv import nasa_footage

    class FakeCap:
        def __init__(self, url): self.pos = None; self.released = False
        def isOpened(self): return True
        def set(self, prop, value): self.pos = value; return True
        def read(self): return True, self.pos
        def release(self): self.released = True

    monkeypatch.setattr(nasa_footage.cv2, "VideoCapture", FakeCap)
    clip = nasa_footage.FootageClip(key="k", nasa_id="n", title="t", recorded="2020-01-01",
                                    segments=((10.0, 20.0), (100.0, 105.0)), step_s=5.0)
    cap = nasa_footage.FootageCapture("https://x", clip)
    seen = []
    while True:
        ok, pos_ms = cap.read()
        if not ok:
            break
        seen.append(pos_ms / 1000)
    assert seen == [10.0, 15.0, 20.0, 100.0, 105.0]

    cap.set(nasa_footage.cv2.CAP_PROP_POS_FRAMES, 0)  # how app.py loops a finished source
    assert cap.read()[1] / 1000 == 10.0
    cap.release()
    assert cap.read() == (False, None)  # never touches FFmpeg after release


def test_resolve_clip_url_picks_rendition_over_https():
    from cv import nasa_footage
    clip = nasa_footage.CLIPS[nasa_footage.DEFAULT_CLIP]
    manifest = {"collection": {"items": [
        {"href": f"http://images-assets.nasa.gov/video/{clip.nasa_id}/{clip.nasa_id}~orig.mp4"},
        {"href": f"http://images-assets.nasa.gov/video/{clip.nasa_id}/{clip.nasa_id}~large.mp4"},
    ]}}
    with patch("cv.nasa_footage.requests.get", return_value=_response(manifest)):
        url = nasa_footage.resolve_clip_url(clip)
    assert url.startswith("https://") and url.endswith("~large.mp4")




# --- NASA Image and Video Library gallery ----------------------------------

def _library_item(nasa_id, title, created, description="", renditions=("~large.jpg", "~medium.jpg")):
    return {
        "data": [{"nasa_id": nasa_id, "title": title, "date_created": created, "center": "JSC",
                  "description": description}],
        "links": [{"href": f"http://images-assets.nasa.gov/image/{nasa_id}/{nasa_id}{r}"} for r in renditions],
    }


def test_library_item_normalizes_renditions_and_description():
    raw = _library_item("iss1", "Dragon approach", "2026-10-15T00:00:00Z",
                        description='A <a href="https://x">SpaceX</a> Dragon &amp; the ISS.\n\n  Credit: NASA')
    item = nasa_library.normalize_item(raw)
    assert item["image_url"].startswith("https://") and item["image_url"].endswith("~medium.jpg")
    assert item["thumb_url"] == item["image_url"]  # no ~thumb rendition -> falls back to the image
    assert item["description"] == "A SpaceX Dragon & the ISS. Credit: NASA"
    assert item["details_url"] == "https://images.nasa.gov/details/iss1"
    assert nasa_library.normalize_item(_library_item("x", "t", "2026-01-01", renditions=("~orig.tif",))) is None


def test_curate_sorts_newest_first_and_drops_burst_duplicates():
    items = [nasa_library.normalize_item(_library_item(i, t, d)) for i, t, d in [
        ("a", "Launch", "2026-10-01"), ("b", "Dragon approach", "2026-10-15"),
        ("c", "Dragon Approach", "2026-10-15"), ("d", "Spacewalk", "2026-10-20"),
    ]]
    assert [i["nasa_id"] for i in nasa_library.curate(items)] == ["d", "b", "a"]


def test_library_client_rejects_unknown_collection(tmp_path):
    with pytest.raises(ValueError):
        nasa_library.NasaLibraryClient(cache_path=tmp_path / "g.json").fetch_collection("nope")
