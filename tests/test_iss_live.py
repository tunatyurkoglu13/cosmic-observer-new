import time
from unittest.mock import patch

import pytest
import yt_dlp

import cv.iss_live as iss_live
from cv.iss_live import ResolvedStream, resolve_iss_stream_url


@pytest.fixture(autouse=True)
def _clear_cache():
    iss_live.clear_cache()
    yield
    iss_live.clear_cache()


def test_resolve_raises_runtime_error_when_not_live():
    fake_info = {"title": "Some Past Broadcast", "is_live": False}
    with patch.object(yt_dlp.YoutubeDL, "extract_info", return_value=fake_info):
        with pytest.raises(RuntimeError, match="not currently live"):
            resolve_iss_stream_url()


def test_resolve_raises_runtime_error_when_no_url_in_info():
    fake_info = {"title": "ISS Live", "is_live": True}  # missing "url"
    with patch.object(yt_dlp.YoutubeDL, "extract_info", return_value=fake_info):
        with pytest.raises(RuntimeError, match="no direct URL"):
            resolve_iss_stream_url()


def test_resolve_raises_runtime_error_on_download_error():
    with patch.object(yt_dlp.YoutubeDL, "extract_info", side_effect=yt_dlp.utils.DownloadError("boom")):
        with pytest.raises(RuntimeError, match="Could not resolve"):
            resolve_iss_stream_url()


def test_resolve_succeeds_and_caches():
    fake_info = {"title": "ISS Live", "is_live": True, "url": "https://example.com/stream.m3u8"}
    with patch.object(yt_dlp.YoutubeDL, "extract_info", return_value=fake_info) as mock_extract:
        result = resolve_iss_stream_url()
        assert isinstance(result, ResolvedStream)
        assert result.url == "https://example.com/stream.m3u8"

        # Second call within the cache window should NOT hit yt-dlp again.
        resolve_iss_stream_url()
        assert mock_extract.call_count == 1


def test_resolve_force_bypasses_cache():
    fake_info = {"title": "ISS Live", "is_live": True, "url": "https://example.com/stream.m3u8"}
    with patch.object(yt_dlp.YoutubeDL, "extract_info", return_value=fake_info) as mock_extract:
        resolve_iss_stream_url()
        resolve_iss_stream_url(force=True)
        assert mock_extract.call_count == 2


def test_resolve_re_resolves_after_cache_expires():
    fake_info = {"title": "ISS Live", "is_live": True, "url": "https://example.com/stream.m3u8"}
    with patch.object(yt_dlp.YoutubeDL, "extract_info", return_value=fake_info) as mock_extract:
        resolve_iss_stream_url()
        iss_live._cache.resolved_at = time.time() - iss_live.URL_CACHE_DURATION_S - 1
        resolve_iss_stream_url()
        assert mock_extract.call_count == 2


@pytest.mark.network
@pytest.mark.slow
def test_resolve_real_nasa_stream_is_live_and_playable():
    """
    Real end-to-end check against NASA's actual live stream (not mocked):
    resolves a genuine direct URL and confirms OpenCV can open and read a
    frame from it. If NASA's feed happens to be down for maintenance at
    test time, that's an external condition, not a bug — skip rather than fail.
    """
    try:
        stream = resolve_iss_stream_url()
    except RuntimeError as e:
        pytest.skip(f"ISS live stream not available right now (external condition): {e}")

    assert stream.url.startswith("http")

    import cv2

    cap = cv2.VideoCapture(stream.url)
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok
        assert frame is not None
        assert frame.ndim == 3
    finally:
        cap.release()
