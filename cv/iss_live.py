"""
cv/iss_live.py — Resolve NASA's real, live ISS video stream into a direct
playable URL for cv2.VideoCapture.

NASA broadcasts a continuous "Live High-Definition Views from the
International Space Station" feed on YouTube (an external camera on the
Harmony module, cycling to interior views when the crew is on duty and
back to Earth views on orbit). YouTube's watch page is not itself a
playable video source — it's a player shell around a signed, time-limited
HLS (.m3u8) manifest URL that only `yt-dlp` (or a full browser JS
runtime) can extract. We resolve that URL here and hand it to OpenCV's
FFmpeg backend, which reads HLS natively — no separate video-download
step, no extra system dependency beyond the `yt-dlp` Python package.

This is a real live external feed, not a bundled asset: it can go
offline (NASA does periodic maintenance/signal handovers), and the
resolved URL expires after a while, so callers must be ready to catch
RuntimeError and fall back to another source (see app.py's /ws/cv, which
falls back to the bundled sample clip and tells the client it did so,
rather than silently failing or pretending to be live when it isn't).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import yt_dlp

# NASA's official continuous ISS live-view stream (verified live and
# resolvable via yt-dlp during development; NASA has run some form of
# this continuous feed under this same title format for years).
DEFAULT_ISS_YOUTUBE_URL = "https://www.youtube.com/watch?v=awQzjn72bI0"

# 480p (854x480): ample resolution for YOLO's 640-ish inference input
# while costing far less bandwidth/decode time than 720p for a
# continuously-running background stream.
DEFAULT_FORMAT_ID = "231"

# Resolved HLS URLs are signed with an expiry by YouTube; re-resolve well
# before they can go stale rather than waiting for a read failure.
URL_CACHE_DURATION_S = 600.0


@dataclass
class ResolvedStream:
    """A resolved, directly-playable stream URL plus when it was resolved."""

    url: str
    title: str
    resolved_at: float

    @property
    def age_s(self) -> float:
        return time.time() - self.resolved_at


_cache: ResolvedStream | None = None


def resolve_iss_stream_url(
    youtube_url: str = DEFAULT_ISS_YOUTUBE_URL,
    format_id: str = DEFAULT_FORMAT_ID,
    force: bool = False,
) -> ResolvedStream:
    """
    Resolve the current direct HLS URL for NASA's live ISS YouTube stream.

    Args:
        youtube_url: YouTube watch-page URL of the live stream.
        format_id: yt-dlp format id to select (see DEFAULT_FORMAT_ID;
            use yt-dlp's own `--list-formats` on the URL to see other options).
        force: bypass the cache and re-resolve even if not yet stale.

    Returns:
        ResolvedStream with a direct URL cv2.VideoCapture can open.

    Raises:
        RuntimeError: if the stream isn't currently live, the video ID is
            no longer valid, or resolution otherwise fails (network
            issue, YouTube extractor changes). This is a real external
            dependency, not a bundled asset — callers must handle this
            and fall back to another source rather than crash.
    """
    global _cache
    if not force and _cache is not None and _cache.age_s < URL_CACHE_DURATION_S:
        return _cache

    ydl_opts = {"quiet": True, "no_warnings": True, "format": format_id}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"Could not resolve ISS live stream ({youtube_url}): {e}") from e

    if not info.get("is_live", False):
        raise RuntimeError(f"'{info.get('title', youtube_url)}' is not currently live")
    if "url" not in info:
        raise RuntimeError(f"yt-dlp returned no direct URL for format '{format_id}' — it may no longer exist on this stream")

    _cache = ResolvedStream(url=info["url"], title=info.get("title", "ISS Live"), resolved_at=time.time())
    return _cache


def clear_cache() -> None:
    """Force the next resolve_iss_stream_url() call to re-resolve rather than use the cache."""
    global _cache
    _cache = None
