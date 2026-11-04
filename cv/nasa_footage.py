"""
cv/nasa_footage.py — real NASA spacecraft footage as a CV frame source.

Clips come from the NASA Image and Video Library (images.nasa.gov). The
asset manifest is resolved through its public API (keyless):
    https://images-api.nasa.gov/asset/<nasa_id>
and the video is read straight over HTTPS — OpenCV's FFmpeg backend seeks
with range requests, so a two-hour mission recording is never downloaded,
only the frames actually read.

Each clip is a set of curated segments of a longer recording: the parts
where a vehicle is actually in frame (mission coverage also contains control-room
shots and graphics, which the space fine-tune is not meant for). Frames
are sampled every `step_s` seconds of mission time, so a slow docking
approach plays back as a time-lapse.

FootageCapture duck-types the cv2.VideoCapture surface app.py's stream
loop uses (read / set / isOpened / release): reaching the end of the last
segment reads as end-of-file, and `set(CAP_PROP_POS_FRAMES, 0)` rewinds
to the start of the first — so the existing "replay a finished file"
logic loops the clip unchanged. Reads and release() are serialised: the
stream loop reads in a worker thread, and a client disconnecting can
release the capture while a slow network read is still in flight —
FFmpeg aborts the whole process if those overlap.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
import requests

ASSET_API_URL = "https://images-api.nasa.gov/asset/{nasa_id}"
DETAILS_URL = "https://images.nasa.gov/details/{nasa_id}"


@dataclass(frozen=True)
class FootageClip:
    key: str
    nasa_id: str
    title: str
    recorded: str        # date of the event (YYYY-MM-DD)
    segments: tuple[tuple[float, float], ...]  # (start_s, end_s) spans of the source recording, played in order
    step_s: float        # mission time between consecutive output frames
    quality: str = "large"  # asset rendition (large = 1280x720)

    @property
    def timeline(self) -> list[float]:
        """Every source timestamp the clip plays, in order."""
        times = []
        for start, end in self.segments:
            t = start
            while t <= end:
                times.append(t)
                t += self.step_s
        return times

    @property
    def details_url(self) -> str:
        return DETAILS_URL.format(nasa_id=self.nasa_id)


CLIPS: dict[str, FootageClip] = {
    "dragon_crs21": FootageClip(
        key="dragon_crs21",
        # NASA's asset id says CRS-20; the footage (7 Dec 2020) is CRS-21.
        nasa_id="iss064m263421729_Expedition_64_SpaceX_CRS-20_Automated_Docking_201207",
        title="SpaceX Dragon CRS-21 final approach to the ISS",
        recorded="2020-12-07",
        # Two stretches where a vehicle fills the frame (scanned at 4 s
        # intervals with the space fine-tune, ~98% detection): the ISS seen
        # from Dragon's approach camera, then Dragon seen from the ISS. The
        # coverage between cuts to docking-camera screens and mission control.
        segments=((5966.0, 6130.0), (6800.0, 6950.0)),
        step_s=5.0,
    ),
}
DEFAULT_CLIP = "dragon_crs21"


def resolve_clip_url(clip: FootageClip, timeout: int = 15) -> str:
    """The clip's video URL from the library's asset manifest (HTTPS)."""
    resp = requests.get(ASSET_API_URL.format(nasa_id=clip.nasa_id), timeout=timeout)
    resp.raise_for_status()
    suffix = f"~{clip.quality}.mp4"
    for item in resp.json()["collection"]["items"]:
        href = item["href"]
        if href.endswith(suffix):
            return href.replace("http://", "https://", 1)
    raise RuntimeError(f"No {suffix} rendition in the asset manifest for {clip.nasa_id}")


class FootageCapture:
    """A segment of a remote video, sampled every `step_s` seconds, behaving like a cv2.VideoCapture."""

    def __init__(self, url: str, clip: FootageClip):
        self.clip = clip
        self._cap = cv2.VideoCapture(url)
        self._timeline = clip.timeline
        self._index = 0
        self._lock = threading.Lock()
        self._released = False

    def isOpened(self) -> bool:  # noqa: N802 — cv2.VideoCapture's own spelling
        return self._cap.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._released or self._index >= len(self._timeline):
                return False, None
            self._cap.set(cv2.CAP_PROP_POS_MSEC, self._timeline[self._index] * 1000.0)
            ok, frame = self._cap.read()
            self._index += 1
            return ok, frame

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._index = int(value)
            return True
        return self._cap.set(prop, value)

    def release(self) -> None:
        with self._lock:
            self._released = True
            self._cap.release()


def open_clip(key: str = DEFAULT_CLIP, timeout: int = 15) -> FootageCapture:
    clip = CLIPS[key]
    return FootageCapture(resolve_clip_url(clip, timeout=timeout), clip)
