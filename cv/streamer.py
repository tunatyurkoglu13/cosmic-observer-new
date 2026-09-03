"""
cv/streamer.py — Video frame source + processing pipeline for the CV live-view page.

Wires together pieces already built in earlier phases rather than
duplicating them:
    cv.detector.SpaceObjectDetector  — YOLOv8 inference (Phase 6)
    cv.hud                          — retro HUD overlay (this phase)

`FrameProcessor` is the per-frame pipeline: raw frame in, HUD-annotated
frame + detections out. `VideoFileStreamer` is a looping frame source
over an arbitrary video file (an upload, or the bundled sample clip).

Live ISS video ingestion (NASA HDEV / YouTube) is deliberately NOT
implemented here: those endpoints are undocumented, change without
notice, and would need yt-dlp/ffmpeg as extra system dependencies (see
data/iss_feed.py's own docstring on the same instability). File-based
streaming (upload or bundled sample) is the reliable, testable path;
swapping in a live source later only requires a new frame-source class
with the same `frames()` generator interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from cv.anomaly import AnomalyDetector, AnomalyResult
from cv.detector import Detection, SpaceObjectDetector
from cv.hud import FrameMetrics, draw_hud

SAMPLE_VIDEO_PATH = Path(__file__).resolve().parent.parent / "static" / "cv_sample.mp4"

ANOMALY_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "anomaly_autoencoder.pt"
ANOMALY_META_PATH = Path(__file__).resolve().parent.parent / "models" / "anomaly_meta.json"


@dataclass
class ProcessedFrame:
    """One HUD-annotated frame plus the raw detection list behind it (for the side-panel JSON message)."""

    frame_bgr: np.ndarray
    detections: list[Detection]
    metrics: FrameMetrics
    anomaly: AnomalyResult | None = None  # None when no trained model is available — see FrameProcessor.__init__


class FrameProcessor:
    """Runs detection + anomaly scoring + HUD annotation on individual frames."""

    def __init__(self, weights: str = "yolov8n.pt", confidence_threshold: float = 0.35):
        self.detector = SpaceObjectDetector(weights=weights)
        self.confidence_threshold = confidence_threshold
        self.metrics = FrameMetrics()

        # Anomaly detection degrades gracefully rather than crashing the
        # whole live stream if `python -m cv.anomaly_train` hasn't been
        # run yet (or its output was deleted) — genuinely optional, same
        # spirit as this project's resilient-fetch data clients.
        self.anomaly_detector: AnomalyDetector | None = None
        if ANOMALY_WEIGHTS_PATH.exists() and ANOMALY_META_PATH.exists():
            try:
                self.anomaly_detector = AnomalyDetector.load(ANOMALY_WEIGHTS_PATH, ANOMALY_META_PATH)
            except Exception:
                self.anomaly_detector = None

    def process(self, frame_bgr: np.ndarray, source_label: str = "SAMPLE") -> ProcessedFrame:
        """Run detection + anomaly scoring on one frame and return it HUD-annotated (frame is modified in place)."""
        detections = self.detector.detect(frame_bgr, confidence_threshold=self.confidence_threshold)
        self.metrics.update(detections)

        anomaly_result = self.anomaly_detector.score_frame(frame_bgr) if self.anomaly_detector else None

        draw_hud(frame_bgr, detections, self.metrics, source_label=source_label, anomaly=anomaly_result)
        return ProcessedFrame(frame_bgr=frame_bgr, detections=detections, metrics=self.metrics, anomaly=anomaly_result)


class VideoFileStreamer:
    """Loops a video file frame-by-frame indefinitely (restarts at EOF) — the frame source for /ws/cv."""

    def __init__(self, path: Path | str, target_fps: float = 15.0):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        self.target_fps = target_fps
        self.frame_interval_s = 1.0 / target_fps

    def frames(self) -> Iterator[np.ndarray]:
        """Yield BGR frames indefinitely, looping back to the start when the video ends."""
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.path}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Video file appears empty or unreadable: {self.path}")
                yield frame
                time.sleep(self.frame_interval_s)
        finally:
            cap.release()


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR frame as JPEG bytes for WebSocket binary transmission."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def build_sample_video(output_path: Path | str = SAMPLE_VIDEO_PATH, fps: int = 2, repeats: int = 8) -> Path:
    """
    Build a short looping demo video out of Ultralytics' own bundled
    sample images (real photos with real detectable objects — a bus and
    pedestrians, people — so the CV pipeline demo shows genuine
    detections rather than synthetic placeholder content). Panning a
    static image would be more "video-like" but adds complexity for a
    demo clip; alternating a couple of real photos already exercises the
    full pipeline (decode -> detect -> HUD -> encode -> stream) honestly.

    Idempotent: skips rebuilding if the output file already exists.
    """
    output_path = Path(output_path)
    if output_path.exists():
        return output_path

    from ultralytics.utils import ASSETS

    image_paths = [ASSETS / "bus.jpg", ASSETS / "zidane.jpg"]
    images = [cv2.imread(str(p)) for p in image_paths if p.exists()]
    if not images:
        raise RuntimeError("Ultralytics sample assets not found; cannot build sample video")

    h, w = images[0].shape[:2]
    images = [cv2.resize(img, (w, h)) for img in images]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    try:
        for _ in range(repeats):
            for img in images:
                writer.write(img)
    finally:
        writer.release()

    return output_path
