"""
cv/hud.py — Retro-futuristic HUD overlay drawing for the CV live-view page.

Draws directly onto video frames with OpenCV (not CSS/canvas overlays),
so the exact same annotated frame that gets JPEG-encoded and streamed to
the browser already contains the targeting reticles/labels — no separate
client-side rendering pass needed, and the overlay is baked into any
saved/screenshotted frame too.

Reuses viz.retro_ui.PALETTE as the single source of truth for the
project's color scheme (rather than hard-coding a second copy of the
palette here), converted from its "#rrggbb" hex strings to OpenCV's BGR
tuple convention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from cv.detector import Detection
from viz.retro_ui import PALETTE


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert '#rrggbb' to an OpenCV BGR tuple (OpenCV's channel order is reversed from RGB)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


CYAN = _hex_to_bgr(PALETTE["cyan"])
MAGENTA = _hex_to_bgr(PALETTE["magenta"])
AMBER = _hex_to_bgr(PALETTE["amber"])
PHOSPHOR_GREEN = _hex_to_bgr(PALETTE["phosphor_green"])
ALERT_RED = _hex_to_bgr("#ff0033")

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class FrameMetrics:
    """Rolling performance/detection metrics for the HUD status bar (project spec: FPS, count, avg confidence)."""

    fps: float = 0.0
    detection_count: int = 0
    avg_confidence: float = 0.0
    frame_index: int = 0

    _last_frame_time: float = field(default_factory=time.time, repr=False)

    def update(self, detections: list[Detection]) -> None:
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        self.fps = 1.0 / dt if dt > 0 else 0.0
        self.detection_count = len(detections)
        self.avg_confidence = float(np.mean([d.confidence for d in detections])) if detections else 0.0
        self.frame_index += 1


def draw_reticle(frame: np.ndarray, box: tuple[float, float, float, float], color: tuple[int, int, int], corner_len: int = 14) -> None:
    """
    Draw a "targeting reticle" around a detection box: short corner
    brackets (rather than a plain rectangle) plus a small crosshair at
    the box center — the classic sci-fi HUD look, and it reads faster at
    a glance than a full outline once many boxes are on screen at once.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    cl = min(corner_len, (x2 - x1) // 2, (y2 - y1) // 2)
    cl = max(cl, 4)

    corners = [
        ((x1, y1), (1, 0), (0, 1)),
        ((x2, y1), (-1, 0), (0, 1)),
        ((x1, y2), (1, 0), (0, -1)),
        ((x2, y2), (-1, 0), (0, -1)),
    ]
    for (cx, cy), (dx, dy), (ex, ey) in corners:
        cv2.line(frame, (cx, cy), (cx + dx * cl, cy + dy * cl), color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx + ex * cl, cy + ey * cl), color, 2, cv2.LINE_AA)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    s = 5
    cv2.line(frame, (cx - s, cy), (cx + s, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - s), (cx, cy + s), color, 1, cv2.LINE_AA)


def draw_label(frame: np.ndarray, box: tuple[float, float, float, float], text: str, color: tuple[int, int, int]) -> None:
    """Draw a small monospace-style tech-readout label above a detection box, e.g. 'PERSON 0.87'."""
    x1, y1, _, _ = (int(v) for v in box)
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    label_y = max(y1 - 6, th + 4)

    cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 6, label_y + 2), (10, 8, 5), -1)
    cv2.putText(frame, text, (x1 + 3, label_y - 2), FONT, 0.45, color, 1, cv2.LINE_AA)


def draw_status_bar(frame: np.ndarray, metrics: FrameMetrics, source_label: str) -> None:
    """Semi-transparent top status bar with system-status text (project spec: '[SYSTEM ONLINE] ...')."""
    h, w = frame.shape[:2]
    bar_h = 28

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (5, 5, 8), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    text = (
        f"[SYSTEM ONLINE] CV PIPELINE ACTIVE | SOURCE: {source_label} | "
        f"FPS: {metrics.fps:4.1f} | TARGETS: {metrics.detection_count} | "
        f"AVG CONF: {metrics.avg_confidence:.2f} | FRAME: {metrics.frame_index}"
    )
    cv2.putText(frame, text, (8, 19), FONT, 0.45, CYAN, 1, cv2.LINE_AA)


def draw_scanlines(frame: np.ndarray, spacing: int = 3, opacity: float = 0.12) -> None:
    """Bake faint horizontal scanlines into the frame (CRT look) — cheap alternative/complement to a CSS overlay."""
    overlay = frame.copy()
    for y in range(0, frame.shape[0], spacing):
        cv2.line(overlay, (0, y), (frame.shape[1], y), (0, 0, 0), 1)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)


def draw_hud(
    frame: np.ndarray,
    detections: list[Detection],
    metrics: FrameMetrics,
    source_label: str = "SAMPLE",
    class_colors: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """
    Draw the full retro HUD onto a frame: per-detection reticle + label,
    top status bar, and scanline texture.

    Args:
        frame: BGR frame (modified in place AND returned for convenience).
        detections: this frame's Detection list (see cv.detector.Detection).
        metrics: FrameMetrics, already .update()-ed for this frame.
        source_label: short string shown in the status bar ("SAMPLE", "UPLOAD", etc.).
        class_colors: optional per-class-name color override; unlisted
            classes cycle through the retro palette by a simple hash so
            different classes are still visually distinguishable.

    Returns:
        The same frame array, annotated in place.
    """
    palette_cycle = [CYAN, MAGENTA, AMBER, PHOSPHOR_GREEN]
    class_colors = class_colors or {}

    for det in detections:
        color = class_colors.get(det.class_name, palette_cycle[hash(det.class_name) % len(palette_cycle)])
        draw_reticle(frame, det.box_xyxy, color)
        draw_label(frame, det.box_xyxy, f"{det.class_name.upper()} {det.confidence:.2f}", color)

    draw_status_bar(frame, metrics, source_label)
    draw_scanlines(frame)
    return frame
