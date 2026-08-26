import time

import numpy as np

from cv.detector import Detection
from cv.hud import (
    CYAN,
    FrameMetrics,
    draw_hud,
    draw_label,
    draw_reticle,
    draw_scanlines,
    draw_status_bar,
)


def _blank_frame(w=320, h=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_frame_metrics_update_computes_fps_and_confidence():
    metrics = FrameMetrics()
    metrics._last_frame_time = time.time() - 0.1  # simulate 100ms since last frame -> ~10 fps
    dets = [
        Detection(class_name="satellite", confidence=0.8, box_xyxy=(0, 0, 10, 10)),
        Detection(class_name="debris", confidence=0.6, box_xyxy=(0, 0, 10, 10)),
    ]
    metrics.update(dets)

    assert metrics.detection_count == 2
    assert abs(metrics.avg_confidence - 0.7) < 1e-9
    assert 5 < metrics.fps < 20  # loose bound, timing-based
    assert metrics.frame_index == 1


def test_frame_metrics_empty_detections_zero_confidence():
    metrics = FrameMetrics()
    metrics.update([])
    assert metrics.detection_count == 0
    assert metrics.avg_confidence == 0.0


def test_draw_reticle_modifies_frame_pixels():
    frame = _blank_frame()
    before = frame.copy()
    draw_reticle(frame, (50, 50, 150, 150), CYAN)
    assert not np.array_equal(frame, before)


def test_draw_reticle_handles_tiny_box_without_error():
    frame = _blank_frame()
    draw_reticle(frame, (10, 10, 14, 14), CYAN)  # smaller than default corner_len


def test_draw_label_modifies_frame_pixels():
    frame = _blank_frame()
    before = frame.copy()
    draw_label(frame, (50, 50, 150, 150), "SATELLITE 0.91", CYAN)
    assert not np.array_equal(frame, before)


def test_draw_status_bar_modifies_top_region():
    frame = _blank_frame()
    before = frame.copy()
    metrics = FrameMetrics()
    metrics.update([])
    draw_status_bar(frame, metrics, "SAMPLE")
    assert not np.array_equal(frame[:28], before[:28])


def test_draw_scanlines_modifies_frame():
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)  # mid-gray so darkening is visible
    before = frame.copy()
    draw_scanlines(frame)
    assert not np.array_equal(frame, before)


def test_draw_hud_full_pipeline_returns_same_array_annotated():
    frame = _blank_frame()
    metrics = FrameMetrics()
    dets = [Detection(class_name="satellite", confidence=0.85, box_xyxy=(40, 40, 120, 120))]
    metrics.update(dets)

    result = draw_hud(frame, dets, metrics, source_label="SAMPLE")
    assert result is frame  # annotated in place
    assert frame.sum() > 0  # something was drawn
