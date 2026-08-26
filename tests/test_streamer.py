import numpy as np
import pytest

from cv.streamer import (
    FrameProcessor,
    VideoFileStreamer,
    build_sample_video,
    encode_jpeg,
)


def test_encode_jpeg_roundtrip_decodable():
    import cv2

    frame = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    jpeg_bytes = encode_jpeg(frame)
    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) > 0

    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == frame.shape


@pytest.mark.slow
def test_build_sample_video_creates_real_playable_file(tmp_path):
    out_path = tmp_path / "sample.mp4"
    result = build_sample_video(output_path=out_path, fps=2, repeats=2)
    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    streamer = VideoFileStreamer(out_path, target_fps=1000)  # fast for the test, not real-time
    gen = streamer.frames()
    frame = next(gen)
    assert frame is not None
    assert frame.ndim == 3


def test_build_sample_video_is_idempotent(tmp_path):
    out_path = tmp_path / "sample.mp4"
    build_sample_video(output_path=out_path, fps=2, repeats=1)
    mtime1 = out_path.stat().st_mtime
    build_sample_video(output_path=out_path, fps=2, repeats=1)
    mtime2 = out_path.stat().st_mtime
    assert mtime1 == mtime2  # second call should skip rebuilding


def test_video_file_streamer_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoFileStreamer(tmp_path / "does_not_exist.mp4")


@pytest.mark.slow
def test_frame_processor_runs_real_detection_and_annotates_frame(tmp_path):
    """
    Full pipeline, real components (not mocked): build the sample video
    from real Ultralytics assets, read a frame, run it through
    FrameProcessor (real YOLOv8 inference + real HUD drawing), and
    confirm genuine detections come back with a visibly annotated frame.
    """
    video_path = build_sample_video(output_path=tmp_path / "sample.mp4", fps=2, repeats=1)
    streamer = VideoFileStreamer(video_path, target_fps=1000)
    frame = next(streamer.frames())
    original = frame.copy()

    processor = FrameProcessor(weights="yolov8n.pt", confidence_threshold=0.25)
    result = processor.process(frame, source_label="TEST")

    assert len(result.detections) > 0
    assert not np.array_equal(frame, original)  # HUD actually drew something
    assert result.metrics.detection_count == len(result.detections)
