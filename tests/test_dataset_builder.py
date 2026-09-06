import numpy as np
import pytest

import cv.dataset_builder as dataset_builder
from cv.dataset_builder import auto_label_with_zero_shot, fetch_nasa_images
from cv.detector import Detection


@pytest.mark.network
@pytest.mark.slow
def test_fetch_nasa_images_downloads_real_images(tmp_path):
    """Real end-to-end check: NASA's public Images API is keyless and should return real, downloadable photos."""
    paths = fetch_nasa_images("satellite", limit=3, output_dir=tmp_path)

    assert len(paths) > 0
    assert len(paths) <= 3
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
        assert p.suffix == ".jpg"
        # Should be saved into the YOLO-expected images/train/ layout.
        assert p.parent == tmp_path / "images" / "train"


@pytest.mark.network
@pytest.mark.slow
def test_auto_label_with_zero_shot_produces_yolo_format_labels(tmp_path):
    """
    Full real pipeline: fetch real NASA images, then auto-label them with
    the real zero-shot detector, and verify the resulting label files are
    valid YOLO format (normalized floats in [0, 1]).
    """
    image_paths = fetch_nasa_images("satellite", limit=2, output_dir=tmp_path)
    if not image_paths:
        pytest.skip("NASA Images API returned no results (external condition)")

    label_paths = auto_label_with_zero_shot(image_paths, output_dir=tmp_path)

    assert len(label_paths) == len(image_paths)
    for label_path in label_paths:
        assert label_path.parent == tmp_path / "labels" / "train"
        assert label_path.exists()

        content = label_path.read_text().strip()
        if not content:
            continue  # a valid outcome: zero-shot found nothing above threshold in that image

        for line in content.splitlines():
            parts = line.split()
            assert len(parts) == 5
            class_id = int(parts[0])
            assert class_id >= 0
            x, y, w, h = (float(v) for v in parts[1:])
            for v in (x, y, w, h):
                assert 0.0 <= v <= 1.0


class _StubZeroShotDetector:
    """Stands in for OpenVocabularyDetector so this test exercises
    query_class_map's remapping logic without a real model download."""

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image_bgr, text_queries=None, confidence_threshold=0.1):
        # Always "detect" the first query, as if OWL-ViT matched it.
        return [Detection(class_name=text_queries[0], confidence=0.9, box_xyxy=(10.0, 10.0, 50.0, 50.0))]


def test_auto_label_with_zero_shot_applies_query_class_map(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_builder, "OpenVocabularyDetector", _StubZeroShotDetector)

    import cv2
    image_path = tmp_path / "images" / "train" / "fake.jpg"
    image_path.parent.mkdir(parents=True)
    cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))

    label_paths = auto_label_with_zero_shot(
        [image_path],
        text_queries=["a satellite in orbit"],
        output_dir=tmp_path,
        query_class_map={"a satellite in orbit": "satellite"},
    )

    assert len(label_paths) == 1
    line = label_paths[0].read_text().strip()
    class_id = int(line.split()[0])
    from cv.detector import SPACE_CLASSES
    assert SPACE_CLASSES[class_id] == "satellite"
