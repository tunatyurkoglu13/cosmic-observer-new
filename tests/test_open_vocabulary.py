import numpy as np
import pytest

from cv.detector import Detection
from cv.open_vocabulary import DEFAULT_SPACE_QUERIES, OpenVocabularyDetector

NASA_GRACE_FO_IMAGE_URL = "https://images-assets.nasa.gov/image/PIA22443/PIA22443~small.jpg"


def test_default_space_queries_are_nonempty_strings():
    assert len(DEFAULT_SPACE_QUERIES) > 0
    assert all(isinstance(q, str) and q for q in DEFAULT_SPACE_QUERIES)


@pytest.mark.network
@pytest.mark.slow
def test_owlvit_detects_real_people_in_real_nasa_photo(tmp_path):
    """
    Real end-to-end zero-shot check (not mocked): downloads a genuine
    NASA photo of the GRACE-FO satellites being assembled in a cleanroom
    (four people visible in cleanroom suits, ground-truth known from the
    actual image), and verifies OWL-ViT's zero-shot 'person' query finds
    real matches at plausible image locations.
    """
    import cv2
    import requests

    try:
        resp = requests.get(NASA_GRACE_FO_IMAGE_URL, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        pytest.skip(f"Could not download real test image (external condition): {e}")

    path = tmp_path / "grace_fo.jpg"
    path.write_bytes(resp.content)
    image = cv2.imread(str(path))
    assert image is not None

    detector = OpenVocabularyDetector()
    detections = detector.detect(image, text_queries=["satellite", "solar panel", "person"], confidence_threshold=0.05)

    assert len(detections) > 0
    assert all(isinstance(d, Detection) for d in detections)

    person_detections = [d for d in detections if d.class_name == "person"]
    assert len(person_detections) >= 2  # the real photo has 4 people; expect to find at least a couple

    for d in detections:
        x1, y1, x2, y2 = d.box_xyxy
        assert 0 <= x1 < x2 <= image.shape[1]
        assert 0 <= y1 < y2 <= image.shape[0]
        assert 0.0 <= d.confidence <= 1.0


@pytest.mark.network
@pytest.mark.slow
def test_owlvit_detections_sorted_by_descending_confidence(tmp_path):
    import cv2
    import requests

    try:
        resp = requests.get(NASA_GRACE_FO_IMAGE_URL, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        pytest.skip(f"Could not download real test image (external condition): {e}")

    path = tmp_path / "grace_fo.jpg"
    path.write_bytes(resp.content)
    image = cv2.imread(str(path))

    detector = OpenVocabularyDetector()
    detections = detector.detect(image, text_queries=["satellite", "person"], confidence_threshold=0.02)

    confidences = [d.confidence for d in detections]
    assert confidences == sorted(confidences, reverse=True)
