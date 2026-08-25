import numpy as np
import pytest

from cv.detector import SPACE_CLASSES, Detection, SpaceObjectDetector, write_dataset_yaml


def test_detection_center_and_area():
    det = Detection(class_name="satellite", confidence=0.9, box_xyxy=(10.0, 20.0, 30.0, 60.0))
    assert det.center == (20.0, 40.0)
    assert det.area == 20.0 * 40.0


def test_write_dataset_yaml_structure(tmp_path):
    yaml_path = tmp_path / "dataset.yaml"
    write_dataset_yaml(yaml_path, dataset_root=tmp_path / "data")

    assert yaml_path.exists()
    import yaml as pyyaml
    with open(yaml_path) as f:
        config = pyyaml.safe_load(f)

    assert config["train"] == "images/train"
    assert config["val"] == "images/val"
    assert config["names"][0] == SPACE_CLASSES[0]
    assert len(config["names"]) == len(SPACE_CLASSES)


@pytest.mark.slow
def test_detector_runs_real_inference_on_bundled_sample_image():
    """
    Verifies the full YOLOv8 inference path is genuinely wired up (not a
    stub) by running it on Ultralytics' own bundled sample asset. Uses
    the stock COCO-pretrained checkpoint, so class labels are COCO
    classes, not space-domain ones (see module docstring for why).
    """
    from ultralytics.utils import ASSETS

    detector = SpaceObjectDetector(weights="yolov8n.pt")
    image_path = ASSETS / "bus.jpg"
    assert image_path.exists()

    import cv2
    image = cv2.imread(str(image_path))
    detections = detector.detect(image, confidence_threshold=0.25)

    assert isinstance(detections, list)
    assert len(detections) > 0
    for det in detections:
        assert 0.0 <= det.confidence <= 1.0
        x1, y1, x2, y2 = det.box_xyxy
        assert x2 > x1 and y2 > y1


@pytest.mark.slow
def test_detect_excluding_glare_drops_detections_in_masked_region():
    detector = SpaceObjectDetector(weights="yolov8n.pt")
    from ultralytics.utils import ASSETS
    import cv2

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    all_detections = detector.detect(image, confidence_threshold=0.25)
    assert len(all_detections) > 0

    # Mask out the entire image -> every detection's center falls inside the mask.
    full_glare_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    filtered = detector.detect_excluding_glare(image, full_glare_mask, confidence_threshold=0.25)
    assert filtered == []
