"""
cv/dataset_builder.py — Bootstrap real training images + draft labels for
a future space-domain YOLOv8 fine-tune.

This does NOT replace a real labeled dataset (SPARK, SPEED+, etc. — those
are academic-licensed and deliberately out of scope this round, per
project decision: no licensed dataset or weights trained on one get
committed to this repo). What this DOES provide, honestly:
  1. Real space imagery, for free, no API key: NASA's own Images API
     (images.nasa.gov) — actual photos from actual NASA missions.
  2. Draft YOLO-format labels via the zero-shot detector (cv.open_vocabulary)
     — good enough to seed a human's review-and-correct pass, not good
     enough to train on blindly (zero-shot boxes are noisier and less
     precise than human-verified ground truth).

Nothing this module downloads or generates is committed to the repo —
`data/cv_training/` is gitignored. This is a bootstrap tool for local
experimentation, not a shipped dataset.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import requests

from cv.detector import SPACE_CLASSES
from cv.open_vocabulary import OpenVocabularyDetector

NASA_IMAGES_API_URL = "https://images-api.nasa.gov/search"

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "cv_training"


def fetch_nasa_images(query: str, limit: int = 20, output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """
    Search NASA's public Images API (no key required) and download real
    photos matching a query, for use as fine-tuning candidates.

    Args:
        query: search term (e.g. "satellite", "ISS spacewalk", "solar panel deployment").
        limit: maximum number of images to download.
        output_dir: directory to save images/train/*.jpg into (YOLO's
            expected layout — see cv.detector.write_dataset_yaml()).

    Returns:
        List of paths to the downloaded images.

    Raises:
        requests.RequestException: on a genuine network/API failure —
        this is a bootstrap convenience tool, not a resilient production
        data source, so it doesn't hide failures behind a fallback.
    """
    output_dir = Path(output_dir) / "images" / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    resp = requests.get(NASA_IMAGES_API_URL, params={"q": query, "media_type": "image"}, timeout=15)
    resp.raise_for_status()
    items = resp.json().get("collection", {}).get("items", [])[:limit]

    saved_paths = []
    for item in items:
        links = item.get("links", [])
        if not links:
            continue
        image_url = links[0]["href"]
        nasa_id = item.get("data", [{}])[0].get("nasa_id", "unknown")

        try:
            img_resp = requests.get(image_url, timeout=15)
            img_resp.raise_for_status()
        except requests.RequestException:
            continue  # one bad image shouldn't abort the whole batch

        dest = output_dir / f"{nasa_id}.jpg"
        dest.write_bytes(img_resp.content)
        saved_paths.append(dest)

    return saved_paths


def auto_label_with_zero_shot(
    image_paths: list[Path],
    text_queries: list[str] | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    confidence_threshold: float = 0.1,
) -> list[Path]:
    """
    Generate DRAFT YOLO-format label files for a set of images using the
    zero-shot detector (cv.open_vocabulary) — a starting point for a
    human to review and correct, not ground truth to train on directly.

    YOLO label format: one .txt file per image, one line per object:
        <class_id> <x_center> <y_center> <width> <height>
    all four spatial values normalized to [0, 1] by image width/height.

    Args:
        image_paths: images to label (e.g. from fetch_nasa_images()).
        text_queries: zero-shot query vocabulary; class IDs in the output
            labels correspond to this list's order matched against
            cv.detector.SPACE_CLASSES where possible, falling back to
            appending any query not already in SPACE_CLASSES.
        output_dir: writes label files to <output_dir>/labels/train/.
        confidence_threshold: passed through to the zero-shot detector.

    Returns:
        List of paths to the written label files.
    """
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)

    queries = text_queries or SPACE_CLASSES
    class_list = list(SPACE_CLASSES)
    for q in queries:
        if q not in class_list:
            class_list.append(q)

    detector = OpenVocabularyDetector()
    label_paths = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        detections = detector.detect(image, text_queries=queries, confidence_threshold=confidence_threshold)

        lines = []
        for det in detections:
            class_id = class_list.index(det.class_name)
            x1, y1, x2, y2 = det.box_xyxy
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            box_w = (x2 - x1) / w
            box_h = (y2 - y1) / h
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        label_path = labels_dir / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(lines))
        label_paths.append(label_path)

    return label_paths
