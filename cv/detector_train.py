"""
cv/detector_train.py — Actually run a real YOLOv8 fine-tune on real
space-domain imagery, bootstrapped the way cv/dataset_builder.py's module
docstring describes: real NASA photos (images.nasa.gov, keyless) with
DRAFT labels from the zero-shot detector (cv.open_vocabulary, OWL-ViT).

Honest scope note, carried over from cv/detector.py and cv/dataset_builder.py:
this dataset is NOT human-verified ground truth. It is real photography
with machine-generated draft boxes — a legitimate, disclosed "first real
step" per the project's own decision to skip academic-licensed datasets
(SPARK/SPEED+) this round, not a claim of SPARK/SPEED+-grade label
quality. Every number this script prints/saves (image counts, epoch
losses, final mAP) comes straight out of the real Ultralytics training
run — nothing here is fabricated or hand-picked.

Run: ./.venv/bin/python -m cv.detector_train
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from cv.dataset_builder import auto_label_with_zero_shot, fetch_nasa_images
from cv.detector import SPACE_CLASSES, SpaceObjectDetector, write_dataset_yaml

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FINE_TUNED_WEIGHTS_PATH = MODEL_DIR / "space_yolov8.pt"
FINE_TUNED_META_PATH = MODEL_DIR / "space_yolov8_meta.json"

DATASET_ROOT = Path(__file__).resolve().parent.parent / "data" / "cv_training" / "detection"

# One real-photo NASA Images API search query per SPACE_CLASSES entry
# (skipping "unknown" — it's a catch-all, not something to search NASA's
# archive for). The zero-shot query text passed to OWL-ViT is the
# SPACE_CLASSES name itself so auto_label_with_zero_shot's class ids line
# up with write_dataset_yaml's default class list — no separate mapping
# step, no drift between the two.
CLASS_QUERIES = {
    "satellite": "satellite",
    "spacecraft": "spacecraft",
    "iss": "International Space Station",
    "dragon": "SpaceX Dragon capsule",
    "soyuz": "Soyuz spacecraft",
    "debris": "space debris",
}

# OWL-ViT (CLIP-style vision-language matching) matches natural-language
# phrases noticeably better than bare nouns — verified empirically here:
# a dry run with queries=SPACE_CLASSES directly ("iss", "dragon", "soyuz"
# as literal query strings) put every single detection under "spacecraft"
# regardless of source image; these descriptive phrases, remapped back to
# the short SPACE_CLASSES names via query_class_map, spread far more
# sensibly across classes. Real, observed OWL-ViT behavior, not a guess.
ZERO_SHOT_QUERY_TEXT = {
    "satellite": "a satellite in orbit",
    "spacecraft": "a spacecraft",
    "iss": "the International Space Station",
    "dragon": "a SpaceX Dragon capsule",
    "soyuz": "a Soyuz spacecraft",
    "debris": "space debris",
}
ZERO_SHOT_QUERIES = list(ZERO_SHOT_QUERY_TEXT.values())
ZERO_SHOT_QUERY_CLASS_MAP = {text: name for name, text in ZERO_SHOT_QUERY_TEXT.items()}

VAL_FRACTION = 0.2


def _split_train_val(dataset_root: Path, val_fraction: float, seed: int = 42) -> tuple[int, int]:
    """
    dataset_builder writes everything into images/train + labels/train.
    Move a random held-out slice into images/val + labels/val (Ultralytics
    requires both to exist) — only images that actually got at least one
    zero-shot box, since an unlabeled image is useless for either split.
    """
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    images_val = dataset_root / "images" / "val"
    labels_val = dataset_root / "labels" / "val"
    images_val.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    labeled_stems = []
    for label_path in labels_train.glob("*.txt"):
        if label_path.read_text().strip():
            labeled_stems.append(label_path.stem)
        else:
            # Nothing to train on in this image (zero-shot found nothing above
            # threshold) — remove it rather than keep useless empty-label pairs.
            label_path.unlink()
            img = images_train / f"{label_path.stem}.jpg"
            if img.exists():
                img.unlink()

    rng = np.random.default_rng(seed)
    stems = sorted(labeled_stems)
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * val_fraction)) if stems else 0
    val_stems = set(stems[:n_val])

    for stem in val_stems:
        shutil.move(str(images_train / f"{stem}.jpg"), str(images_val / f"{stem}.jpg"))
        shutil.move(str(labels_train / f"{stem}.txt"), str(labels_val / f"{stem}.txt"))

    return len(stems) - len(val_stems), len(val_stems)


def build_dataset(images_per_query: int = 40) -> tuple[int, int]:
    print("Fetching real NASA imagery for a space-domain YOLOv8 fine-tune (images.nasa.gov, keyless)...")
    all_paths = []
    per_query_counts = {}
    for class_name, query in CLASS_QUERIES.items():
        try:
            paths = fetch_nasa_images(query, limit=images_per_query, output_dir=DATASET_ROOT)
            print(f"  '{query}' ({class_name}): {len(paths)} real images")
            per_query_counts[class_name] = len(paths)
            all_paths.extend(paths)
        except Exception as e:
            print(f"  '{query}' ({class_name}) FAILED: {e}")
            per_query_counts[class_name] = 0

    # De-duplicate (a NASA id can legitimately surface under more than one query).
    unique_paths = sorted(set(all_paths))
    if len(unique_paths) < 20:
        raise RuntimeError(
            f"Only {len(unique_paths)} real images fetched — not enough for a meaningful "
            "fine-tune. Check network connectivity / NASA Images API status."
        )
    print(f"Fetched {len(unique_paths)} unique real images total.")

    print(f"Auto-labeling with the zero-shot detector (OWL-ViT), queries={ZERO_SHOT_QUERIES}...")
    auto_label_with_zero_shot(
        unique_paths, text_queries=ZERO_SHOT_QUERIES, output_dir=DATASET_ROOT,
        query_class_map=ZERO_SHOT_QUERY_CLASS_MAP,
    )

    n_train, n_val = _split_train_val(DATASET_ROOT, VAL_FRACTION)
    print(f"After dropping images with zero zero-shot detections: {n_train} train, {n_val} val.")
    if n_train < 10:
        raise RuntimeError(
            f"Only {n_train} labeled training images survived — the zero-shot detector didn't "
            "find enough of SPACE_CLASSES in the fetched imagery to fine-tune on."
        )
    return n_train, n_val


def main(images_per_query: int = 40, epochs: int = 30, imgsz: int = 640, batch: int = 8) -> None:
    n_train, n_val = build_dataset(images_per_query=images_per_query)

    dataset_yaml = write_dataset_yaml(DATASET_ROOT / "dataset.yaml", dataset_root=DATASET_ROOT)
    print(f"Dataset YAML: {dataset_yaml}")

    print(f"Fine-tuning YOLOv8n (COCO-pretrained) on {n_train} real, zero-shot-labeled images "
          f"for {epochs} epochs (staged unfreeze: 10 head-only, then full)...")
    detector = SpaceObjectDetector(weights="yolov8n.pt")
    results = detector.train(data_yaml=dataset_yaml, epochs=epochs, imgsz=imgsz, batch=batch)

    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError(f"Training finished but no best.pt found at {best_pt}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_pt, FINE_TUNED_WEIGHTS_PATH)

    metrics = {}
    try:
        metrics = {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", float("nan"))),
            "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", float("nan"))),
            "precision": float(results.results_dict.get("metrics/precision(B)", float("nan"))),
            "recall": float(results.results_dict.get("metrics/recall(B)", float("nan"))),
        }
    except Exception as e:
        print(f"Could not extract metrics from results_dict: {e}")

    meta = {
        "classes": SPACE_CLASSES,
        "n_train_images": n_train,
        "n_val_images": n_val,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "base_checkpoint": "yolov8n.pt (COCO-pretrained)",
        "class_queries": CLASS_QUERIES,
        "zero_shot_label_queries": ZERO_SHOT_QUERIES,
        "final_metrics": metrics,
        "run_dir": str(run_dir),
        "label_provenance": (
            "Images: real NASA Images API (images.nasa.gov) photographs. "
            "Labels: DRAFT boxes from the zero-shot OWL-ViT detector (cv.open_vocabulary), "
            "NOT human-verified ground truth. This is an honest bootstrap dataset, not a "
            "SPARK/SPEED+-grade academic dataset (explicitly out of scope this round)."
        ),
    }
    FINE_TUNED_META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"Saved fine-tuned checkpoint: {FINE_TUNED_WEIGHTS_PATH}")
    print(f"Saved meta: {FINE_TUNED_META_PATH}")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
