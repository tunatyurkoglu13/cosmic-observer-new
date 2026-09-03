"""
cv/anomaly_train.py — Train SpaceAutoencoder on real NASA space imagery.

Training data: real photos from NASA's own public Images API
(images.nasa.gov, keyless — see cv.dataset_builder.fetch_nasa_images),
queried for genuinely nominal, non-anomalous space/spacecraft scenes.
Nothing here is synthetic or fabricated as "normal" — these are real
NASA-published photographs.

Threshold: mean + 2*std of reconstruction error on a HELD-OUT slice of
the same real normal images (never seen during training) — the project
spec's own rule. Computed once after training and saved alongside the
weights (see cv.anomaly.AnomalyDetector.save/.load), not baked into the
model, so it can be re-tuned later (e.g. against cv.anomaly_evaluate's
ROC analysis) without retraining.

Run: ./.venv/bin/python -m cv.anomaly_train
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from cv.anomaly import (
    AnomalyDetector,
    SpaceAutoencoder,
    preprocess_frame_for_autoencoder,
    reconstruction_error,
    train_autoencoder,
)
from cv.dataset_builder import fetch_nasa_images

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
WEIGHTS_PATH = MODEL_DIR / "anomaly_autoencoder.pt"
META_PATH = MODEL_DIR / "anomaly_meta.json"
TRAINING_LOG_PATH = MODEL_DIR / "anomaly_training_log.json"
IMAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "cv_training" / "anomaly"

# Real NASA Images API search queries chosen to return genuinely nominal
# ("normal") spacecraft/space imagery — not filtered/curated beyond the
# search terms themselves, so this is honestly whatever NASA's own
# archive returns for these queries, warts and all.
NORMAL_QUERIES = [
    "International Space Station",
    "satellite orbit",
    "Earth from space",
    "astronaut spacewalk",
    "solar panel spacecraft",
]

THRESHOLD_SIGMA = 2.0  # project spec: threshold = mean + 2*std
HOLDOUT_FRACTION = 0.2  # shared with cv.anomaly_evaluate so both scripts derive the exact same holdout split


def load_and_preprocess_images(paths: list[Path]) -> np.ndarray:
    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue  # a handful of NASA archive entries are corrupt/non-image downloads — skip, don't abort the batch
        images.append(preprocess_frame_for_autoencoder(img))
    return np.stack(images) if images else np.zeros((0, 64, 64), dtype=np.float32)


def main(images_per_query: int = 60, epochs: int = 60, holdout_fraction: float = HOLDOUT_FRACTION) -> None:
    print("Fetching real NASA imagery for training (images.nasa.gov, keyless)...")
    all_paths: list[Path] = []
    for query in NORMAL_QUERIES:
        try:
            paths = fetch_nasa_images(query, limit=images_per_query, output_dir=IMAGE_DIR)
            print(f"  '{query}': {len(paths)} real images")
            all_paths.extend(paths)
        except Exception as e:
            print(f"  '{query}' FAILED: {e}")

    if len(all_paths) < 20:
        raise RuntimeError(
            f"Only {len(all_paths)} real images fetched — not enough to train a meaningful "
            "autoencoder. Check network connectivity / NASA Images API status."
        )

    images = load_and_preprocess_images(all_paths)
    print(f"Loaded {len(images)} real images, preprocessed to {SpaceAutoencoder.INPUT_SIZE}x{SpaceAutoencoder.INPUT_SIZE} grayscale.")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(images))
    n_holdout = max(4, int(len(images) * holdout_fraction))
    holdout_idx, train_idx = idx[:n_holdout], idx[n_holdout:]
    train_images, holdout_images = images[train_idx], images[holdout_idx]
    print(f"Train: {len(train_images)} images, held-out (for threshold + eval): {len(holdout_images)} images")

    model = SpaceAutoencoder()
    losses = train_autoencoder(model, train_images, epochs=epochs, batch_size=16, lr=1e-3)
    print(f"Training loss: {losses[0]:.5f} -> {losses[-1]:.5f} over {epochs} epochs")

    holdout_errors = reconstruction_error(model, holdout_images)
    threshold = float(holdout_errors.mean() + THRESHOLD_SIGMA * holdout_errors.std())
    print(f"Held-out normal reconstruction error: mean={holdout_errors.mean():.5f}, std={holdout_errors.std():.5f}")
    print(f"Anomaly threshold (mean + {THRESHOLD_SIGMA}*std): {threshold:.5f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    detector = AnomalyDetector(model=model, threshold=threshold)
    detector.save(WEIGHTS_PATH, META_PATH)
    TRAINING_LOG_PATH.write_text(json.dumps({
        "epoch_losses": losses,
        "n_train_images": int(len(train_images)),
        "n_holdout_images": int(len(holdout_images)),
        "holdout_error_mean": float(holdout_errors.mean()),
        "holdout_error_std": float(holdout_errors.std()),
        "threshold": threshold,
        "threshold_sigma": THRESHOLD_SIGMA,
        "queries": NORMAL_QUERIES,
    }, indent=2))
    print(f"Saved: {WEIGHTS_PATH}, {META_PATH}, {TRAINING_LOG_PATH}")


if __name__ == "__main__":
    main()
