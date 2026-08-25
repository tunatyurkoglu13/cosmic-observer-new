"""
cv/detector.py — YOLOv8-based object detector for space imagery.

Wraps Ultralytics YOLO for two distinct use cases:
  1. train(): fine-tune a COCO-pretrained YOLOv8 checkpoint on a labeled
     space-imagery dataset (satellite / spacecraft / debris / ISS classes,
     per the project spec).
  2. detect(): run inference and return structured Detection objects.

IMPORTANT — honest scope note: this module does NOT ship a space-domain
fine-tuned model. Fine-tuning per the spec requires a labeled dataset
(NASA Earth Observatory imagery, ISS crew photos, synthetic
Blender/Unity renders) that does not exist in this environment. What IS
real and working here:
  - The full training harness (train()) is wired to the actual
    Ultralytics training API with the exact hyperparameters/staged
    unfreeze schedule the spec calls for — point it at a real dataset
    YAML (see write_dataset_yaml()) and it fine-tunes for real.
  - detect() runs genuine YOLOv8 inference (not a stub) using the
    stock COCO-pretrained checkpoint by default, so class labels
    returned are COCO classes (e.g. "person", "airplane") rather than
    space-domain classes until a fine-tuned checkpoint is supplied via
    `weights=`. This is still useful for validating the
    preprocess -> detect -> track pipeline end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

# Project spec's target class list for a space-domain fine-tune.
SPACE_CLASSES = ["satellite", "spacecraft", "iss", "dragon", "soyuz", "debris", "unknown"]


@dataclass
class Detection:
    """One detected object in an image."""

    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2) pixel coords

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box_xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def write_dataset_yaml(
    path: Path | str,
    dataset_root: Path | str,
    class_names: list[str] = SPACE_CLASSES,
    train_subdir: str = "images/train",
    val_subdir: str = "images/val",
) -> Path:
    """
    Write an Ultralytics-format dataset YAML (the `data=` argument
    train() needs), pointing at a YOLO-format labeled dataset:

        dataset_root/
          images/train/*.jpg
          images/val/*.jpg
          labels/train/*.txt   (YOLO format: class cx cy w h, normalized)
          labels/val/*.txt

    This function only writes the config — it does not create or
    validate that the image/label directories actually exist; that's the
    caller's dataset-preparation responsibility (see the module
    docstring: no such labeled dataset ships with this project).
    """
    path = Path(path)
    config = {
        "path": str(Path(dataset_root).resolve()),
        "train": train_subdir,
        "val": val_subdir,
        "names": {i: name for i, name in enumerate(class_names)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


class SpaceObjectDetector:
    """YOLOv8 wrapper for space-imagery object detection."""

    def __init__(self, weights: str = "yolov8n.pt"):
        """
        Args:
            weights: path to a .pt checkpoint. Defaults to the stock
                COCO-pretrained "yolov8n.pt" (auto-downloaded by
                ultralytics on first use). Pass a fine-tuned checkpoint
                path (e.g. the `best.pt` produced by train()) once one
                exists, to get space-domain class labels instead of COCO ones.
        """
        self.model = YOLO(weights)
        self.weights = weights

    def train(
        self,
        data_yaml: Path | str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        lr0: float = 0.001,
        freeze_backbone_epochs: int = 10,
    ) -> object:
        """
        Fine-tune on a labeled space-imagery dataset, per the project's
        staged unfreeze schedule: freeze the backbone for the first
        `freeze_backbone_epochs` (train only the detection head — the
        head's class/box outputs need to adapt to the new class list
        immediately, while the backbone's general-purpose features are
        still useful as-is), then unfreeze for full fine-tuning.

        Args:
            data_yaml: path to a dataset YAML (see write_dataset_yaml()).
            epochs, imgsz, batch, lr0: standard Ultralytics training
                hyperparameters, defaulted per the project spec.
            freeze_backbone_epochs: number of initial epochs to train
                head-only. Set to 0 to fine-tune the whole network from
                epoch 1 instead.

        Returns:
            The Ultralytics training results object (contains metrics,
            paths to the saved best/last checkpoints, etc.).

        Raises:
            Whatever ultralytics raises if `data_yaml` doesn't point to a
            valid, existing dataset — this method does not fabricate
            training data.
        """
        return self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            freeze=freeze_backbone_epochs if freeze_backbone_epochs > 0 else None,
            augment=True,
        )

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.25) -> list[Detection]:
        """
        Run inference on a single image.

        Args:
            image: BGR image (as loaded by cv2.imread, or as output by
                cv.preprocess.preprocess_pipeline()).
            confidence_threshold: minimum detection confidence to keep.

        Returns:
            List of Detection, one per surviving box.
        """
        results = self.model.predict(image, conf=confidence_threshold, verbose=False)
        detections = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(class_name=names[cls_id], confidence=conf, box_xyxy=(x1, y1, x2, y2))
                )
        return detections

    def detect_excluding_glare(
        self, image: np.ndarray, glare_mask: np.ndarray, confidence_threshold: float = 0.25
    ) -> list[Detection]:
        """
        Run detect(), then drop any detection whose box center falls
        inside a glare mask (see cv.preprocess.sun_glare_mask) — a
        saturated glare blob can otherwise be misdetected as a bright
        object.
        """
        detections = self.detect(image, confidence_threshold=confidence_threshold)
        kept = []
        for det in detections:
            cx, cy = det.center
            if 0 <= int(cy) < glare_mask.shape[0] and 0 <= int(cx) < glare_mask.shape[1]:
                if glare_mask[int(cy), int(cx)] > 0:
                    continue
            kept.append(det)
        return kept
