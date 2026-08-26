"""
cv/open_vocabulary.py — Zero-shot, open-vocabulary object detection (OWL-ViT).

YOLOv8 (cv/detector.py) only recognizes its fixed training classes (COCO's
80 categories, or — once fine-tuned — a specific SPACE_CLASSES list). To
ask "is there anything in this image matching an arbitrary text
description, e.g. 'satellite', 'solar panel', 'asteroid', 'comet'"
without training a single example, we need an open-vocabulary detector:
one trained to match image regions against arbitrary text embeddings
rather than a fixed class list.

OWL-ViT (Minderer et al., "Simple Open-Vocabulary Object Detection with
Vision Transformers", ECCV 2022) does exactly this: a CLIP-style
vision-language backbone repurposed for detection by adding per-patch
class/box heads, matched against text-query embeddings at inference
time. Zero training data needed for a new class — just describe it.

Deliberately NOT wired into the real-time /ws/cv stream: even the
lightest OWL-ViT checkpoint runs at roughly 1-3 FPS on CPU (a full
transformer forward pass per image, versus YOLOv8's purpose-built,
heavily optimized single-shot conv-net architecture), which can't sustain
the live stream's target frame rate. This is exposed instead as a
separate, on-demand "identify this image" tool (see app.py's
POST /api/cv/identify) — you trade speed for open-ended vocabulary,
which is the right trade for a one-shot "what is this?" query, wrong for
continuous streaming.

Reference: Minderer et al., ECCV 2022 (arXiv:2205.06230).
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from cv.detector import Detection

DEFAULT_MODEL_NAME = "google/owlvit-base-patch32"

# A reasonable default vocabulary for "what's in this space image" when
# the caller doesn't supply their own text queries.
DEFAULT_SPACE_QUERIES = [
    "satellite", "space station", "solar panel", "spacecraft",
    "asteroid", "comet", "space debris", "rocket", "astronaut",
]


class OpenVocabularyDetector:
    """Wraps OWL-ViT for zero-shot text-query-driven object detection."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.model.eval()

    def detect(
        self,
        image_bgr: np.ndarray,
        text_queries: list[str] | None = None,
        confidence_threshold: float = 0.1,
    ) -> list[Detection]:
        """
        Run zero-shot detection against a list of free-text queries.

        Args:
            image_bgr: BGR image array (OpenCV convention — converted to
                RGB internally for the model).
            text_queries: arbitrary class descriptions to search for
                (e.g. ["satellite", "solar panel"]); defaults to
                DEFAULT_SPACE_QUERIES if not given.
            confidence_threshold: minimum matching score to keep. OWL-ViT's
                scores are calibrated differently from YOLO's — 0.1 is a
                reasonable starting point for open-vocabulary matching,
                noticeably lower than a typical closed-set detector's
                default, since text-image matching scores run lower
                overall than a purpose-trained single-class head's.

        Returns:
            List of Detection (same shape as cv.detector.Detection, so
            existing HUD-drawing code works unmodified), sorted by
            descending confidence.
        """
        queries = text_queries or DEFAULT_SPACE_QUERIES
        image_rgb = image_bgr[:, :, ::-1]
        pil_image = Image.fromarray(image_rgb)

        inputs = self.processor(text=[queries], images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]])  # (height, width)
        # post_process_grounded_object_detection (the current transformers
        # API — this project briefly hit an older `post_process_object_detection`
        # name that was renamed) resolves text_labels for us directly,
        # rather than us mapping a returned label index back into `queries`.
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=confidence_threshold, target_sizes=target_sizes,
            text_labels=[queries],
        )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            detections.append(
                Detection(class_name=label, confidence=float(score), box_xyxy=(x1, y1, x2, y2))
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
