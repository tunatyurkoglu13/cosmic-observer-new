"""
cv/anomaly.py — Anomaly detection for space imagery and satellite telemetry.

Two independent methods, per the project spec, for two different kinds
of "anomaly":

  1. SpaceAutoencoder (image-based): a convolutional autoencoder trained
     only on *normal* imagery. At inference time, an image that
     reconstructs poorly (high reconstruction error) is flagged as
     anomalous — the network never learned to reproduce whatever novel
     structure is in it. This is the standard unsupervised approach for
     "flag anything unusual" when you have plenty of normal data but no
     labeled examples of the anomalies themselves (you can't enumerate
     "unknown object near ISS" in advance).

  2. TelemetryAnomalyDetector (tabular): Isolation Forest over satellite
     housekeeping telemetry (solar panel efficiency, attitude/orientation
     rates, thermal profile, etc.) — isolates outliers by how few random
     splits it takes to separate a point from the rest of the data,
     which works well for the kind of multivariate "this combination of
     readings is unusual" signal telemetry anomalies produce, without
     needing a reconstruction model per telemetry channel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest


class SpaceAutoencoder(nn.Module):
    """
    Convolutional autoencoder for space imagery anomaly detection.

    Architecture: 3 downsampling conv blocks (encoder) to a 128-d latent
    vector, mirrored by 3 upsampling transposed-conv blocks (decoder).
    Expects fixed 64x64 single-channel (grayscale) input crops — small
    and single-channel to keep training fast on CPU for this project's
    scope; a production system would likely use larger multi-channel
    crops with a deeper encoder.
    """

    LATENT_DIM = 128
    INPUT_SIZE = 64

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
        )
        self.to_latent = nn.Linear(64 * 8 * 8, self.LATENT_DIM)
        self.from_latent = nn.Linear(self.LATENT_DIM, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # 32 -> 64
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.flatten(1)
        return self.to_latent(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = h.view(-1, 64, 8, 8)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def reconstruction_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    MSE + (1 - SSIM-like structural term) combined loss, per the project
    spec ("Loss: MSE + SSIM"). We use a lightweight single-scale
    structural-similarity approximation (rather than pulling in a
    separate SSIM dependency) so the loss also penalizes structural
    (edge/contrast) mismatches that pure per-pixel MSE is insensitive to.
    """
    mse = F.mse_loss(reconstructed, original)

    c1, c2 = 0.01**2, 0.03**2
    mu_x, mu_y = original.mean(), reconstructed.mean()
    var_x, var_y = original.var(), reconstructed.var()
    cov_xy = ((original - mu_x) * (reconstructed - mu_y)).mean()

    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    ssim_loss = 1.0 - ssim

    return mse + ssim_loss


def train_autoencoder(
    model: SpaceAutoencoder,
    normal_images: np.ndarray,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> list[float]:
    """
    Train the autoencoder on a batch of normal (non-anomalous) images.

    Args:
        model: a SpaceAutoencoder instance.
        normal_images: float32 array of shape (N, 64, 64), values in [0, 1].
        epochs, batch_size, lr: standard training hyperparameters.

    Returns:
        List of mean loss per epoch (for monitoring convergence).
    """
    if normal_images.shape[1:] != (SpaceAutoencoder.INPUT_SIZE, SpaceAutoencoder.INPUT_SIZE):
        raise ValueError(
            f"normal_images must have shape (N, {SpaceAutoencoder.INPUT_SIZE}, "
            f"{SpaceAutoencoder.INPUT_SIZE}); got {normal_images.shape}"
        )

    tensor_data = torch.from_numpy(normal_images).float().unsqueeze(1)  # (N, 1, 64, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = tensor_data.shape[0]
    epoch_losses = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        losses = []
        for i in range(0, n, batch_size):
            batch = tensor_data[perm[i : i + batch_size]]
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = reconstruction_loss(batch, reconstructed)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_losses.append(float(np.mean(losses)))

    return epoch_losses


def reconstruction_error(model: SpaceAutoencoder, images: np.ndarray) -> np.ndarray:
    """
    Per-image reconstruction error (mean squared error), for anomaly
    scoring: higher error = more anomalous relative to what the model
    learned as "normal".

    Args:
        model: a trained SpaceAutoencoder.
        images: float32 array of shape (N, 64, 64), values in [0, 1].

    Returns:
        1D array of per-image MSE, length N.
    """
    model.eval()
    with torch.no_grad():
        tensor_data = torch.from_numpy(images).float().unsqueeze(1)
        reconstructed = model(tensor_data)
        error = ((tensor_data - reconstructed) ** 2).mean(dim=(1, 2, 3))
    return error.numpy()


@dataclass
class TelemetryAnomalyDetector:
    """
    Isolation Forest wrapper for multivariate satellite telemetry anomaly
    detection (e.g. solar panel efficiency, attitude rates, thermal
    readings — any fixed set of numeric channels sampled over time).
    """

    contamination: float = 0.05
    n_estimators: int = 100
    random_state: int = 42

    def __post_init__(self):
        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self._fitted = False

    def fit(self, telemetry: np.ndarray) -> "TelemetryAnomalyDetector":
        """Fit on historical telemetry, shape (N, n_channels)."""
        self._model.fit(telemetry)
        self._fitted = True
        return self

    def predict(self, telemetry: np.ndarray) -> np.ndarray:
        """
        Returns:
            1D array, one entry per row: True if anomalous, False if normal.
        """
        if not self._fitted:
            raise RuntimeError("TelemetryAnomalyDetector.fit() must be called before predict()")
        # IsolationForest.predict returns 1 for inliers, -1 for outliers.
        return self._model.predict(telemetry) == -1

    def anomaly_score(self, telemetry: np.ndarray) -> np.ndarray:
        """
        Returns:
            1D array of anomaly scores: lower (more negative) = more anomalous.
        """
        if not self._fitted:
            raise RuntimeError("TelemetryAnomalyDetector.fit() must be called before anomaly_score()")
        return self._model.decision_function(telemetry)


def preprocess_frame_for_autoencoder(frame_bgr: np.ndarray) -> np.ndarray:
    """
    BGR video frame -> the exact (64, 64) float32 [0, 1] grayscale array
    SpaceAutoencoder expects. Shared by training, evaluation, and live
    inference so all three paths preprocess identically — a training/
    inference preprocessing mismatch is a classic, easy-to-miss source of
    silently-wrong anomaly scores.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (SpaceAutoencoder.INPUT_SIZE, SpaceAutoencoder.INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)


@dataclass
class AnomalyResult:
    """One frame/image's anomaly score against a trained AnomalyDetector."""

    reconstruction_error: float
    threshold: float
    is_anomaly: bool

    @property
    def severity(self) -> float:
        """How far past the threshold, as a ratio (0 = right at threshold, 1 = double the threshold, etc.) — for a HUD intensity/color ramp rather than a flat on/off flag."""
        if self.threshold <= 0:
            return 0.0
        return max(0.0, (self.reconstruction_error - self.threshold) / self.threshold)


class AnomalyDetector:
    """
    Deployable wrapper around SpaceAutoencoder: bundles trained weights +
    a decision threshold + the exact preprocessing pipeline, so a caller
    (the live /ws/cv stream, a batch evaluation script) can go straight
    from a raw BGR frame to an anomaly flag without re-deriving any of
    train_autoencoder()'s conventions.

    The threshold is NOT baked into the model architecture — it's a
    statistic (mean + k*std of reconstruction error on held-out NORMAL
    data, per this project's own evaluation script) computed once after
    training and saved alongside the weights, so re-tuning it doesn't
    require retraining.
    """

    def __init__(self, model: SpaceAutoencoder, threshold: float):
        self.model = model
        self.threshold = threshold
        self.model.eval()

    def score_frame(self, frame_bgr: np.ndarray) -> AnomalyResult:
        image = preprocess_frame_for_autoencoder(frame_bgr)
        error = float(reconstruction_error(self.model, image[np.newaxis, ...])[0])
        return AnomalyResult(reconstruction_error=error, threshold=self.threshold, is_anomaly=error > self.threshold)

    def save(self, weights_path: Path | str, meta_path: Path | str) -> None:
        torch.save(self.model.state_dict(), weights_path)
        Path(meta_path).write_text(json.dumps({"threshold": self.threshold}))

    @classmethod
    def load(cls, weights_path: Path | str, meta_path: Path | str) -> "AnomalyDetector":
        model = SpaceAutoencoder()
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        threshold = json.loads(Path(meta_path).read_text())["threshold"]
        return cls(model=model, threshold=threshold)
