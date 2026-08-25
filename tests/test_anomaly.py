import numpy as np
import pytest
import torch

from cv.anomaly import (
    SpaceAutoencoder,
    TelemetryAnomalyDetector,
    reconstruction_error,
    reconstruction_loss,
    train_autoencoder,
)


def _make_normal_images(n=64, size=64, seed=0):
    """Synthetic 'normal' images: a bright disk of fixed radius at a fixed center, plus noise."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, size, size), dtype=np.float32)
    yy, xx = np.indices((size, size))
    cy, cx = size / 2, size / 2
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2) < (size / 4) ** 2
    for i in range(n):
        img = disk.astype(np.float32) * 0.8
        img += rng.normal(0, 0.03, size=(size, size)).astype(np.float32)
        images[i] = np.clip(img, 0, 1)
    return images


def _make_anomalous_images(n=8, size=64, seed=1):
    """Images with a structurally different pattern (offset disk, or a ring instead of a disk)."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, size, size), dtype=np.float32)
    yy, xx = np.indices((size, size))
    for i in range(n):
        cy, cx = size * 0.75, size * 0.75  # off-center, unlike training data
        ring = (((xx - cx) ** 2 + (yy - cy) ** 2) > (size / 6) ** 2) & (
            ((xx - cx) ** 2 + (yy - cy) ** 2) < (size / 4) ** 2
        )
        img = ring.astype(np.float32) * 0.8
        img += rng.normal(0, 0.03, size=(size, size)).astype(np.float32)
        images[i] = np.clip(img, 0, 1)
    return images


def test_autoencoder_forward_pass_shape():
    model = SpaceAutoencoder()
    x = torch.rand(4, 1, 64, 64)
    out = model(x)
    assert out.shape == x.shape


def test_reconstruction_loss_zero_for_identical_images():
    # MSE is exactly 0 and SSIM is mathematically exactly 1 for x==x, but
    # the SSIM epsilon terms (c1, c2) plus float32 precision leave a
    # residual on the order of 1e-4 rather than a literal 0.
    x = torch.rand(2, 1, 64, 64)
    loss = reconstruction_loss(x, x)
    assert loss.item() < 1e-3


def test_reconstruction_loss_positive_for_different_images():
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    y = torch.rand(2, 1, 64, 64)
    loss = reconstruction_loss(x, y)
    assert loss.item() > 0


def test_train_autoencoder_rejects_wrong_input_shape():
    model = SpaceAutoencoder()
    bad_images = np.zeros((4, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        train_autoencoder(model, bad_images, epochs=1)


@pytest.mark.slow
def test_autoencoder_training_converges_and_flags_real_anomalies():
    """
    Real end-to-end training run (not mocked): trains on synthetic
    'normal' images (centered disk), then verifies reconstruction error
    is measurably higher on structurally different 'anomalous' images
    (off-center ring) than on held-out normal images.
    """
    torch.manual_seed(0)
    model = SpaceAutoencoder()

    normal_train = _make_normal_images(n=64, seed=0)
    losses = train_autoencoder(model, normal_train, epochs=15, batch_size=16, lr=2e-3)

    assert losses[-1] < losses[0], "training loss should decrease"

    normal_holdout = _make_normal_images(n=16, seed=99)
    anomalous = _make_anomalous_images(n=16, seed=1)

    normal_errors = reconstruction_error(model, normal_holdout)
    anomalous_errors = reconstruction_error(model, anomalous)

    assert anomalous_errors.mean() > normal_errors.mean()


def test_telemetry_anomaly_detector_flags_obvious_outlier():
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=[50.0, 0.0, 20.0], scale=[2.0, 0.5, 1.0], size=(200, 3))
    detector = TelemetryAnomalyDetector(contamination=0.05).fit(normal)

    outlier = np.array([[150.0, 20.0, -50.0]])  # wildly out of the normal distribution
    inlier = np.array([[50.0, 0.0, 20.0]])

    assert detector.predict(outlier)[0] == True
    assert detector.predict(inlier)[0] == False


def test_telemetry_anomaly_detector_score_ordering():
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=0.0, scale=1.0, size=(200, 2))
    detector = TelemetryAnomalyDetector(contamination=0.05).fit(normal)

    mild = np.array([[1.0, 1.0]])
    extreme = np.array([[10.0, 10.0]])
    scores = detector.anomaly_score(np.vstack([mild, extreme]))
    # More anomalous point should have a lower (more negative) score.
    assert scores[1] < scores[0]


def test_telemetry_anomaly_detector_requires_fit_before_predict():
    detector = TelemetryAnomalyDetector()
    with pytest.raises(RuntimeError):
        detector.predict(np.zeros((1, 3)))
    with pytest.raises(RuntimeError):
        detector.anomaly_score(np.zeros((1, 3)))
