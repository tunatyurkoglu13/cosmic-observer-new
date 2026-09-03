import numpy as np
import pytest
import torch

from cv.anomaly import (
    AnomalyDetector,
    AnomalyResult,
    SpaceAutoencoder,
    TelemetryAnomalyDetector,
    preprocess_frame_for_autoencoder,
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


def test_preprocess_frame_for_autoencoder_shape_and_range():
    frame_bgr = (np.random.default_rng(0).random((200, 300, 3)) * 255).astype(np.uint8)
    out = preprocess_frame_for_autoencoder(frame_bgr)
    assert out.shape == (SpaceAutoencoder.INPUT_SIZE, SpaceAutoencoder.INPUT_SIZE)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_anomaly_result_severity():
    below = AnomalyResult(reconstruction_error=0.05, threshold=0.1, is_anomaly=False)
    assert below.severity == 0.0

    at_threshold = AnomalyResult(reconstruction_error=0.1, threshold=0.1, is_anomaly=False)
    assert at_threshold.severity == pytest.approx(0.0)

    double = AnomalyResult(reconstruction_error=0.2, threshold=0.1, is_anomaly=True)
    assert double.severity == pytest.approx(1.0)


def test_anomaly_detector_score_frame_flags_high_error():
    torch.manual_seed(0)
    model = SpaceAutoencoder()
    # An untrained (random-weight) model reconstructs essentially anything
    # poorly and consistently — enough to exercise the thresholding logic
    # without needing a real training run in this fast unit test.
    frame = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype(np.uint8)
    baseline_error = float(reconstruction_error(model, preprocess_frame_for_autoencoder(frame)[np.newaxis, ...])[0])

    low_threshold_detector = AnomalyDetector(model=model, threshold=baseline_error - 0.01)
    result = low_threshold_detector.score_frame(frame)
    assert result.is_anomaly is True
    assert result.reconstruction_error == pytest.approx(baseline_error, abs=1e-6)

    high_threshold_detector = AnomalyDetector(model=model, threshold=baseline_error + 10.0)
    assert high_threshold_detector.score_frame(frame).is_anomaly is False


def test_anomaly_detector_save_and_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = SpaceAutoencoder()
    detector = AnomalyDetector(model=model, threshold=0.0321)

    weights_path = tmp_path / "weights.pt"
    meta_path = tmp_path / "meta.json"
    detector.save(weights_path, meta_path)

    loaded = AnomalyDetector.load(weights_path, meta_path)
    assert loaded.threshold == pytest.approx(0.0321)

    frame = (np.random.default_rng(2).random((64, 64, 3)) * 255).astype(np.uint8)
    original_result = detector.score_frame(frame)
    loaded_result = loaded.score_frame(frame)
    assert loaded_result.reconstruction_error == pytest.approx(original_result.reconstruction_error, abs=1e-6)
