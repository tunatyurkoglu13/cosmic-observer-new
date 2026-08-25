import cv2
import numpy as np

from cv.preprocess import (
    apply_clahe,
    correct_vignette,
    demosaic_bayer,
    denoise,
    preprocess_pipeline,
    sun_glare_mask,
)


def _synthetic_space_image(size=256):
    """A dim starfield with a bright saturated 'sun glare' patch and vignetted corners."""
    rng = np.random.default_rng(42)
    img = (rng.normal(loc=20, scale=5, size=(size, size, 3)).clip(0, 255)).astype(np.uint8)

    # A few bright "stars"/objects.
    for _ in range(15):
        x, y = rng.integers(0, size, size=2)
        cv2.circle(img, (int(x), int(y)), 1, (200, 200, 200), -1)

    # Synthetic vignette: darken corners.
    yy, xx = np.indices((size, size))
    cx, cy = size / 2, size / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size / 2 * 1.4142)
    vignette_gain = np.clip(1.0 - 0.5 * r**2, 0.2, 1.0)[:, :, None]
    img = (img.astype(np.float32) * vignette_gain).astype(np.uint8)

    # Saturated glare patch, drawn AFTER vignetting: a real sensor
    # saturates (clips to max value) regardless of the scene's
    # illumination falloff, so this must stay at full value.
    cv2.circle(img, (30, 30), 15, (255, 255, 255), -1)

    return img


def test_demosaic_bayer_produces_bgr_image():
    raw = np.random.default_rng(0).integers(0, 256, size=(64, 64), dtype=np.uint8)
    color = demosaic_bayer(raw, pattern="BG")
    assert color.shape == (64, 64, 3)
    assert color.dtype == np.uint8


def test_demosaic_bayer_rejects_unknown_pattern():
    raw = np.zeros((16, 16), dtype=np.uint8)
    try:
        demosaic_bayer(raw, pattern="XX")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_correct_vignette_brightens_corners_more_than_center():
    img = _synthetic_space_image()
    corrected = correct_vignette(img, strength=1.0)

    center_before = img[128, 128].astype(float).mean()
    corner_before = img[10, 10].astype(float).mean()
    center_after = corrected[128, 128].astype(float).mean()
    corner_after = corrected[10, 10].astype(float).mean()

    corner_gain = corner_after / max(corner_before, 1.0)
    center_gain = center_after / max(center_before, 1.0)
    assert corner_gain > center_gain


def test_apply_clahe_color_and_grayscale():
    img = _synthetic_space_image()
    result_color = apply_clahe(img)
    assert result_color.shape == img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result_gray = apply_clahe(gray)
    assert result_gray.shape == gray.shape


def test_apply_clahe_increases_local_contrast():
    img = _synthetic_space_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = apply_clahe(gray, clip_limit=3.0)
    # Std deviation of the faint-background region should increase (more visible detail).
    assert enhanced.astype(float).std() >= gray.astype(float).std()


def test_denoise_bilateral_and_nlm_run_without_error():
    img = _synthetic_space_image()
    out_bilateral = denoise(img, method="bilateral")
    out_nlm = denoise(img, method="nlm")
    assert out_bilateral.shape == img.shape
    assert out_nlm.shape == img.shape


def test_denoise_rejects_unknown_method():
    img = _synthetic_space_image()
    try:
        denoise(img, method="unknown")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_sun_glare_mask_flags_saturated_region_not_dim_background():
    img = _synthetic_space_image()
    mask = sun_glare_mask(img)
    assert mask.shape == img.shape[:2]
    # The known glare patch at (30,30) should be flagged.
    assert mask[30, 30] == 255
    # A dim background pixel far from the glare should not be flagged.
    assert mask[200, 200] == 0


def test_preprocess_pipeline_returns_processed_image_and_mask():
    img = _synthetic_space_image()
    processed, mask = preprocess_pipeline(img)
    assert processed.shape == img.shape
    assert mask.shape == img.shape[:2]
    assert set(np.unique(mask)) <= {0, 255}
