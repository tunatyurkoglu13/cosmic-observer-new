"""
cv/preprocess.py — Image preprocessing for space imagery (ISS cameras,
ground-based tracking telescopes, etc.).

Space imagery has a distinctive set of problems that generic photo
preprocessing pipelines don't target well:
  - Extreme dynamic range: a sunlit spacecraft/Earth limb next to
    absolute-black space in the same frame blows out normal auto-exposure.
  - Vignetting from wide-angle lens/optical systems, worse than typical
    consumer photography.
  - Sun glare / direct reflection saturating a region of the frame.
  - Low SNR in the faint background (stars, dim debris) that a detector
    needs to still resolve without amplifying the vignette/noise.

This module addresses each with a standard, well-understood classical
technique (deliberately not deep-learning-based — these run before the
detector in cv/detector.py, and need to be fast/deterministic and not
introduce their own failure modes into the pipeline).

Pipeline order matters (per project spec): demosaic -> CLAHE -> denoise ->
glare mask. Each function is independent/composable so callers can skip a
stage if their input doesn't need it (e.g. already-debayered images skip
demosaic_bayer()).
"""

from __future__ import annotations

import cv2
import numpy as np


def demosaic_bayer(raw: np.ndarray, pattern: str = "BG") -> np.ndarray:
    """
    Demosaic a single-channel Bayer-pattern raw sensor image into a BGR
    color image.

    Args:
        raw: 2D uint8/uint16 array, single-channel Bayer mosaic.
        pattern: Bayer pattern arrangement — one of "BG", "GB", "RG", "GR"
            (matches the top-left 2x2 tile's channel order). Check your
            sensor's datasheet; getting this wrong produces a
            color-swapped but structurally plausible-looking image, which
            is an easy silent bug to miss.

    Returns:
        BGR image, same height/width as `raw`, 3 channels.
    """
    code_map = {
        "BG": cv2.COLOR_BayerBG2BGR,
        "GB": cv2.COLOR_BayerGB2BGR,
        "RG": cv2.COLOR_BayerRG2BGR,
        "GR": cv2.COLOR_BayerGR2BGR,
    }
    if pattern not in code_map:
        raise ValueError(f"Unknown Bayer pattern '{pattern}'; expected one of {list(code_map)}")
    return cv2.cvtColor(raw, code_map[pattern])


def correct_vignette(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Correct radial vignetting (corner darkening) by dividing out a
    synthetic radial falloff model and rescaling to the original dynamic
    range.

    This uses a generic parametric vignette model (brightness falls off
    with the square of normalized radius from center) rather than a
    per-camera calibration map, since we don't have per-device
    flat-field calibration data — good enough to flatten the
    illumination gradient for downstream contrast enhancement, not a
    substitute for true flat-fielding in a calibrated imaging pipeline.

    Args:
        image: BGR or grayscale uint8 image.
        strength: 0 = no correction, 1 = full model-predicted correction,
            >1 = over-correct (brightens corners beyond flat).

    Returns:
        Vignette-corrected image, same shape/dtype as input.
    """
    h, w = image.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    cx, cy = w / 2.0, h / 2.0
    max_r = np.sqrt(cx**2 + cy**2)
    r_norm = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_r

    # Simple radial falloff model: brightness ~ 1 - k*r^2, k tuned so the
    # extreme corner (r_norm=1) is dimmed to ~60% of center brightness.
    k = 0.4
    falloff = 1.0 - k * r_norm**2
    falloff = np.clip(falloff, 0.1, 1.0)  # avoid divide-by-near-zero at the very corner
    gain = 1.0 + strength * (1.0 / falloff - 1.0)

    if image.ndim == 3:
        gain = gain[:, :, None]

    corrected = image.astype(np.float32) * gain
    return np.clip(corrected, 0, 255).astype(image.dtype)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalization: locally boosts
    contrast (so a faint object near a bright limb is still visible)
    while clip_limit caps how much any single histogram bin can be
    amplified, preventing noise in near-uniform dark regions (deep space)
    from being blown up into visible speckle.

    For color images, CLAHE is applied to the L channel in LAB color
    space (equalizing luminance without distorting hue/saturation).

    Args:
        image: BGR (3-channel) or grayscale (single-channel) uint8 image.
        clip_limit: contrast limiting threshold (higher = more contrast, more noise risk).
        tile_grid_size: number of tiles the image is divided into for local equalization.

    Returns:
        Contrast-enhanced image, same shape/dtype as input.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise(image: np.ndarray, method: str = "bilateral", strength: float = 10.0) -> np.ndarray:
    """
    Reduce sensor noise while preserving edges (important for keeping
    small/dim objects distinguishable from noise rather than smearing
    them away, which a naive Gaussian blur would do).

    Args:
        image: BGR or grayscale uint8 image.
        method: "bilateral" (fast, edge-preserving, good default) or
            "nlm" (Non-Local Means — better quality on heavier noise,
            noticeably slower; use for a final-quality pass rather than
            a real-time pipeline).
        strength: for "bilateral", used as both sigmaColor and
            sigmaSpace; for "nlm", used as the filter strength `h`.

    Returns:
        Denoised image, same shape/dtype as input.
    """
    if method == "bilateral":
        return cv2.bilateralFilter(image, d=9, sigmaColor=strength, sigmaSpace=strength)
    if method == "nlm":
        if image.ndim == 2:
            return cv2.fastNlMeansDenoising(image, h=strength)
        return cv2.fastNlMeansDenoisingColored(image, h=strength, hColor=strength)
    raise ValueError(f"Unknown denoise method '{method}'; expected 'bilateral' or 'nlm'")


def sun_glare_mask(image: np.ndarray, saturation_threshold: int = 250, dilate_px: int = 5) -> np.ndarray:
    """
    Flag saturated/overexposed regions (direct sun glare or reflections)
    so a detector can exclude them from consideration rather than
    misinterpreting a glare blob as an object.

    Method: threshold on near-maximum pixel value (in grayscale/luminance),
    then dilate the mask slightly to also cover the glare's soft bloom
    edge (which sits just under the hard saturation threshold but is
    still glare, not signal).

    Args:
        image: BGR or grayscale uint8 image.
        saturation_threshold: pixel value (0-255) above which a pixel is
            considered saturated.
        dilate_px: how many pixels to grow the mask outward.

    Returns:
        Binary mask (uint8, 0 or 255), same height/width as input, where
        255 marks glare-affected pixels to exclude.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, saturation_threshold, 255, cv2.THRESH_BINARY)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, kernel)

    return mask


def preprocess_pipeline(
    image: np.ndarray,
    correct_vignette_first: bool = True,
    clahe_clip_limit: float = 2.0,
    denoise_method: str = "bilateral",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline in the project's specified order:
    vignette correction -> CLAHE -> denoise -> glare mask computation.

    Args:
        image: input BGR image (already demosaiced if it came from a
            Bayer sensor — call demosaic_bayer() first if not).
        correct_vignette_first: whether to apply vignette correction
            (skip if the image is known to already be flat-fielded).
        clahe_clip_limit: passed to apply_clahe().
        denoise_method: passed to denoise().

    Returns:
        (processed_image, glare_mask) — the glare mask is computed from
        the *original* image (saturation is a property of the raw signal,
        not of our own contrast enhancement) and returned alongside so a
        detector can mask it out of its input/results.
    """
    glare_mask = sun_glare_mask(image)

    result = image
    if correct_vignette_first:
        result = correct_vignette(result)
    result = apply_clahe(result, clip_limit=clahe_clip_limit)
    result = denoise(result, method=denoise_method)

    return result, glare_mask
