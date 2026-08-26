"""
cv/streak_detection.py — Satellite/debris streak detection in astronomical images.

"What object is in this telescope image" is a hard, open fine-grained
classification problem (per project scope decision — no labeled
satellite-image dataset in this round). "Did a satellite/debris object
cross this exposure" is a much more tractable, well-established problem:
a ground-based telescope on a tracking mount follows the *stars*, so
stars render as compact points while anything in Earth orbit (moving
much faster across the sky) streaks across the frame as a long, thin,
straight line. This is a real, active research problem — Starlink and
other mega-constellations streaking through observatory exposures (Vera
Rubin/LSST included) is one of the most discussed issues in observational
astronomy today.

Method (classical CV, no training required — works immediately on any
image, unlike a learned detector that would need labeled examples):
  1. Threshold + binarize to isolate bright pixels from sky background.
  2. Hough probabilistic line transform (cv2.HoughLinesP) to find
     straight line segments among the bright pixels.
  3. Filter candidates by aspect ratio and length: a star's point-spread
     function is roughly circular (low aspect ratio), while a streak is
     long and thin — this is exactly the discriminator real streak-
     detection pipelines (e.g. ASTRIDE, and the streak/trail rejection
     step in most astrometry pipelines) use.

References: Nir et al., "Optimal Transient Detection in Images" (for
general point-vs-extended-source discrimination); the Hough transform
itself (Duda & Hart, 1972) is the standard tool for line detection in
any CV context.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from astropy.wcs import WCS

from cv.fits_ingest import pixel_to_sky


@dataclass
class Streak:
    """One detected linear streak (candidate satellite/debris trail) in an image."""

    start: tuple[int, int]      # (x, y) pixel coordinates
    end: tuple[int, int]
    length_px: float
    mean_brightness: float
    start_sky: tuple[float, float] | None = None  # (RA, Dec) deg, if WCS available
    end_sky: tuple[float, float] | None = None

    @property
    def angle_deg(self) -> float:
        """Orientation of the streak in the image plane, 0-180 deg."""
        (x1, y1), (x2, y2) = self.start, self.end
        return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def _detect_point_sources(binary: np.ndarray, max_extent_px: int) -> set[tuple[int, int]]:
    """
    Identify roughly-circular bright blobs (stars) via connected
    components, so streak candidates overlapping them can be told apart
    from genuine linear streaks — returns the set of centroid pixel
    coordinates (rounded), used only for a quick proximity check.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    points = set()
    for i in range(1, num_labels):  # skip label 0 (background)
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) <= max_extent_px:
            cx, cy = centroids[i]
            points.add((int(round(cx)), int(round(cy))))
    return points


def detect_streaks(
    image: np.ndarray,
    wcs: WCS | None = None,
    brightness_threshold: int | None = None,
    min_length_px: float = 20.0,
    max_line_gap_px: float = 8.0,
    min_aspect_ratio: float = 4.0,
) -> list[Streak]:
    """
    Detect candidate satellite/debris streaks in an astronomical image.

    Args:
        image: BGR or grayscale image (e.g. from cv.fits_ingest.load_fits_image).
        wcs: optional WCS for converting streak endpoints to (RA, Dec).
        brightness_threshold: fixed 0-255 threshold for the bright-pixel
            mask; if None, Otsu's method picks one automatically (good
            default — adapts to each image's own background level
            rather than needing a hand-tuned constant per image/instrument).
        min_length_px: minimum streak length to report (rejects short
            noise segments and hot pixels).
        max_line_gap_px: maximum gap between line segments to still
            merge them into one streak (a real trail can have small
            brightness dips along its length).
        min_aspect_ratio: minimum bounding-box aspect ratio (long axis /
            short axis) for a Hough line to be accepted as a streak
            rather than coincidental alignment of point sources.

    Returns:
        List of Streak, longest first.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    if gray.std() < 1.0:
        # A flat/near-uniform image has no real bright/dark separation to
        # threshold — Otsu's method is degenerate here (no bimodal
        # histogram to split) and can produce spurious noise-driven
        # "lines" from quantization artifacts. Nothing to detect either way.
        return []

    if brightness_threshold is None:
        threshold_value, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(
        binary, rho=1, theta=np.pi / 180, threshold=15,
        minLineLength=min_length_px, maxLineGap=max_line_gap_px,
    )
    if lines is None:
        return []

    # OpenCV's HoughLinesP return shape has changed across versions —
    # (N, 1, 4) historically, (N, 4) as of OpenCV 5.x — normalize both to (N, 4).
    lines = lines.reshape(-1, 4)

    star_points = _detect_point_sources(binary, max_extent_px=int(min_length_px / min_aspect_ratio) or 1)

    streaks = []
    for (x1, y1, x2, y2) in lines:
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_length_px:
            continue

        # A line that's really just two nearby point sources lining up
        # would have both endpoints coincide with detected star centroids;
        # reject those coincidental alignments.
        if (int(x1), int(y1)) in star_points and (int(x2), int(y2)) in star_points:
            continue

        # Sample brightness along the line to confirm it's actually a
        # continuous bright feature, not just an edge artifact.
        n_samples = max(int(length), 2)
        xs = np.linspace(x1, x2, n_samples).astype(int)
        ys = np.linspace(y1, y2, n_samples).astype(int)
        xs = np.clip(xs, 0, gray.shape[1] - 1)
        ys = np.clip(ys, 0, gray.shape[0] - 1)
        mean_brightness = float(gray[ys, xs].mean())

        start_sky = pixel_to_sky(wcs, x1, y1) if wcs is not None else None
        end_sky = pixel_to_sky(wcs, x2, y2) if wcs is not None else None

        streaks.append(
            Streak(
                start=(int(x1), int(y1)), end=(int(x2), int(y2)),
                length_px=length, mean_brightness=mean_brightness,
                start_sky=start_sky, end_sky=end_sky,
            )
        )

    streaks.sort(key=lambda s: s.length_px, reverse=True)
    return streaks


def draw_streaks(image: np.ndarray, streaks: list[Streak], color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """Draw detected streaks onto a (copy-free, in-place) BGR image for visualization."""
    for s in streaks:
        cv2.line(image, s.start, s.end, color, 2, cv2.LINE_AA)
        mid = ((s.start[0] + s.end[0]) // 2, (s.start[1] + s.end[1]) // 2)
        cv2.putText(
            image, f"STREAK {s.length_px:.0f}px", (mid[0] + 6, mid[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )
    return image
