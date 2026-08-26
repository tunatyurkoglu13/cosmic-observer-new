"""
cv/fits_ingest.py — FITS (Flexible Image Transport System) image ingestion.

Real telescope/astrophotography data is almost never JPEG/PNG — it's
FITS: a 16/32-bit (often float) per-pixel format with a metadata header
that can carry WCS (World Coordinate System) keywords mapping pixel
coordinates to sky coordinates (RA/Dec). cv/preprocess.py and cv/detector.py
otherwise assume standard 8-bit BGR arrays (what cv2.imread and video
frames give you), so this module is the bridge: read a real FITS file,
extract its WCS if present, and stretch its high-dynamic-range data down
to an 8-bit image OpenCV/YOLO can consume.

`astropy` has been in requirements.txt since Phase 0 but never actually
used until now — this is its first real job in the project.

Stretch method: FITS pixel values commonly span many orders of magnitude
(a bright star vs. faint background), so a naive linear min-max stretch
either blows out bright sources or crushes faint ones to black. The
standard astronomical-imaging approach (used by DS9, SAOImage, and
astropy.visualization itself) is:
    1. ZScale (Tody 1980, the algorithm IRAF's `display` popularized): an
       iterative sigma-clipped linear fit that picks a good [z1, z2]
       display range automatically, without a human tuning percentiles
       per image.
    2. Optionally follow with an asinh (inverse hyperbolic sine) stretch
       (Lupton et al. 2004) instead of a plain linear map within that
       range — asinh compresses the bright end logarithmically while
       staying linear near zero, showing faint detail AND bright cores
       in the same image (useful for a bright satellite streak against
       a faint starfield, exactly this project's use case).

References: Tody, "Data Reduction Techniques..." (IRAF, 1980);
Lupton et al., "Preparing Red-Green-Blue Images from CCD Data" (PASP, 2004).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ZScaleInterval
from astropy.wcs import WCS

FITS_EXTENSIONS = {".fits", ".fit", ".fts"}


@dataclass
class FitsImage:
    """A FITS file's image data plus whatever WCS/header metadata it carries."""

    data_bgr: np.ndarray          # 8-bit BGR, ready for cv2/YOLO
    raw_data: np.ndarray          # original float pixel values, unstretched
    wcs: WCS | None               # sky-coordinate mapping, if the header had one
    header: fits.Header


def is_fits_path(path: Path | str) -> bool:
    """True if the file extension indicates FITS (case-insensitive)."""
    return Path(path).suffix.lower() in FITS_EXTENSIONS


def _stretch_to_uint8(data: np.ndarray, use_asinh: bool = True) -> np.ndarray:
    """
    Map high-dynamic-range float pixel data to an 8-bit grayscale image
    using ZScale to pick the display range, then optionally an asinh
    stretch within it (see module docstring for why).
    """
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.uint8)

    clean = np.where(finite, data, np.nanmin(data[finite]))

    z1, z2 = ZScaleInterval().get_limits(clean)
    if z2 <= z1:
        z1, z2 = float(np.min(clean)), float(np.max(clean)) or 1.0

    normalized = np.clip((clean - z1) / (z2 - z1), 0.0, 1.0)
    if use_asinh:
        normalized = AsinhStretch()(normalized)

    return (normalized * 255).astype(np.uint8)


def load_fits_image(path: Path | str, hdu_index: int = 0, use_asinh: bool = True) -> FitsImage:
    """
    Load a FITS file and return an 8-bit BGR image ready for the rest of
    the CV pipeline, plus the original data/WCS/header for anything that
    needs sky-coordinate context (e.g. streak_detection's RA/Dec output).

    Args:
        path: path to a .fits/.fit/.fts file.
        hdu_index: which HDU (Header/Data Unit) to read — index 0 (the
            primary HDU) is correct for the vast majority of
            single-image FITS files; multi-extension files (e.g. some
            multi-CCD mosaics) may need a different index.
        use_asinh: apply the asinh stretch on top of ZScale's range (see
            module docstring); set False for a plain linear stretch
            within the ZScale range instead.

    Returns:
        FitsImage with an 8-bit BGR array plus WCS/header metadata.

    Raises:
        FileNotFoundError: if the path doesn't exist.
        ValueError: if the requested HDU has no image data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {path}")

    with fits.open(path) as hdul:
        hdu = hdul[hdu_index]
        if hdu.data is None:
            raise ValueError(f"HDU {hdu_index} in {path} has no image data")

        raw_data = np.asarray(hdu.data, dtype=np.float64)
        header = hdu.header.copy()

        try:
            wcs = WCS(header)
            if not wcs.has_celestial:
                wcs = None
        except Exception:
            wcs = None

    gray = _stretch_to_uint8(raw_data, use_asinh=use_asinh)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return FitsImage(data_bgr=bgr, raw_data=raw_data, wcs=wcs, header=header)


def pixel_to_sky(wcs: WCS, x: float, y: float) -> tuple[float, float] | None:
    """
    Convert a pixel (x, y) to (RA, Dec) in degrees using the image's WCS.
    Returns None if there's no valid celestial WCS.
    """
    if wcs is None or not wcs.has_celestial:
        return None
    ra, dec = wcs.all_pix2world(x, y, 0)
    return float(ra), float(dec)


def load_image_any(path: Path | str) -> np.ndarray:
    """
    Load an image file into an 8-bit BGR array regardless of whether
    it's FITS or a standard format (JPEG/PNG/etc.) — a single entry
    point for cv/preprocess.py and cv/detector.py so callers don't need
    to branch on file extension themselves.
    """
    path = Path(path)
    if is_fits_path(path):
        return load_fits_image(path).data_bgr

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image (not valid FITS or a format OpenCV supports): {path}")
    return image
