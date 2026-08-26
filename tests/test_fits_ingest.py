import numpy as np
import pytest
from astropy.io import fits

from cv.fits_ingest import is_fits_path, load_fits_image, load_image_any, pixel_to_sky


def _write_synthetic_fits(path, with_wcs=True, with_nan=False):
    """A small synthetic FITS file: a bright Gaussian 'star' on a noisy background, float64 data."""
    rng = np.random.default_rng(0)
    size = 64
    data = rng.normal(loc=100.0, scale=5.0, size=(size, size)).astype(np.float64)

    yy, xx = np.indices((size, size))
    cy, cx = size / 2, size / 2
    star = 2000.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0**2)))
    data += star

    if with_nan:
        data[0, 0] = np.nan

    header = fits.Header()
    if with_wcs:
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRVAL1"] = 150.0
        header["CRVAL2"] = 2.0
        header["CRPIX1"] = size / 2
        header["CRPIX2"] = size / 2
        header["CDELT1"] = -0.0002777
        header["CDELT2"] = 0.0002777
        header["CUNIT1"] = "deg"
        header["CUNIT2"] = "deg"

    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(path, overwrite=True)


def test_is_fits_path_recognizes_extensions():
    assert is_fits_path("image.fits")
    assert is_fits_path("image.fit")
    assert is_fits_path("image.fts")
    assert is_fits_path("IMAGE.FITS")  # case-insensitive
    assert not is_fits_path("image.jpg")
    assert not is_fits_path("image.png")


def test_load_fits_image_produces_valid_8bit_bgr(tmp_path):
    path = tmp_path / "synthetic.fits"
    _write_synthetic_fits(path)

    result = load_fits_image(path)
    assert result.data_bgr.dtype == np.uint8
    assert result.data_bgr.shape == (64, 64, 3)
    assert result.data_bgr.min() >= 0
    assert result.data_bgr.max() <= 255
    # The synthetic star should be much brighter than the background after stretching.
    center_brightness = result.data_bgr[32, 32, 0]
    corner_brightness = result.data_bgr[2, 2, 0]
    assert center_brightness > corner_brightness


def test_load_fits_image_handles_nan_pixels_without_error(tmp_path):
    path = tmp_path / "with_nan.fits"
    _write_synthetic_fits(path, with_nan=True)

    result = load_fits_image(path)
    assert np.isfinite(result.data_bgr).all()


def test_load_fits_image_extracts_wcs(tmp_path):
    path = tmp_path / "with_wcs.fits"
    _write_synthetic_fits(path, with_wcs=True)

    result = load_fits_image(path)
    assert result.wcs is not None
    assert result.wcs.has_celestial


def test_load_fits_image_no_wcs_when_header_lacks_it(tmp_path):
    path = tmp_path / "no_wcs.fits"
    _write_synthetic_fits(path, with_wcs=False)

    result = load_fits_image(path)
    assert result.wcs is None


def test_pixel_to_sky_returns_none_without_wcs():
    assert pixel_to_sky(None, 10, 10) is None


def test_pixel_to_sky_returns_coordinates_with_wcs(tmp_path):
    path = tmp_path / "with_wcs.fits"
    _write_synthetic_fits(path, with_wcs=True)
    result = load_fits_image(path)

    sky = pixel_to_sky(result.wcs, 32, 32)
    assert sky is not None
    ra, dec = sky
    # Center pixel should map close to the CRVAL reference point.
    assert abs(ra - 150.0) < 0.01
    assert abs(dec - 2.0) < 0.01


def test_load_fits_image_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fits_image(tmp_path / "does_not_exist.fits")


def test_load_image_any_dispatches_fits(tmp_path):
    path = tmp_path / "synthetic.fits"
    _write_synthetic_fits(path)
    image = load_image_any(path)
    assert image.dtype == np.uint8
    assert image.shape == (64, 64, 3)


def test_load_image_any_dispatches_standard_formats(tmp_path):
    import cv2

    path = tmp_path / "plain.png"
    cv2.imwrite(str(path), np.full((32, 32, 3), 128, dtype=np.uint8))
    image = load_image_any(path)
    assert image.shape == (32, 32, 3)


def test_load_image_any_raises_for_invalid_file(tmp_path):
    path = tmp_path / "garbage.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image_any(path)


@pytest.mark.network
@pytest.mark.slow
def test_load_real_spitzer_telescope_fits_image(tmp_path):
    """
    Real end-to-end check against genuine telescope data (not synthetic):
    downloads a real Spitzer Space Telescope FITS image and verifies our
    ingestion pipeline produces a sane, correctly-stretched result.
    """
    import requests

    url = "https://astropy.stsci.edu/data/photometry/spitzer_example_image.fits"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        pytest.skip(f"Could not download real test FITS file (external condition): {e}")

    path = tmp_path / "spitzer.fits"
    path.write_bytes(resp.content)

    result = load_fits_image(path)
    assert result.data_bgr.dtype == np.uint8
    assert result.wcs is not None
    assert result.wcs.has_celestial
    # A real star field should have real dynamic range, not be flat/blank.
    assert result.data_bgr.std() > 10
