import pytest

from cv.dataset_builder import auto_label_with_zero_shot, fetch_nasa_images


@pytest.mark.network
@pytest.mark.slow
def test_fetch_nasa_images_downloads_real_images(tmp_path):
    """Real end-to-end check: NASA's public Images API is keyless and should return real, downloadable photos."""
    paths = fetch_nasa_images("satellite", limit=3, output_dir=tmp_path)

    assert len(paths) > 0
    assert len(paths) <= 3
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
        assert p.suffix == ".jpg"
        # Should be saved into the YOLO-expected images/train/ layout.
        assert p.parent == tmp_path / "images" / "train"


@pytest.mark.network
@pytest.mark.slow
def test_auto_label_with_zero_shot_produces_yolo_format_labels(tmp_path):
    """
    Full real pipeline: fetch real NASA images, then auto-label them with
    the real zero-shot detector, and verify the resulting label files are
    valid YOLO format (normalized floats in [0, 1]).
    """
    image_paths = fetch_nasa_images("satellite", limit=2, output_dir=tmp_path)
    if not image_paths:
        pytest.skip("NASA Images API returned no results (external condition)")

    label_paths = auto_label_with_zero_shot(image_paths, output_dir=tmp_path)

    assert len(label_paths) == len(image_paths)
    for label_path in label_paths:
        assert label_path.parent == tmp_path / "labels" / "train"
        assert label_path.exists()

        content = label_path.read_text().strip()
        if not content:
            continue  # a valid outcome: zero-shot found nothing above threshold in that image

        for line in content.splitlines():
            parts = line.split()
            assert len(parts) == 5
            class_id = int(parts[0])
            assert class_id >= 0
            x, y, w, h = (float(v) for v in parts[1:])
            for v in (x, y, w, h):
                assert 0.0 <= v <= 1.0
