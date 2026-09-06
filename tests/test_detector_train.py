import pytest

from cv.detector_train import _split_train_val


def _make_pair(dataset_root, stem, label_content):
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    images_train.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)
    (images_train / f"{stem}.jpg").write_bytes(b"fake-jpeg-bytes")
    (labels_train / f"{stem}.txt").write_text(label_content)


def test_split_train_val_drops_unlabeled_images(tmp_path):
    _make_pair(tmp_path, "labeled1", "1 0.5 0.5 0.2 0.2")
    _make_pair(tmp_path, "labeled2", "0 0.3 0.3 0.1 0.1")
    _make_pair(tmp_path, "labeled3", "2 0.6 0.6 0.1 0.1")
    _make_pair(tmp_path, "empty", "")

    n_train, n_val = _split_train_val(tmp_path, val_fraction=0.0)

    # val_fraction=0.0 still keeps >=1 held-out sample when any labeled data
    # exists (Ultralytics needs a non-empty val split) — so of the 3 labeled
    # images, exactly 1 goes to val and 2 stay in train.
    assert n_train == 2
    assert n_val == 1
    assert not (tmp_path / "images" / "train" / "empty.jpg").exists()
    assert not (tmp_path / "labels" / "train" / "empty.txt").exists()


def test_split_train_val_moves_a_fraction_into_val(tmp_path):
    for i in range(10):
        _make_pair(tmp_path, f"img{i}", "0 0.5 0.5 0.1 0.1")

    n_train, n_val = _split_train_val(tmp_path, val_fraction=0.2, seed=1)

    assert n_train == 8
    assert n_val == 2
    val_images = list((tmp_path / "images" / "val").glob("*.jpg"))
    val_labels = list((tmp_path / "labels" / "val").glob("*.txt"))
    assert len(val_images) == 2
    assert len(val_labels) == 2
    # Every moved image kept its matching label (same stem).
    assert {p.stem for p in val_images} == {p.stem for p in val_labels}


def test_split_train_val_with_no_labeled_images_returns_zero(tmp_path):
    _make_pair(tmp_path, "empty1", "")
    _make_pair(tmp_path, "empty2", "  \n")

    n_train, n_val = _split_train_val(tmp_path, val_fraction=0.2)

    assert n_train == 0
    assert n_val == 0
