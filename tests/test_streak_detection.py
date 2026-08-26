import cv2
import numpy as np

from cv.streak_detection import detect_streaks, draw_streaks


def _make_starfield_with_streak(size=256, seed=0, streak_start=(20, 200), streak_end=(230, 30), num_stars=40):
    """Synthetic image: round point-source 'stars' plus one long, thin, straight 'streak'."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 15, dtype=np.uint8)  # dim sky background

    # Point sources (stars): small filled circles, roughly circular.
    for _ in range(num_stars):
        cx, cy = rng.integers(10, size - 10, size=2)
        radius = rng.integers(1, 3)
        cv2.circle(img, (int(cx), int(cy)), int(radius), 220, -1)

    # The streak: a long, thin, bright line (a satellite trail during a tracked exposure).
    cv2.line(img, streak_start, streak_end, 200, 2, cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_detect_streaks_finds_the_synthetic_streak():
    img = _make_starfield_with_streak()
    streaks = detect_streaks(img, min_length_px=30.0)

    assert len(streaks) >= 1
    longest = streaks[0]
    assert longest.length_px > 100  # the synthetic streak spans most of the 256px frame diagonally


def test_detect_streaks_length_matches_known_geometry():
    start, end = (20, 200), (230, 30)
    img = _make_starfield_with_streak(streak_start=start, streak_end=end)
    expected_length = np.hypot(end[0] - start[0], end[1] - start[1])

    streaks = detect_streaks(img, min_length_px=30.0)
    assert len(streaks) >= 1
    # Hough detection may not find the exact same endpoints, but length should be close.
    assert abs(streaks[0].length_px - expected_length) < expected_length * 0.15


def test_detect_streaks_empty_image_returns_no_streaks():
    img = np.full((128, 128, 3), 10, dtype=np.uint8)
    streaks = detect_streaks(img)
    assert streaks == []


def test_detect_streaks_stars_only_no_false_streak():
    """A field of only round point sources (no actual streak) should not report a long streak."""
    rng = np.random.default_rng(1)
    img = np.full((256, 256), 15, dtype=np.uint8)
    for _ in range(60):
        cx, cy = rng.integers(10, 246, size=2)
        cv2.circle(img, (int(cx), int(cy)), int(rng.integers(1, 3)), 220, -1)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    streaks = detect_streaks(img, min_length_px=30.0)
    # Any spurious detections from coincidental point alignment must be short.
    for s in streaks:
        assert s.length_px < 60.0


def test_streak_angle_deg_matches_known_orientation():
    # A perfectly horizontal streak should have angle_deg == 0.
    img = _make_starfield_with_streak(streak_start=(20, 128), streak_end=(230, 128), num_stars=5)
    streaks = detect_streaks(img, min_length_px=30.0)
    assert len(streaks) >= 1
    assert abs(streaks[0].angle_deg) < 3.0 or abs(streaks[0].angle_deg - 180) < 3.0


def test_draw_streaks_modifies_image():
    img = _make_starfield_with_streak()
    streaks = detect_streaks(img, min_length_px=30.0)
    before = img.copy()
    draw_streaks(img, streaks)
    assert not np.array_equal(img, before)
