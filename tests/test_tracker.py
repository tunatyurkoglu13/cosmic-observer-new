from cv.detector import Detection
from cv.tracker import MultiObjectTracker, Track, iou


def _det(x1, y1, x2, y2, cls="satellite", conf=0.9):
    return Detection(class_name=cls, confidence=conf, box_xyxy=(x1, y1, x2, y2))


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == 1.0


def test_iou_non_overlapping_boxes_is_zero():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (100.0, 100.0, 110.0, 110.0)
    assert iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 15.0, 15.0)
    # Intersection = 5x5=25, union = 100+100-25=175
    assert abs(iou(a, b) - 25.0 / 175.0) < 1e-9


def test_tracker_assigns_new_ids_on_first_frame():
    tracker = MultiObjectTracker()
    tracks = tracker.update([_det(0, 0, 10, 10), _det(50, 50, 60, 60)])
    assert len(tracks) == 2
    assert {t.track_id for t in tracks} == {1, 2}


def test_tracker_maintains_id_across_frames_for_moving_object():
    tracker = MultiObjectTracker()
    tracker.update([_det(0, 0, 10, 10)])
    # Small motion next frame -> should still match via IOU.
    tracks = tracker.update([_det(2, 2, 12, 12)])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].hits == 2


def test_tracker_spawns_new_track_for_unmatched_detection():
    tracker = MultiObjectTracker()
    tracker.update([_det(0, 0, 10, 10)])
    tracks = tracker.update([_det(2, 2, 12, 12), _det(200, 200, 210, 210)])
    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert 1 in ids  # original track continued
    assert 2 in ids  # new track for the unrelated far-away detection


def test_tracker_drops_track_after_max_age_unmatched_frames():
    tracker = MultiObjectTracker(max_age=2)
    tracker.update([_det(0, 0, 10, 10)])

    tracker.update([])  # age 1
    tracker.update([])  # age 2
    tracks = tracker.update([])  # age 3 -> exceeds max_age=2, dropped
    assert tracks == []


def test_tracker_keeps_track_alive_within_max_age_then_rematches():
    tracker = MultiObjectTracker(max_age=2)
    tracker.update([_det(0, 0, 10, 10)])
    tracker.update([])  # missed one frame, age=1, still alive
    tracks = tracker.update([_det(1, 1, 11, 11)])  # reappears, should rematch same ID
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].age == 0


def test_tracker_does_not_match_across_low_iou_even_if_only_option():
    tracker = MultiObjectTracker(iou_threshold=0.5)
    tracker.update([_det(0, 0, 10, 10)])
    # Barely overlapping box -> IOU well below 0.5 -> should NOT match; spawns new track instead.
    tracks = tracker.update([_det(9, 9, 19, 19)])
    assert len(tracks) == 2


def test_track_history_accumulates_centers():
    tracker = MultiObjectTracker()
    tracker.update([_det(0, 0, 10, 10)])
    tracker.update([_det(2, 2, 12, 12)])
    tracks = tracker.update([_det(4, 4, 14, 14)])
    assert len(tracks[0].history) == 3


def test_tracker_reset_clears_state():
    tracker = MultiObjectTracker()
    tracker.update([_det(0, 0, 10, 10)])
    tracker.reset()
    tracks = tracker.update([_det(5, 5, 15, 15)])
    assert tracks[0].track_id == 1  # ID counter restarted
