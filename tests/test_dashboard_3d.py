import json
from datetime import datetime, timedelta, timezone

from core.tle_manager import Satellite
from viz.dashboard_3d import add_ground_tracks, build_snapshot, snapshot_to_dict, write_snapshot_json

LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
LINE2 = "2 00005  34.2682 348.7242 1627203 289.2321  55.4550 10.82419157413667"


def _make_sat():
    return Satellite(
        norad_id=5, name="TEST-SAT", line1=LINE1, line2=LINE2,
        epoch=datetime(2000, 6, 27, 18, 50, 19, tzinfo=timezone.utc),
        inclination_deg=34.2682, eccentricity=0.1627203,
        mean_motion_rev_per_day=10.82419157, semi_major_axis_km=8617.0,
        classification="active",
    )


def test_build_snapshot_frame_count_and_metadata():
    sat = _make_sat()
    epoch = datetime(2000, 6, 27, 18, 50, 19, tzinfo=timezone.utc)
    snapshot = build_snapshot([sat], epoch, n_frames=5, frame_step=timedelta(minutes=10))

    assert len(snapshot.frame_times_iso) == 5
    assert len(snapshot.subsolar_track) == 5
    assert len(snapshot.terminator_tracks) == 5
    assert len(snapshot.satellites) == 1

    sat_track = snapshot.satellites[0]
    assert sat_track.norad_id == 5
    assert sat_track.classification == "active"
    assert len(sat_track.track) == 5
    for lat, lon, alt in sat_track.track:
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 180.0
        assert alt > 0


def test_build_snapshot_drops_satellite_that_fails_every_frame():
    bad_sat = Satellite(
        norad_id=99999, name="BROKEN", line1="garbage", line2="garbage",
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        inclination_deg=0.0, eccentricity=0.0, mean_motion_rev_per_day=15.0,
        semi_major_axis_km=7000.0, classification="debris",
    )
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    snapshot = build_snapshot([bad_sat], epoch, n_frames=3, frame_step=timedelta(minutes=10))
    assert snapshot.satellites == []


def test_snapshot_to_dict_is_json_serializable():
    sat = _make_sat()
    epoch = datetime(2000, 6, 27, 18, 50, 19, tzinfo=timezone.utc)
    snapshot = build_snapshot([sat], epoch, n_frames=3, frame_step=timedelta(minutes=30))
    d = snapshot_to_dict(snapshot)
    serialized = json.dumps(d)
    assert len(serialized) > 0


def test_add_ground_tracks_populates_segments_for_given_satellite():
    sat = _make_sat()
    epoch = datetime(2000, 6, 27, 18, 50, 19, tzinfo=timezone.utc)
    snapshot = build_snapshot([sat], epoch, n_frames=2, frame_step=timedelta(hours=1))

    add_ground_tracks(snapshot, [sat], epoch, step=timedelta(minutes=2))

    assert sat.norad_id in snapshot.ground_tracks
    segments = snapshot.ground_tracks[sat.norad_id]
    assert len(segments) >= 1
    for seg in segments:
        for lat, lon in seg:
            assert -90.0 <= lat <= 90.0
            assert -180.0 <= lon <= 180.0


def test_add_ground_tracks_is_json_serializable_via_snapshot_to_dict():
    sat = _make_sat()
    epoch = datetime(2000, 6, 27, 18, 50, 19, tzinfo=timezone.utc)
    snapshot = build_snapshot([sat], epoch, n_frames=2, frame_step=timedelta(hours=1))
    add_ground_tracks(snapshot, [sat], epoch, step=timedelta(minutes=5))

    d = snapshot_to_dict(snapshot)
    serialized = json.dumps(d)
    assert str(sat.norad_id) in serialized or sat.norad_id in d["ground_tracks"]


def test_write_snapshot_json_writes_valid_file(tmp_path):
    sat = _make_sat()
    epoch = datetime(2000, 6, 27, 18, 50, 19, tzinfo=timezone.utc)
    snapshot = build_snapshot([sat], epoch, n_frames=2, frame_step=timedelta(hours=1))

    out_path = tmp_path / "snapshot.json"
    write_snapshot_json(snapshot, out_path)

    assert out_path.exists()
    with open(out_path) as f:
        data = json.load(f)
    assert len(data["satellites"]) == 1
    assert len(data["frame_times_iso"]) == 2
