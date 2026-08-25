from datetime import datetime, timezone

from core.tle_manager import Satellite, TLEManager

LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
LINE2 = "2 00005  34.2682 348.7242 1627203 289.2321  55.4550 10.82419157413667"


def _make_sat(norad_id, classification):
    return Satellite(
        norad_id=norad_id, name=f"SAT-{norad_id}", line1=LINE1, line2=LINE2,
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        inclination_deg=34.27, eccentricity=0.1627, mean_motion_rev_per_day=10.82,
        semi_major_axis_km=8617.0, classification=classification,
    )


def test_satellite_belonging_to_two_groups_is_returned_for_both(tmp_path):
    """
    Regression test: a satellite fetched under group A and later also
    fetched under group B (e.g. the ISS appearing in both "stations" and
    "visual") must remain visible to load_cached("A") even after the "B"
    fetch — group membership is per-group, not a single overwritable field.
    """
    mgr = TLEManager(db_path=tmp_path / "test_cache.sqlite3")

    sat_as_stations = _make_sat(25544, "stations")
    mgr._upsert([sat_as_stations])

    sat_as_visual = _make_sat(25544, "visual")
    mgr._upsert([sat_as_visual])

    stations_result = mgr.load_cached(classification="stations")
    visual_result = mgr.load_cached(classification="visual")

    assert any(s.norad_id == 25544 for s in stations_result)
    assert any(s.norad_id == 25544 for s in visual_result)


def test_load_cached_classification_field_matches_queried_group(tmp_path):
    mgr = TLEManager(db_path=tmp_path / "test_cache.sqlite3")
    mgr._upsert([_make_sat(25544, "stations")])
    mgr._upsert([_make_sat(25544, "visual")])

    stations_result = mgr.load_cached(classification="stations")
    assert stations_result[0].classification == "stations"

    visual_result = mgr.load_cached(classification="visual")
    assert visual_result[0].classification == "visual"


def test_load_cached_no_filter_returns_first_seen_classification(tmp_path):
    mgr = TLEManager(db_path=tmp_path / "test_cache.sqlite3")
    mgr._upsert([_make_sat(25544, "stations")])
    mgr._upsert([_make_sat(25544, "visual")])  # should NOT overwrite the stored classification

    unfiltered = mgr.load_cached()
    assert len(unfiltered) == 1
    assert unfiltered[0].classification == "stations"


def test_satellite_from_tle_parses_norad_id_and_epoch():
    sat = Satellite.from_tle("TEST", LINE1, LINE2, classification="active")
    assert sat.norad_id == 5
    assert sat.inclination_deg == 34.2682
    assert sat.semi_major_axis_km > 0
