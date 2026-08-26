from datetime import date, timedelta

import pytest

from data.neows import CloseApproach, NearEarthObject, NeoWsClient


def _fake_record():
    return {
        "id": "3542519", "neo_reference_id": "3542519", "name": "(2010 PK9)",
        "estimated_diameter": {"kilometers": {"estimated_diameter_min": 0.115, "estimated_diameter_max": 0.258}},
        "is_potentially_hazardous_asteroid": True,
        "close_approach_data": [
            {
                "close_approach_date": "2026-08-26", "orbiting_body": "Earth",
                "relative_velocity": {"kilometers_per_second": "11.3"},
                "miss_distance": {"kilometers": "13506722.2", "lunar": "35.1"},
            }
        ],
    }


def test_near_earth_object_from_api_record():
    neo = NearEarthObject.from_api_record(_fake_record())
    assert neo.name == "(2010 PK9)"
    assert neo.is_potentially_hazardous is True
    assert neo.estimated_diameter_min_km == pytest.approx(0.115)
    assert len(neo.close_approaches) == 1
    assert neo.close_approaches[0].miss_distance_km == pytest.approx(13506722.2)


def test_feed_rejects_ranges_over_seven_days():
    client = NeoWsClient(api_key="DEMO_KEY")
    with pytest.raises(ValueError, match="7-day"):
        client.feed(date(2026, 1, 1), date(2026, 1, 10))


@pytest.mark.network
def test_feed_live_returns_objects_for_today():
    client = NeoWsClient(api_key="DEMO_KEY")
    today = date.today()
    objects = client.feed(today, today + timedelta(days=1))
    assert len(objects) > 0
    assert all(isinstance(o, NearEarthObject) for o in objects)
    assert all(o.close_approaches for o in objects)


@pytest.mark.network
def test_lookup_live_known_object():
    client = NeoWsClient(api_key="DEMO_KEY")
    neo = client.lookup("3542519")
    assert neo.neo_reference_id == "3542519"
    assert neo.estimated_diameter_min_km > 0


@pytest.mark.network
def test_closest_approach_live_returns_smallest_miss_distance():
    client = NeoWsClient(api_key="DEMO_KEY")
    today = date.today()
    result = client.closest_approach(today, today + timedelta(days=1))
    assert result is not None
    objects = client.feed(today, today + timedelta(days=1))
    all_min_distances = [min(ca.miss_distance_km for ca in o.close_approaches) for o in objects if o.close_approaches]
    assert min(ca.miss_distance_km for ca in result.close_approaches) == min(all_min_distances)
