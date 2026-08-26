import pytest

from data.sbdb import OrbitalElements, SBDBClient, SmallBody


def test_orbital_elements_from_orbit_record():
    orbit = {
        "epoch": "2461200.5",
        "elements": [
            {"name": "e", "value": ".2228779627700761"},
            {"name": "a", "value": "1.458243716760167"},
            {"name": "q", "value": "1.133233327946397"},
            {"name": "i", "value": "10.82854410314273"},
            {"name": "om", "value": "304.2679713350896"},
            {"name": "w", "value": "178.9181319135911"},
            {"name": "ma", "value": "62.51145501986792"},
            {"name": "per", "value": "643.1963890927677"},
        ],
    }
    els = OrbitalElements.from_orbit_record(orbit)
    assert els.eccentricity == pytest.approx(0.22287796277)
    assert els.semi_major_axis_au == pytest.approx(1.45824371676)
    assert els.epoch_jd == 2461200.5


def test_small_body_from_api_response():
    payload = {
        "object": {
            "des": "433", "fullname": "433 Eros (A898 PA)",
            "orbit_class": {"name": "Amor", "code": "AMO"},
            "neo": True, "pha": False,
        },
        "orbit": {
            "epoch": "2461200.5",
            "elements": [
                {"name": "e", "value": "0.22"}, {"name": "a", "value": "1.45"},
                {"name": "q", "value": "1.13"}, {"name": "i", "value": "10.8"},
                {"name": "om", "value": "304.2"}, {"name": "w", "value": "178.9"},
                {"name": "ma", "value": "62.5"}, {"name": "per", "value": "643.1"},
            ],
        },
    }
    body = SmallBody.from_api_response(payload)
    assert body.designation == "433"
    assert body.orbit_class_name == "Amor"
    assert body.is_neo is True
    assert body.is_potentially_hazardous is False


@pytest.mark.network
def test_lookup_eros_live():
    client = SBDBClient()
    body = client.lookup("433")
    assert body.designation == "433"
    assert "Eros" in body.full_name
    assert body.elements.semi_major_axis_au == pytest.approx(1.458, abs=0.05)


@pytest.mark.network
def test_lookup_unknown_designation_raises_live():
    client = SBDBClient()
    with pytest.raises(ValueError, match="no match"):
        client.lookup("zzzznotarealdesignation")


def test_lookup_ambiguous_response_raises(monkeypatch):
    """
    SBDB returns a `list` (not `object`) payload when a query string
    matches multiple bodies ambiguously. Exercised with a mocked response
    since we can't rely on any specific search string reproducing that
    live (most name searches resolve directly to a single best match).
    """
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"list": [{"pdes": "433", "name": "Eros"}, {"pdes": "1998 XY1", "name": "1998 XY1"}]}

    client = SBDBClient()
    monkeypatch.setattr("data.sbdb.requests.get", lambda *a, **k: FakeResponse())
    with pytest.raises(ValueError, match="ambiguous"):
        client.lookup("something_ambiguous")
