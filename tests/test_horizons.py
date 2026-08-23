import pytest

from data.horizons import HorizonsClient


def test_parse_vector_table_handles_minimal_block():
    raw = """
Some header text
$$SOE
2460000.500000000, A.D. 2023-Feb-25 00:00:00.0000,  1.0, 2.0, 3.0, 0.1, 0.2, 0.3,
2460001.500000000, A.D. 2023-Feb-26 00:00:00.0000,  1.1, 2.1, 3.1, 0.1, 0.2, 0.3,
$$EOE
Footer text
"""
    samples = HorizonsClient._parse_vector_table(raw)
    assert len(samples) == 2
    assert samples[0].jd_tdb == 2460000.5
    assert samples[0].r_km == (1.0, 2.0, 3.0)
    assert samples[0].v_km_s == (0.1, 0.2, 0.3)


def test_parse_vector_table_raises_without_markers():
    with pytest.raises(ValueError):
        HorizonsClient._parse_vector_table("no markers here")


@pytest.mark.network
def test_fetch_vectors_earth_live():
    client = HorizonsClient()
    samples = client.fetch_vectors("earth", "2026-01-01", "2026-01-03", step_size="1d")
    assert len(samples) >= 2
    # Earth-Sun distance should be roughly 1 AU (~1.47e8 km at perihelion-ish in January).
    r = samples[0].r_km
    dist = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5
    assert 1.40e8 < dist < 1.55e8
