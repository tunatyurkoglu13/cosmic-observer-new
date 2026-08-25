from viz.retro_ui import (
    ALERT_COLORS,
    CLASSIFICATION_COLORS,
    PALETTE,
    generate_crt_flicker_css,
    generate_hex_grid_svg,
    generate_holo_panel_css,
    generate_scanline_css,
    generate_theme_css,
)


def test_palette_has_required_colors():
    for key in ("background", "cyan", "magenta", "amber", "phosphor_green"):
        assert key in PALETTE
        assert PALETTE[key].startswith("#")


def test_classification_and_alert_colors_reference_valid_hex():
    for value in list(CLASSIFICATION_COLORS.values()) + list(ALERT_COLORS.values()):
        assert value.startswith("#")
        assert len(value) in (4, 7)  # #fff or #ffffff


def test_generate_theme_css_contains_all_palette_entries():
    css = generate_theme_css()
    assert ":root" in css
    for value in PALETTE.values():
        assert value in css


def test_generate_scanline_css_is_valid_looking_css():
    css = generate_scanline_css()
    assert ".co-scanlines" in css
    assert "repeating-linear-gradient" in css
    assert "pointer-events: none" in css


def test_generate_crt_flicker_css_has_keyframes():
    css = generate_crt_flicker_css()
    assert "@keyframes co-crt-flicker" in css
    assert ".co-crt-flicker" in css


def test_generate_holo_panel_css_has_clip_path_and_border():
    css = generate_holo_panel_css()
    assert ".co-panel" in css
    assert "clip-path" in css
    assert PALETTE["cyan"] in css


def test_generate_hex_grid_svg_is_well_formed():
    svg = generate_hex_grid_svg(width=100, height=100, hex_size=10)
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "polygon" in svg
    assert 'width="100"' in svg
    assert 'height="100"' in svg


def test_generate_hex_grid_svg_scales_with_size():
    small = generate_hex_grid_svg(width=50, height=50, hex_size=25)
    large = generate_hex_grid_svg(width=400, height=400, hex_size=10)
    assert small.count("polygon") < large.count("polygon")
