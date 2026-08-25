"""
viz/retro_ui.py — Retro-futuristic UI theme: palette, CSS, and decorative SVG.

Generates the visual assets for the dashboard's "hologram HUD" aesthetic
(per project spec: Star Wars-hologram-style angled panels, Evangelion-style
hexagonal grid overlays, CRT scanlines/flicker), as plain strings so they
can be written to static asset files (or embedded inline) by whatever
build step assembles the actual dashboard page.

Kept in Python (rather than hand-written CSS) so the palette is a single
source of truth shared by anything else in the platform that needs these
colors (e.g. a future matplotlib/plotly report using the same theme).
"""

from __future__ import annotations

# Core palette (project spec).
PALETTE = {
    "background": "#0a0a14",
    "cyan": "#00ffff",
    "magenta": "#ff0066",
    "amber": "#ffcc00",
    "phosphor_green": "#00ff66",
}

FONT_STACK = "'JetBrains Mono', 'IBM Plex Mono', 'Fira Code', ui-monospace, monospace"

GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=JetBrains+Mono:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap"
)

# Classification -> marker color, used consistently across the 3D scene and any 2D charts.
CLASSIFICATION_COLORS = {
    "active": PALETTE["cyan"],
    "debris": PALETTE["magenta"],
    "stations": PALETTE["amber"],
    "iss": PALETTE["phosphor_green"],
}

ALERT_COLORS = {
    "emergency": "#ff0033",
    "high": PALETTE["magenta"],
    "medium": PALETTE["amber"],
    "info": PALETTE["cyan"],
}


def generate_theme_css() -> str:
    """CSS custom properties for the full retro palette + font stack, for a page-wide :root block."""
    lines = [":root {"]
    for name, value in PALETTE.items():
        lines.append(f"  --co-{name.replace('_', '-')}: {value};")
    for name, value in ALERT_COLORS.items():
        lines.append(f"  --co-alert-{name}: {value};")
    lines.append(f"  --co-font: {FONT_STACK};")
    lines.append("}")
    return "\n".join(lines)


def generate_scanline_css(opacity: float = 0.06, line_height_px: int = 3) -> str:
    """
    CSS for a full-viewport scanline overlay: a repeating horizontal
    gradient over a fixed, click-through (pointer-events: none) layer.
    Apply by adding an empty <div class="co-scanlines"></div> to the page.
    """
    return f"""
.co-scanlines {{
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 9999;
  background: repeating-linear-gradient(
    to bottom,
    rgba(0, 0, 0, 0) 0px,
    rgba(0, 0, 0, 0) {line_height_px - 1}px,
    rgba(0, 0, 0, {opacity}) {line_height_px}px
  );
  mix-blend-mode: overlay;
}}
""".strip()


def generate_crt_flicker_css(duration_s: float = 6.0) -> str:
    """CSS keyframes for a subtle CRT brightness flicker, applied to a full-viewport overlay div."""
    return f"""
@keyframes co-crt-flicker {{
  0%   {{ opacity: 0.0; }}
  3%   {{ opacity: 0.02; }}
  6%   {{ opacity: 0.0; }}
  47%  {{ opacity: 0.0; }}
  50%  {{ opacity: 0.03; }}
  53%  {{ opacity: 0.0; }}
  100% {{ opacity: 0.0; }}
}}
.co-crt-flicker {{
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 9998;
  background: {PALETTE['cyan']};
  animation: co-crt-flicker {duration_s}s infinite;
}}
""".strip()


def generate_holo_panel_css() -> str:
    """
    CSS for an angled, holographic-bordered panel (the "Star Wars
    hologram" HUD panel look): a translucent dark background, a glowing
    cyan border, and a slight skew on two corners via clip-path.
    """
    return f"""
.co-panel {{
  background: rgba(10, 10, 20, 0.72);
  border: 1px solid {PALETTE['cyan']};
  box-shadow: 0 0 8px {PALETTE['cyan']}66, inset 0 0 12px {PALETTE['cyan']}22;
  color: {PALETTE['cyan']};
  font-family: {FONT_STACK};
  padding: 12px 16px;
  clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 14px 100%, 0 calc(100% - 14px));
  backdrop-filter: blur(2px);
}}
.co-panel h1, .co-panel h2, .co-panel h3 {{
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin: 0 0 8px 0;
}}
""".strip()


def generate_hex_grid_svg(width: int = 400, height: int = 400, hex_size: int = 24, stroke_opacity: float = 0.15) -> str:
    """
    Generate a tileable hexagonal-grid SVG (Evangelion-style decorative
    overlay), suitable for use as a CSS background-image data URI or an
    inline <svg> background layer.

    Args:
        width, height: SVG canvas size [px].
        hex_size: hexagon "radius" (center to vertex) [px].
        stroke_opacity: line opacity, 0-1.

    Returns:
        A complete <svg>...</svg> string.
    """
    import math

    hex_w = hex_size * 2
    hex_h = math.sqrt(3) * hex_size

    paths = []
    row = 0
    y = 0.0
    while y < height + hex_h:
        x_offset = (hex_w * 0.75) if row % 2 == 0 else 0.0
        x = -hex_w
        while x < width + hex_w:
            cx, cy = x + x_offset, y
            points = []
            for k in range(6):
                angle = math.radians(60 * k)
                points.append(f"{cx + hex_size * math.cos(angle):.1f},{cy + hex_size * math.sin(angle):.1f}")
            paths.append(f'<polygon points="{" ".join(points)}" />')
            x += hex_w * 1.5
        y += hex_h
        row += 1

    body = "\n".join(paths)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<g fill="none" stroke="{PALETTE["cyan"]}" stroke-opacity="{stroke_opacity}" stroke-width="1">'
        f"{body}</g></svg>"
    )
