"""
data/nasa_imagery.py — real planetary imagery layers for the 3D dashboard.

Earth — NASA GIBS (Global Imagery Browse Services), WMS, keyless, CORS-open:
    https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi
  * VIIRS_SNPP_CorrectedReflectance_TrueColor — Suomi NPP's daily true-
    colour mosaic. A day's mosaic fills in swath by swath over the
    following hours, so "yesterday" is often still half black: the most
    recent date is picked by measuring how much of it actually exists
    (see pick_true_color_date).
  * BlueMarble_ShadedRelief_Bathymetry — static cloud-free base, used by
    the globe shader to fill whatever swath gaps the daily mosaic has.
  * VIIRS_Black_Marble — night lights, shown on the globe's night side.

Moon / Mars — NASA Solar System Treks, WMTS, keyless, CORS-open:
    https://trek.nasa.gov/tiles/<Body>/EQ/<layer>/1.0.0/default/default028mm/{z}/{y}/{x}.jpg
  Equirectangular: zoom level z is 2^(z+1) x 2^z tiles of 256 px. The
  browser stitches one level into a single texture.

Every layer is equirectangular (plate carrée) and only URLs are served —
the browser loads the pixels straight from NASA. This module's job is to
decide *which* imagery is worth showing and to describe it.
"""

from __future__ import annotations

import io
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import requests
from PIL import Image

from core.resilient_fetch import JsonFileFetcher

GIBS_WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
TRUE_COLOR_LAYER = "VIIRS_SNPP_CorrectedReflectance_TrueColor"
BASE_LAYER = "BlueMarble_ShadedRelief_Bathymetry"
NIGHT_LAYER = "VIIRS_Black_Marble"

TREK_TILE_URL = "https://trek.nasa.gov/tiles/{body}/EQ/{layer}/1.0.0/default/default028mm/{{z}}/{{y}}/{{x}}.jpg"
TREK_SURFACES: dict[str, dict] = {
    "moon": {
        "body": "Moon",
        "layer": "LRO_WAC_Mosaic_Global_303ppd_v02",
        "credit": "NASA/GSFC/Arizona State University — LRO Wide Angle Camera global mosaic",
    },
    "mars": {
        "body": "Mars",
        "layer": "Mars_Viking_MDIM21_ClrMosaic_global_232m",
        "credit": "NASA/JPL/USGS — Viking MDIM 2.1 colour mosaic",
    },
}
TREK_LEVEL = 2  # 8 x 4 tiles -> 2048 x 1024 texture

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "nasa_imagery.json"

# A pixel this dark in every channel is "no data", not dark ocean
# (VIIRS true-colour ocean is ~(10, 20, 50)). The globe shader uses the
# same cut-off when filling gaps from the base layer.
NO_DATA_MAX_CHANNEL = 8
# Coverage is judged on |lat| <= 60 deg: polar night is legitimately
# black for months and would otherwise make every winter date "incomplete".
COVERAGE_LAT_LIMIT_DEG = 60.0


def gibs_map_url(layer: str, width: int, height: int, day: date | None = None) -> str:
    """Whole-globe equirectangular GetMap URL (WMS 1.1.1, so the bbox is lon,lat order)."""
    params = {
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.1.1",
        "LAYERS": layer, "SRS": "EPSG:4326", "BBOX": "-180,-90,180,90",
        "WIDTH": width, "HEIGHT": height, "FORMAT": "image/jpeg",
    }
    if day is not None:
        params["TIME"] = day.isoformat()
    return f"{GIBS_WMS_URL}?{urlencode(params)}"


def measure_coverage(image_bytes: bytes) -> float:
    """Fraction of the |lat| <= 60 deg band that has data, area-weighted by cos(latitude)."""
    pixels = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    height = pixels.shape[0]
    lats = 90.0 - (np.arange(height) + 0.5) * 180.0 / height
    band = np.abs(lats) <= COVERAGE_LAT_LIMIT_DEG
    has_data = pixels[band].max(axis=2) > NO_DATA_MAX_CHANNEL
    weights = np.cos(np.radians(lats[band]))[:, None]
    return float((has_data * weights).sum() / (weights.sum() * pixels.shape[1]))


def pick_true_color_date(
    today: date,
    lookback_days: int = 5,
    min_coverage: float = 0.92,
    timeout: int = 20,
) -> tuple[date, float]:
    """
    The most recent day whose true-colour mosaic is at least `min_coverage`
    complete; if none is, the best-covered day in the window. Each probe is
    a 256x128 thumbnail, so this costs a few KB per candidate date.
    """
    best: tuple[date, float] | None = None
    for offset in range(1, lookback_days + 1):
        day = today - timedelta(days=offset)
        resp = requests.get(gibs_map_url(TRUE_COLOR_LAYER, 256, 128, day), timeout=timeout)
        resp.raise_for_status()
        coverage = measure_coverage(resp.content)
        if coverage >= min_coverage:
            return day, coverage
        if best is None or coverage > best[1]:
            best = (day, coverage)
    return best


def trek_surface(key: str) -> dict:
    meta = TREK_SURFACES[key]
    return {
        "tile_url": TREK_TILE_URL.format(body=meta["body"], layer=meta["layer"]),
        "level": TREK_LEVEL,
        "columns": 2 ** (TREK_LEVEL + 1),
        "rows": 2 ** TREK_LEVEL,
        "tile_size": 256,
        "layer": meta["layer"],
        "credit": meta["credit"],
    }


class NasaImageryClient(JsonFileFetcher[dict]):
    """Resilient-cached description of the imagery layers the dashboard should load."""

    KEY = "layers"

    def __init__(self, cache_path: Path | str = DEFAULT_CACHE_PATH, timeout: int = 20,
                 staleness: timedelta = timedelta(hours=3)):
        super().__init__(cache_path, staleness)
        self.timeout = timeout

    def fetch_layers(self, force: bool = False) -> dict:
        return self.fetch(self.KEY, force=force)

    def _fetch_live(self, key: str) -> dict:
        day, coverage = pick_true_color_date(date.today(), timeout=self.timeout)
        return {
            "earth": {
                "true_color": {
                    "url": gibs_map_url(TRUE_COLOR_LAYER, 4096, 2048, day),
                    "date": day.isoformat(),
                    "coverage": round(coverage, 3),
                    "layer": TRUE_COLOR_LAYER,
                    "credit": "NASA GIBS — Suomi NPP VIIRS corrected reflectance",
                },
                "base": {
                    "url": gibs_map_url(BASE_LAYER, 2048, 1024),
                    "layer": BASE_LAYER,
                    "credit": "NASA GIBS — Blue Marble shaded relief & bathymetry",
                },
                "night": {
                    "url": gibs_map_url(NIGHT_LAYER, 4096, 2048),
                    "layer": NIGHT_LAYER,
                    "credit": "NASA GIBS — VIIRS Black Marble night lights",
                },
                "no_data_max_channel": NO_DATA_MAX_CHANNEL,
            },
            "surfaces": {key: trek_surface(key) for key in TREK_SURFACES},
        }
