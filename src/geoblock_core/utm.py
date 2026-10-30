"""Which projected CRS a touchdown terrain uses, from the one ``zone`` number the build passes around.

The touchdown route grew up in the US, where ``zone`` is a UTM zone 1..60 read as NAD83 / UTM (EPSG:269xx) with
NAVD88 heights (EPSG:5703). Elsewhere the same field holds a WGS 84 / UTM EPSG code (32601-32660 north,
32701-32760 south), so every caller keeps passing one integer and a US build's strings stay exactly what they were.
Light on purpose (pyproj only, lazily): it is imported by the Blender-side and ComfyUI-side helpers too.
"""
from __future__ import annotations

US_VERTICAL_CRS = "EPSG:5703"  # NAVD88 height, the 3DEP products' datum


def is_wgs84_utm(zone: int | str) -> bool:
    z = int(zone)
    return 32601 <= z <= 32660 or 32701 <= z <= 32760


def projected_crs(zone: int | str) -> str:
    """``EPSG:269xx`` for a bare UTM zone (the US route, unchanged), ``EPSG:<zone>`` for a WGS 84 / UTM code."""
    z = int(zone)
    if 1 <= z <= 60:
        return f"EPSG:{26900 + z}"
    if is_wgs84_utm(z):
        return f"EPSG:{z}"
    raise ValueError(f"UTM zone must be 1..60 (NAD83, US) or a WGS 84 / UTM EPSG code 326xx/327xx, got {zone}")


def wgs84_utm_code(lat: float, lon: float) -> int:
    """The WGS 84 / UTM EPSG code holding (lat, lon), with the Norway and Svalbard exceptions to the 6-degree grid
    (Preikestolen, 6.19 E, is in zone 32 either way; Bergen's coast would be 31 without the exception)."""
    if not -80 <= lat <= 84:
        raise ValueError("UTM does not cover the polar regions")
    zone = min(int((lon + 180) // 6) + 1, 60)
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    elif 72 <= lat <= 84 and 0 <= lon < 42:
        zone = 31 if lon < 9 else 33 if lon < 21 else 35 if lon < 33 else 37
    return (32600 if lat >= 0 else 32700) + zone


def vertical_crs(zone: int | str, provider_vertical: str | None = None) -> str:
    """NAVD88 for the US route; elsewhere the height datum the DEM source reports (EGM2008 if none)."""
    if 1 <= int(zone) <= 60:
        return US_VERTICAL_CRS
    return provider_vertical or "EPSG:3855"
