"""Public-domain satellite imagery draped on the terrain, from NASA GIBS.

The blocking colours steer render-to-real models, so real imagery of the place beats a
height palette: the rim woodland, side canyons and river land where they are. Layers:

- ``HLS_S30_Nadir_BRDF_Adjusted_Reflectance``: Harmonized Landsat Sentinel-2, 30 m, daily.
  Dates have cloud and swath gaps, so the cleanest dates are stacked until the square is filled.
- USDA NAIP, about 0.6 m aerial photography of the US, from the USGS National Map image service, for the
  1 m lidar landing patch (``fetch_naip_imagery``).
- ``BlueMarble_NextGeneration``: 500 m monthly composite for the whole globe.

NASA imagery is public domain; GIBS asks for this credit, kept in the manifest.
"""

from __future__ import annotations

from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
USER_AGENT = "GeoBlock/0.1 (https://github.com/jamiejk/geoblock-scenecompiler)"
HLS = "HLS_S30_Nadir_BRDF_Adjusted_Reflectance"
HLS_LAYERS = (HLS, "HLS_L30_Nadir_BRDF_Adjusted_Reflectance")  # Sentinel-2 and Landsat passes
BLUE_MARBLE = "BlueMarble_NextGeneration"
CREDIT = ("We acknowledge the use of imagery provided by services from NASA's Global Imagery Browse Services "
          "(GIBS), part of NASA's Earth Science Data and Information System (ESDIS).")


def wms_image(layer: str, bbox: tuple[float, float, float, float], width: int, height: int,
              when: date | None = None, client: httpx.Client | None = None) -> Image.Image:
    """One GIBS GetMap in EPSG:4326; ``bbox`` is (west, south, east, north)."""
    west, south, east, north = bbox
    params = {"SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0", "LAYERS": layer, "CRS": "EPSG:4326",
              "BBOX": f"{south},{west},{north},{east}", "WIDTH": width, "HEIGHT": height, "FORMAT": "image/jpeg"}
    if when:
        params["TIME"] = when.isoformat()
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120)
    try:
        response = client.get(WMS, params=params)
        response.raise_for_status()
        if not response.headers.get("content-type", "").startswith("image/"):
            raise ValueError(f"GIBS returned {response.headers.get('content-type')} for {layer}: {response.text[:200]}")
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    finally:
        if own:
            client.close()


def unusable(image: Image.Image) -> np.ndarray:
    """Swath gaps (black, a little dilated over the JPEG fringe along the swath edge) and cloud (bright, grey)."""
    pixels = np.asarray(image, dtype=np.float32) / 255
    brightness = pixels.mean(axis=2)
    spread = pixels.max(axis=2) - pixels.min(axis=2)
    gap = brightness < 0.04
    for axis in (0, 1):
        gap = gap | np.roll(gap, 1, axis) | np.roll(gap, -1, axis)
    return gap | ((brightness > 0.62) & (spread < 0.12))


def usable_fraction(image: Image.Image) -> float:
    """Share of pixels that are neither swath gaps nor cloud."""
    return float(1 - unusable(image).mean())


def late_summer(latitude: float, today: date | None = None) -> date:
    """End of the last full summer at this latitude: least snow and greenest land."""
    today = today or date.today()
    if latitude >= 0:
        return date(today.year - 1, 9, 30)
    end = date(today.year, 3, 31)
    return end if end < today else date(today.year - 1, 3, 31)


def blend_swath(stack: np.ndarray, pixels: np.ndarray, overlap: np.ndarray, fill: np.ndarray, *,
                feather_px: int) -> np.ndarray:
    """Make a later date's pixels meet the stack without a straight tone edge along the swath boundary: match
    its colour to the stack where both see the ground, then feather the stack toward it just outside the
    filled area. Different dates differ in sun, haze and season, and the first stacked Glacier Point square
    kept a dark diagonal edge between a green and a brown date. Changes ``stack`` in place near the edge and
    returns the colour-matched pixels."""
    if overlap.sum() >= 1000:
        out = pixels.astype(np.float32)
        for channel in range(3):
            source, target = out[..., channel][overlap], stack[..., channel][overlap].astype(np.float32)
            out[..., channel] = (out[..., channel] - source.mean()) / (source.std() or 1.0) * target.std() + target.mean()
        pixels = np.clip(out, 0, 255).astype(np.uint8)
    from PIL import ImageFilter
    reach = np.asarray(Image.fromarray(fill.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(feather_px)),
                       dtype=np.float32) / 255  # 0.5 at the boundary, falling off outside
    edge = overlap & (reach > 0.01)
    weight = reach[edge][:, None]
    stack[edge] = (stack[edge] * (1 - weight) + pixels[edge] * weight).astype(np.uint8)
    return pixels


def clearest_hls(bbox: tuple[float, float, float, float], width: int, height: int, *,
                 end: date | None = None, days: int = 120, step: int = 5, enough: float = 0.985,
                 most_dates: int = 8) -> tuple[Image.Image, list[str], float]:
    """HLS (Sentinel-2 and Landsat layers) from dates back from ``end`` (late summer by default), stacked.

    One date rarely covers a 200 km square: Glacier Point's regional square first showed a dark half with a
    diagonal swath edge, and ranking whole dates by how clean they are then still left 72% of it to Blue
    Marble, because the cleanest dates were the same swath. So small probes of every date pick, greedily,
    the date that adds the most new ground; each is colour-matched and feathered into the stack, and only
    what no date covers comes from Blue Marble, matched and feathered the same way.
    Returns the image, the layer:date list used, and the share covered by HLS."""
    end = end or late_summer((bbox[1] + bbox[3]) / 2)
    feather = max(4, width // 40)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120) as client:
        probe_w, probe_h = max(64, width // 8), max(64, height // 8)
        probes = []
        for offset in range(0, days, step):
            when = end - timedelta(days=offset)
            for layer in HLS_LAYERS:
                try:
                    probes.append((layer, when, ~unusable(wms_image(layer, bbox, probe_w, probe_h, when, client))))
                except (httpx.HTTPError, ValueError):
                    continue  # a layer or date GIBS does not serve
        covered = np.zeros((probe_h, probe_w), dtype=bool)
        chosen = []
        while probes and len(chosen) < most_dates and covered.mean() < enough:
            gain, index = max(((ok & ~covered).sum(), -i) for i, (_, _, ok) in enumerate(probes))
            index = -index  # ties go to the latest date, Sentinel-2 first
            if gain < 0.002 * covered.size:
                break
            layer, when, ok = probes.pop(index)
            chosen.append((layer, when))
            covered |= ok
        stack = np.zeros((height, width, 3), dtype=np.uint8)
        missing = np.ones((height, width), dtype=bool)
        used = []
        for layer, when in chosen:
            image = wms_image(layer, bbox, width, height, when, client)
            ok = ~unusable(image)
            fill = missing & ok
            if not fill.any():
                continue
            pixels = np.asarray(image)
            if used:
                pixels = blend_swath(stack, pixels, ~missing & ok, fill, feather_px=feather)
            stack[fill] = pixels[fill]
            missing &= ~fill
            used.append(f"{layer}:{when.isoformat()}")
        if missing.any():
            base = np.asarray(wms_image(BLUE_MARBLE, bbox, width, height, None, client))
            if used:
                base = blend_swath(stack, base, ~missing, missing, feather_px=feather)
            stack[missing] = base[missing]
        return Image.fromarray(stack), used, float(1 - missing.mean())


def fetch_terrain_imagery(scene: dict, folder: Path, *, pixels: int = 4096, end: date | None = None) -> dict:
    """Satellite texture covering the scene's projected square, warped onto the terrain raster's grid.

    Writes ``imagery.png`` (north up, same extent as terrain_metric.tif) and ``imagery.json``.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds

    folder = folder.resolve()  # Blender runs from another directory
    folder.mkdir(parents=True, exist_ok=True)
    texture, manifest_path = folder / "imagery.png", folder / "imagery.json"
    with rasterio.open(scene["terrain_raster"]) as dem:
        bounds, crs = dem.bounds, dem.crs
    key = hashlib.sha256(json.dumps([list(bounds), str(crs), pixels, str(end), "stacked-greedy"]).encode()).hexdigest()[:16]
    if manifest_path.is_file() and texture.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("key") == key:
            return manifest
    west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    aspect = (east - west) / (north - south)
    width, height = (pixels, max(64, round(pixels / aspect))) if aspect >= 1 else (max(64, round(pixels * aspect)), pixels)
    image, dates, usable = clearest_hls((west, south, east, north), width, height, end=end)
    source = np.moveaxis(np.asarray(image), 2, 0)
    target = np.zeros((3, pixels, pixels), dtype=np.uint8)
    reproject(source, target,
              src_transform=rasterio.transform.from_bounds(west, south, east, north, width, height),
              src_crs="EPSG:4326",
              dst_transform=rasterio.transform.from_bounds(*bounds, pixels, pixels), dst_crs=crs,
              resampling=Resampling.bilinear)
    Image.fromarray(np.moveaxis(target, 0, 2)).save(texture)
    manifest = {"key": key, "texture": str(texture), "layer": "HLS", "dates": dates,
                "usable_fraction": round(usable, 4), "bounds_projected_m": list(bounds), "crs": str(crs),
                "source": WMS, "license": "Public domain (NASA)", "credit": CREDIT}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


NAIP_SERVICE = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"


def naip_image(bounds: tuple[float, float, float, float], epsg: int, pixels: int,
               client: httpx.Client | None = None) -> Image.Image:
    """NAIP natural colour for a projected square, straight from the USGS image service in that CRS."""
    params = {"bbox": ",".join(f"{v:.2f}" for v in bounds), "bboxSR": epsg, "imageSR": epsg,
              "size": f"{pixels},{pixels}", "format": "jpgpng", "interpolation": "RSP_BilinearInterpolation",
              "f": "image"}
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=180)
    try:
        response = client.get(NAIP_SERVICE, params=params)
        response.raise_for_status()
        if not response.headers.get("content-type", "").startswith("image/"):
            raise ValueError(f"NAIP service returned {response.headers.get('content-type')}: {response.text[:200]}")
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    finally:
        if own:
            client.close()


def fade_into(image: np.ndarray, around: np.ndarray, start: float = 0.85) -> np.ndarray:
    """Keep the sharp image's own brightness and contrast, and blend it into the surrounding imagery of the
    same ground only toward its edge (the outer 15%, where blender_earth_zoom feathers the 1 m terrain into
    the local mesh too), so the patch does not show as a square. Matching the whole patch to 30 m HLS
    instead crushed Glacier Point's NAIP to near-black olive: the HLS there is dark and shadowed, and the
    blocking kept giving klein nothing to paint but a dark slope."""
    size = image.shape[0]
    around = np.asarray(Image.fromarray(around).resize((image.shape[1], size), Image.BILINEAR), dtype=np.float32)
    rows, cols = np.indices(image.shape[:2])
    edge = np.maximum(np.abs(cols - (image.shape[1] - 1) / 2) / (image.shape[1] / 2),
                      np.abs(rows - (size - 1) / 2) / (size / 2))
    t = np.clip((edge - start) / (1 - start), 0, 1)
    weight = (t * t * (3 - 2 * t))[..., None]
    return np.clip(image * (1 - weight) + around * weight, 0, 255).astype(np.uint8)


def fetch_naip_imagery(scene: dict, folder: Path, *, pixels: int = 2048, reference: dict | None = None) -> dict:
    """USDA NAIP aerial photography (about 0.6 m, public domain) for a small US scene such as the 1 m lidar
    landing patch, where 30 m HLS gives the keyframe model nothing but colour classes to go on (Glacier
    Point's granite came out as a grassy slope). Keeps NAIP's own colours and fades into ``reference`` (the
    surrounding scene's imagery manifest) over its outer 15%. Same files and manifest shape as ``fetch_terrain_imagery``."""
    import rasterio
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    texture, manifest_path = folder / "imagery.png", folder / "imagery.json"
    with rasterio.open(scene["terrain_raster"]) as dem:
        bounds, crs = tuple(dem.bounds), dem.crs
    key = hashlib.sha256(json.dumps([list(bounds), str(crs), pixels, "naip-faded",
                                     (reference or {}).get("key")]).encode()).hexdigest()[:16]
    if manifest_path.is_file() and texture.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("key") == key:
            return manifest
    image = np.asarray(naip_image(bounds, crs.to_epsg(), pixels))
    if (image.mean(axis=2) < 4).mean() > 0.05:
        raise ValueError("NAIP does not cover this square")
    matched = False
    if reference:
        around = np.asarray(Image.open(reference["texture"]).convert("RGB"))
        left, bottom, right, top = reference["bounds_projected_m"]
        size = around.shape[0]
        col = lambda x: int(round((x - left) / (right - left) * size))
        row = lambda y: int(round((top - y) / (top - bottom) * size))
        crop = around[row(bounds[3]):row(bounds[1]), col(bounds[0]):col(bounds[2])]
        if crop.size >= 3 * 16:
            image, matched = fade_into(image, crop), True
    Image.fromarray(image).save(texture)
    manifest = {"key": key, "texture": str(texture), "layer": "USDA NAIP (USGS National Map image service)",
                "faded_into": reference["texture"] if matched else None, "bounds_projected_m": list(bounds),
                "crs": str(crs), "source": NAIP_SERVICE, "license": "Public domain (USDA)",
                "credit": "USDA Farm Service Agency, National Agriculture Imagery Program (NAIP)"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


GROUND_WORDS = {"rock": "bare pale rock slabs and boulders", "trees": "dark conifers and shrubs",
                "grass": "green grass", "earth": "bare dry earth and gravel", "scrub": "low grey-green scrub",
                "paved": "paved paths and roads"}
# Overture road classes and the paved width each is drawn at, in metres: from above, paving is as pale and
# colourless as granite (Twin Peaks' summit lookout came out as Yosemite rock slabs).
ROAD_WIDTH_M = {"motorway": 14, "trunk": 12, "primary": 11, "secondary": 9, "tertiary": 8, "residential": 7,
                "unclassified": 6, "living_street": 6, "service": 5, "pedestrian": 5, "cycleway": 3, "footway": 2.5,
                "steps": 2.5, "path": 1.5, "bridleway": 1.5, "track": 3}
ROCK_BRIGHTNESS = 0.5  # NAIP mean RGB: Twin Peaks' tan earth is 0.42-0.45, granite and limestone brighter
ROAD_SLACK_M = 3.0  # added to every width: centrelines are a few metres off, and kerbs, verges and pull-outs are paved


def ground_shares(image: np.ndarray, paved: np.ndarray | None = None) -> dict[str, float]:
    """What a natural-colour aerial photo shows, as shares of the sunlit ground: bare rock (bright and
    colourless), trees and shrubs (dark and green), grass (lighter green) and bare earth (the rest).
    Deep shadow is left out. RGB only, so it is a rough split, but it sees what 10 m WorldCover cannot:
    Glacier Point's granite slabs are 'grassland' there. ``paved`` (a mask of mapped roads and paths) takes
    the pale pixels on it out of rock, as their own class."""
    pixels = np.asarray(image, dtype=np.float32)[..., :3] / 255
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    brightness = pixels.mean(axis=2)
    spread = (pixels.max(axis=2) - pixels.min(axis=2)) / np.maximum(pixels.max(axis=2), 1e-3)
    excess_green = (2 * g - r - b) / np.maximum(r + g + b, 1e-3)
    lit = (brightness >= 0.15) & ~((brightness < 0.3) & (b >= g))  # cliff and tree shadows are dark and blue
    dark = brightness < 0.32
    classes = {"trees": dark & (g >= 0.95 * r) & (g > 1.1 * b), "grass": ~dark & (excess_green > 0.08)}
    # Rock is pale (granite, limestone): brighter than dry soil, which read as rock at Twin Peaks by colour alone.
    # Warm granite is as tinted as tan earth, so brightness, not saturation, tells them apart.
    classes["rock"] = ~dark & ~classes["grass"] & (spread < 0.25) & (brightness >= ROCK_BRIGHTNESS)
    classes["scrub"] = (~dark & ~classes["grass"] & ~classes["rock"] & (brightness < 0.45)
                        & (g >= r) & (g >= b) & (excess_green > 0.02))
    classes["earth"] = ~classes["trees"] & ~classes["grass"] & ~classes["rock"] & ~classes["scrub"]
    if paved is not None:
        classes["paved"] = paved & (classes["rock"] | classes["earth"] | classes["scrub"])
        classes["rock"] &= ~paved
        classes["earth"] &= ~paved
        classes["scrub"] &= ~paved
    total = max(int(lit.sum()), 1)
    return {name: round(float((mask & lit).sum() / total), 3) for name, mask in classes.items()}


def describe_ground(shares: dict[str, float], floor: float = 0.1) -> str:
    """'mostly bare pale rock slabs and boulders, with dark conifers and shrubs' from ground_shares."""
    ranked = [(name, share) for name, share in sorted(shares.items(), key=lambda item: -item[1]) if share >= floor]
    ranked = ranked[:3] or sorted(shares.items(), key=lambda item: -item[1])[:1]
    words = [GROUND_WORDS[name] for name, _ in ranked]
    lead = "mostly " if ranked[0][1] >= 0.5 else ""
    if len(words) == 1:
        return lead + words[0]
    if len(words) == 2:
        return f"{lead}{words[0]}, with {words[1]}"
    return f"{lead}{words[0]}, with {words[1]} and some {words[2]}"


def paved_mask(bounds: tuple[float, float, float, float], epsg: int, pixels: int, cache: Path) -> np.ndarray:
    """Mapped roads and paths (Overture transportation segments, each at its class's ROAD_WIDTH_M) over a
    projected square, as a ``pixels`` x ``pixels`` mask, row 0 north."""
    import shapely
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from .buildings import read_overture

    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    to_xy = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    margin = 1500  # Overture is filtered by each segment's bounding-box centre: reach long roads too
    corners = [to_ll.transform(x, y) for x in (bounds[0] - margin, bounds[2] + margin)
               for y in (bounds[1] - margin, bounds[3] + margin)]
    box = (min(c[0] for c in corners), min(c[1] for c in corners), max(c[0] for c in corners),
           max(c[1] for c in corners))
    rows = read_overture("segment", box, cache, ["geometry", "subtype", "class"], theme="transportation")
    square = shapely.box(*bounds)
    shapes = []
    for row in rows:
        width = ROAD_WIDTH_M.get(row.get("class") or "") if row.get("subtype") == "road" else None
        if not width:
            continue
        line = shapely.transform(shapely.from_wkb(row["geometry"]),
                                 lambda xy: np.column_stack(to_xy.transform(xy[:, 0], xy[:, 1])))
        if line.distance(square) < width:
            shapes.append((line.buffer(width / 2 + ROAD_SLACK_M, cap_style="flat"), 1))
    if not shapes:
        return np.zeros((pixels, pixels), dtype=bool)
    return rasterize(shapes, out_shape=(pixels, pixels), transform=from_bounds(*bounds, pixels, pixels),
                     all_touched=True).astype(bool)


def naip_ground(latitude: float, longitude: float, cache: Path, *, radius_m: float = 40,
                pixels: int = 160) -> dict:
    """The ground within ``radius_m`` of a US point, measured on NAIP (ground_shares), for the prompt words
    about the ground underfoot. Cached by point. Raises ValueError where NAIP has no cover."""
    from rasterio.warp import transform
    path = cache / "naip_ground" / f"{latitude:.6f}_{longitude:.6f}_{radius_m:g}_roads.json"
    if path.is_file():
        return json.loads(path.read_text())
    epsg = 32600 + int((longitude + 180) // 6) + 1
    (east,), (north,) = transform("EPSG:4326", f"EPSG:{epsg}", [longitude], [latitude])
    bounds = (east - radius_m, north - radius_m, east + radius_m, north + radius_m)
    image = np.asarray(naip_image(bounds, epsg, pixels))
    if (image.mean(axis=2) < 4).mean() > 0.05:
        raise ValueError("NAIP does not cover this point")
    try:
        paved = paved_mask(bounds, epsg, image.shape[0], cache)
    except Exception as error:  # the photo alone still gives words; pale paving may then read as rock
        print(f"Roads for the ground underfoot skipped: {error}", flush=True)
        paved = None
    shares = ground_shares(image, paved)
    result = {"shares": shares, "words": describe_ground(shares), "radius_m": radius_m,
              "source": "USDA NAIP (USGS National Map image service)"
                        + (", roads from Overture Maps transportation" if paved is not None else ""),
              "license": "Public domain (USDA)" + ("; ODbL 1.0 (roads)" if paved is not None else "")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result
