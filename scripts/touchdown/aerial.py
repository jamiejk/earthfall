"""Open aerial imagery for the dive squares, by country (Jamie, 11 Oct: world coverage for the place picker).

The dive (touchdown/dive.py) needs nested squares from 72.9 km down to 300 m at 2048 px, i.e. 36 m down to 0.15 m per
pixel, so only sources of about 0.6 m or finer carry the low part of it (docs/review/coverage/world-coverage.md: a
10 m Sentinel-2 dive turns to mush below about 2 km). Sources:

- ``naip``          USA, USDA NAIP 0.6 m via the USGS image service (nadir_dive._naip), public domain
- ``ign-fr``        France, IGN BD ORTHO 20 cm, Géoplateforme WMTS, Licence Ouverte 2.0
- ``pnoa-es``       Spain, IGN/CNIG PNOA máxima actualidad 15-25 cm, WMTS, CC BY 4.0
- ``swissimage-ch`` Switzerland, swisstopo SWISSIMAGE 10 cm, WMTS, swisstopo open-data terms (attribution)
- ``pdok-nl``       Netherlands, PDOK Luchtfoto actueel HR 8 cm, WMTS, CC BY 4.0
- ``gsi-jp``        Japan, GSI seamless photo tiles (zoom 18, ~0.5 m), GSI website terms (attribution)
- ``bz-it``         Italy, South Tyrol only: Provincia di Bolzano orthophoto 2023 (20 cm), WMS in EPSG:3857, no access
                    constraints in its capabilities (the rest of Italy gets the fill)
- ``basemap-at``    Austria, basemap.at orthofoto (about 30 cm), WMTS
- ``dop20-by``      Germany, Bavaria only: LDBV DOP20 (20 cm), WMS in EPSG:3857, CC BY 4.0 (other states get the fill)
- ``hk-imagery``    Hong Kong, Lands Department imagery tiles (zoom 19)
- ``linz-nz``       New Zealand, LINZ aerial photos 5-50 cm from the open nz-imagery bucket (COGs), CC BY 4.0, no key
                    (touchdown.aerial_open); squares of 4.5 m/px and finer
- ``linz-nz-basemaps`` the same photos through the LINZ Basemaps tile API; needs a free key in ``LINZ_BASEMAPS_KEY``
- ``s2-l2a``        whole globe, Copernicus Sentinel-2 L2A true colour (10 m) from the sentinel-cogs bucket, a median of
                    a few low-cloud summer scenes (touchdown.aerial_open); free for any use: the default fill
- ``s2cloudless``   whole globe, EOX Sentinel-2 cloudless 2023 (10 m), CC BY-NC-SA 4.0 (non-commercial): only on request,
                    as ``imagery = "s2cloudless"`` or ``TOUCHDOWN_IMAGERY_FILL=s2cloudless``

A square is drawn from a chain: the place's own source, then any other national source whose area touches the square
(a Swiss landing's 72.9 km square reaches into France and Italy), then the fill (``s2-l2a`` unless
``TOUCHDOWN_IMAGERY_FILL`` names another source). Each source fills only what
the ones before it left empty (outside its country, sea, missing tiles), so the returned alpha marks only what no
source has. Tile servers all serve Web Mercator; tiles are mosaicked at the zoom whose pixel matches the square's and
resampled to the plate carree grid ``dive.fetch`` expects from ``nadir_dive._naip``.

These run on the desktop: the cloud containers cannot reach the national tile servers (the AWS buckets behind
``linz-nz`` and ``s2-l2a`` do answer there).
"""
from __future__ import annotations

import io
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image

USER_AGENT = "GeoBlock/0.1 (https://github.com/jamiejk/geoblock-scenecompiler)"
TILE = 256
EARTH_M = 2 * math.pi * 6378137.0  # Web Mercator equator length
WORKERS = 6


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    country: str                                       # ISO code the catalog uses ("" = global)
    areas: tuple[tuple[float, float, float, float], ...]  # (west, south, east, north) boxes it can cover
    template: str = ""                                 # {z} {x} {y} XYZ tile URL ({key} from ``key_env``), or a
                                                       # WMS GetMap URL with {bbox}: the tile's EPSG:3857 bounds
    max_zoom: int = 19
    metres: float = 0.0                                # native ground resolution, for the manifest
    licence: str = ""
    credit: str = ""
    key_env: str = ""
    neighbour_min_mpp: float = 0.0                     # as a neighbour, only for squares at least this coarse (m/px)
    max_mpp: float = math.inf                          # only for squares at least this fine (m/px)
    fetch: Callable | None = field(default=None, compare=False)  # (bbox, pixels) -> RGBA, instead of tiles

    def available(self) -> bool:
        return not self.key_env or bool(os.environ.get(self.key_env))

    def touches(self, bbox: tuple[float, float, float, float]) -> bool:
        w, s, e, n = bbox
        return any(w < ae and e > aw and s < an and n > as_ for aw, as_, ae, an in self.areas)


def _naip(bbox, pixels):
    import nadir_dive
    return nadir_dive._naip(bbox, pixels)


def _linz(bbox, pixels):
    from . import aerial_open
    return aerial_open.linz_rgba(bbox, pixels)


def _s2(bbox, pixels):
    from . import aerial_open
    return aerial_open.s2_l2a_rgba(bbox, pixels)


SOURCES: dict[str, Source] = {s.id: s for s in (
    Source("naip", "USDA NAIP", "US", ((-125.0, 24.4, -66.9, 49.5), (-160.3, 18.9, -154.8, 22.3)),
           metres=0.6, licence="Public domain (USDA)", credit="USDA National Agriculture Imagery Program, via USGS",
           fetch=_naip),
    Source("ign-fr", "IGN BD ORTHO", "FR", ((-5.3, 41.3, 9.6, 51.15),),
           "https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS"
           "&STYLE=normal&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/jpeg",
           max_zoom=19, metres=0.2, licence="Licence Ouverte / Etalab 2.0", credit="IGN, BD ORTHO"),
    Source("pnoa-es", "PNOA máxima actualidad", "ES", ((-9.4, 35.9, 4.4, 43.8), (-18.2, 27.6, -13.4, 29.5)),
           "https://www.ign.es/wmts/pnoa-ma?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=OI.OrthoimageCoverage"
           "&STYLE=default&TILEMATRIXSET=GoogleMapsCompatible&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/jpeg",
           max_zoom=19, metres=0.25, licence="CC BY 4.0", credit="PNOA cedido por © Instituto Geográfico Nacional"),
    Source("swissimage-ch", "swisstopo SWISSIMAGE", "CH", ((5.95, 45.8, 10.5, 47.81),),
           "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{z}/{x}/{y}.jpeg",
           max_zoom=20, metres=0.1, licence="swisstopo open data (free use with attribution)", credit="© swisstopo",
           neighbour_min_mpp=3.5),  # outside Switzerland it serves a blocky ~5 m backdrop (north-face shadow went navy
                                    # in an Aiguille du Midi hole, 11 Oct), so it only fills coarse squares abroad
    Source("pdok-nl", "PDOK Luchtfoto actueel HR", "NL", ((3.3, 50.75, 7.25, 53.6),),
           "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg",
           max_zoom=19, metres=0.08, licence="CC BY 4.0", credit="Beeldmateriaal.nl / PDOK"),
    Source("gsi-jp", "GSI seamless photo", "JP", ((122.9, 24.0, 146.0, 45.6),),
           "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg",
           max_zoom=18, metres=0.5, licence="GSI website terms (attribution)", credit="地理院タイル (GSI Japan)"),
    Source("bz-it", "Provincia di Bolzano orthophoto 2023", "IT", ((10.38, 46.22, 12.48, 47.09),),
           "https://geoservices.buergernetz.bz.it/mapproxy/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
           "&LAYERS=p_bz-Orthoimagery:Aerial-2023-RGB&STYLES=&CRS=EPSG:3857&BBOX={bbox}&WIDTH=256&HEIGHT=256"
           "&FORMAT=image/jpeg", max_zoom=19, metres=0.2, licence="Provincia autonoma di Bolzano open data",
           credit="Orthophoto 2023, Autonome Provinz Bozen - Südtirol"),
    Source("basemap-at", "basemap.at orthofoto", "AT", ((9.5, 46.37, 17.17, 49.02),),
           "https://mapsneu.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/{z}/{y}/{x}.jpeg",
           max_zoom=19, metres=0.3, licence="CC BY 4.0", credit="basemap.at"),
    Source("dop20-by", "Bayern DOP20", "DE", ((8.97, 47.27, 13.84, 50.57),),
           "https://geoservices.bayern.de/od/wms/dop/v1/dop20?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
           "&LAYERS=by_dop20c&STYLES=&CRS=EPSG:3857&BBOX={bbox}&WIDTH=256&HEIGHT=256&FORMAT=image/jpeg",
           max_zoom=19, metres=0.2, licence="CC BY 4.0", credit="Bayerische Vermessungsverwaltung, DOP20"),
    Source("hk-imagery", "Hong Kong Lands Department imagery", "HK", ((113.8, 22.13, 114.45, 22.57),),
           "https://mapapi.geodata.gov.hk/gs/api/v1.0.0/xyz/imagery/WGS84/{z}/{x}/{y}.png",
           max_zoom=19, metres=0.5, licence="Hong Kong Lands Department map API terms (attribution)",
           credit="© Map from Lands Department, HKSAR"),
    Source("linz-nz", "LINZ aerial imagery (nz-imagery)", "NZ", ((166.0, -47.4, 178.7, -34.2),),
           metres=0.25, licence="CC BY 4.0", credit="Sourced from the LINZ Data Service and licensed for reuse under "
           "CC BY 4.0", max_mpp=4.5, fetch=_linz),  # COGs per 1:5000 sheet tile: too many to read for coarser squares
    Source("linz-nz-basemaps", "LINZ Basemaps aerial", "NZ", ((166.0, -47.4, 178.7, -34.2),),
           "https://basemaps.linz.govt.nz/v1/tiles/aerial/WebMercatorQuad/{z}/{x}/{y}.webp?api={key}",
           max_zoom=20, metres=0.3, licence="CC BY 4.0", credit="Sourced from LINZ. CC BY 4.0", key_env="LINZ_BASEMAPS_KEY"),
    Source("s2-l2a", "Copernicus Sentinel-2 L2A (median of low-cloud scenes)", "", ((-180.0, -85.0, 180.0, 85.0),),
           metres=10.0, licence="Copernicus data: free and open for any use, including commercial",
           credit="Contains modified Copernicus Sentinel data (ESA), via the Sentinel-2 COGs on AWS Open Data",
           fetch=_s2),
    Source("s2cloudless", "EOX Sentinel-2 cloudless 2023", "", ((-180.0, -85.0, 180.0, 85.0),),
           "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg",
           max_zoom=15, metres=10.0, licence="CC BY-NC-SA 4.0",
           credit="Sentinel-2 cloudless 2023 by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2023)"),
)}
FALLBACK = "s2-l2a"


def fill() -> str:
    """The global fill under every chain: Copernicus Sentinel-2 L2A unless ``TOUCHDOWN_IMAGERY_FILL`` names another."""
    chosen = os.environ.get("TOUCHDOWN_IMAGERY_FILL") or FALLBACK
    if chosen not in SOURCES:
        raise ValueError(f"TOUCHDOWN_IMAGERY_FILL={chosen!r} is not a touchdown.aerial source")
    return chosen


def pick(country: str = "", lat: float | None = None, lon: float | None = None) -> str:
    """The place's own source: by catalog country code; with no country given, the first national source whose area
    holds the point. A country with no source gets the global fill (neighbours still join the chain at the edges)."""
    fallback = fill()
    for s in SOURCES.values():
        if s.country and s.country == country:
            return s.id
    if not country and lat is not None and lon is not None:
        for s in SOURCES.values():
            if s.country and s.touches((lon, lat, lon, lat)):
                return s.id
    return fallback


def chain(primary: str, bbox: tuple[float, float, float, float], mpp: float = 0.0) -> list[Source]:
    """``primary``, then the other national sources touching ``bbox`` (those with a coarse backdrop abroad only when the
    square's ``mpp`` is coarse enough), then the global fill; sources with no key, or meant only for finer squares
    than this one, dropped."""
    ids = [primary] + [s.id for s in SOURCES.values() if s.country and s.id != primary and s.touches(bbox)
                       and mpp >= s.neighbour_min_mpp] + [fill()]
    out = []
    for i in dict.fromkeys(ids):
        s = SOURCES[i]
        if not s.available():
            if s.id == primary:
                print(f"  {s.id}: no {s.key_env} set, skipped", flush=True)
            continue
        if mpp > s.max_mpp:
            continue
        if s.touches(bbox):
            out.append(s)
    return out


# --- Web Mercator tiles -------------------------------------------------------------------------------------

def tile_url(source: Source, z: int, x: int, y: int, key: str = "") -> str:
    """``source``'s URL for tile (z, x, y); a WMS template gets the tile's EPSG:3857 bounds as {bbox}."""
    size = EARTH_M / 2 ** z
    west, north = -EARTH_M / 2 + x * size, EARTH_M / 2 - y * size
    bbox = f"{west:.3f},{north - size:.3f},{west + size:.3f},{north:.3f}"
    return source.template.format(z=z, x=x, y=y, key=key, bbox=bbox)


def _merc(lon, lat, z: int):
    """Global pixel coordinates at zoom ``z`` (arrays welcome)."""
    n = TILE * 2 ** z
    lat = np.clip(lat, -85.05112878, 85.05112878)
    x = (np.asarray(lon) + 180.0) / 360.0 * n
    y = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / math.pi) / 2 * n
    return x, y


def zoom_for(metres_per_px: float, lat: float, max_zoom: int) -> int:
    """The coarsest zoom whose pixel is no bigger than ``metres_per_px`` at ``lat``, capped at ``max_zoom``."""
    z = math.ceil(math.log2(EARTH_M * math.cos(math.radians(lat)) / TILE / metres_per_px))
    return int(min(max(z, 0), max_zoom))


def _get_tile(client, url: str) -> np.ndarray | None:
    """One tile as RGBA, or None where the server has nothing (404/204, not an image, transparent, or one flat black
    or pure white colour). Flat near-white is kept: blown-out glacier snow is 251-253 with std < 1 (Aiguille du Midi)."""
    for attempt in range(4):
        try:
            r = client.get(url)
            if r.status_code in (204, 404, 400):
                return None
            r.raise_for_status()
            if not r.headers.get("content-type", "").startswith("image/"):
                return None
            im = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGBA"))
            rgb = im[..., :3]
            if im[..., 3].max() == 0:
                return None
            if rgb.std(axis=(0, 1)).max() < 0.8 and not 8 <= rgb.mean() <= 254.5:  # a flat black or white "no data" fill
                return None
            return im
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    return None


def tiles_rgba(source: Source, bbox: tuple[float, float, float, float], pixels: int,
               get_tile: Callable[[str], np.ndarray | None] | None = None) -> np.ndarray:
    """``source``'s tiles over ``bbox`` (west, south, east, north) as plate carree RGBA, ``pixels`` rows, its width
    in the bbox's degree aspect (the contract of nadir_dive._naip). Alpha 0 where the server had no tile."""
    import cv2
    west, south, east, north = bbox
    height, width = pixels, round(pixels * (east - west) / (north - south))
    mid = (south + north) / 2
    mpp = (north - south) * 111_320 / height
    z = zoom_for(mpp, mid, source.max_zoom)
    x0, y0 = _merc(west, north, z)
    x1, y1 = _merc(east, south, z)
    tx0, ty0, tx1, ty1 = int(x0 // TILE), int(y0 // TILE), int(x1 // TILE), int(y1 // TILE)
    jobs = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    key = os.environ.get(source.key_env, "") if source.key_env else ""
    print(f"  {source.id}: zoom {z}, {len(jobs)} tiles", flush=True)
    mosaic = np.zeros(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE, 4), np.uint8)
    if get_tile is None:
        import httpx
        client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True)
        get = lambda url: _get_tile(client, url)  # noqa: E731
    else:
        client, get = None, get_tile
    try:
        with ThreadPoolExecutor(WORKERS) as pool:
            urls = [tile_url(source, z, tx, ty, key) for tx, ty in jobs]
            missing = []
            for (tx, ty), url, tile in zip(jobs, urls, pool.map(get, urls)):
                if tile is None:
                    missing.append(((tx, ty), url))
                else:
                    r, c = (ty - ty0) * TILE, (tx - tx0) * TILE
                    mosaic[r:r + TILE, c:c + TILE] = tile[:TILE, :TILE]
            if missing and len(missing) < len(jobs):  # servers drop the odd tile (IGN 404s intermittently): ask once more
                time.sleep(2)
                again = list(pool.map(get, [u for _, u in missing]))
                for ((tx, ty), _), tile in zip(missing, again):
                    if tile is not None:
                        r, c = (ty - ty0) * TILE, (tx - tx0) * TILE
                        mosaic[r:r + TILE, c:c + TILE] = tile[:TILE, :TILE]
                print(f"    {len(missing)} missing, {sum(t is not None for t in again)} recovered on retry", flush=True)
    finally:
        if client is not None:
            client.close()
    lons = west + (np.arange(width) + 0.5) / width * (east - west)
    lats = north - (np.arange(height) + 0.5) / height * (north - south)
    mx, _ = _merc(lons, np.full_like(lons, mid), z)
    _, my = _merc(np.zeros_like(lats), lats, z)
    map_x = np.broadcast_to((mx - tx0 * TILE - 0.5)[None, :], (height, width)).astype(np.float32)
    map_y = np.broadcast_to((my - ty0 * TILE - 0.5)[:, None], (height, width)).astype(np.float32)
    scale = (x1 - x0) / width  # mosaic px per output px; > 1 when the zoom cap did not bind tightly
    src = mosaic
    if scale > 1.5:  # pre-shrink so the bilinear remap does not alias
        f = 1 / scale
        src = cv2.resize(mosaic, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        map_x, map_y = (map_x + 0.5) * f - 0.5, (map_y + 0.5) * f - 0.5
    out = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out[..., 3] = np.where(out[..., 3] > 250, 255, 0)  # a tile edge or hole blends alpha: keep only solid pixels
    return out


def square_rgba(primary: str, bbox: tuple[float, float, float, float], pixels: int,
                get_tile: Callable[[str], np.ndarray | None] | None = None) -> tuple[np.ndarray, dict]:
    """``bbox`` from the chain starting at ``primary``; returns RGBA (alpha 0 = no source) and per-source cover."""
    out, cover = None, {}
    mpp = (bbox[3] - bbox[1]) * 111_320 / pixels
    sources = chain(primary, bbox, mpp)
    for k, s in enumerate(sources):
        try:
            part = s.fetch(bbox, pixels) if s.fetch else tiles_rgba(s, bbox, pixels, get_tile)
        except ImportError:  # a missing library is a setup error, not a server down: never fall back silently
            raise
        except Exception as exc:  # noqa: BLE001 - a server down: the next source in the chain fills, said loudly
            if k == len(sources) - 1:
                raise
            print(f"  {s.id} failed ({type(exc).__name__}: {str(exc)[:160]}); the next source fills", flush=True)
            cover[s.id] = 0.0
            continue
        if out is None:
            out = part.copy()
            took = part[..., 3] > 127
        else:
            hole = out[..., 3] <= 127
            took = hole & (part[..., 3] > 127)
            out[took] = part[took]
        cover[s.id] = round(float(took.mean()), 4)
        if (out[..., 3] > 127).all():
            break
    return out, cover


def credits(ids) -> list[dict]:
    return [{"source": SOURCES[i].name, "licence": SOURCES[i].licence, "credit": SOURCES[i].credit,
             "resolution_m": SOURCES[i].metres} for i in ids]
