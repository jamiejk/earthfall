"""Fine national DEMs outside the US for the touchdown terrain route, with Copernicus GLO-30 as the fallback.

The touchdown terrain stage wants the US shape of ``terrain_route.acquire``: a near DEM of about 1 m around the
landing (a 400 m radius patch) and a far DEM over 17 km. In the US those are the 3DEP 1 m project and the
1/3 arc-second seamless DEM. Elsewhere:

=================  =======  ===========================================================  ====================
source id          country  product and access                                           CRS / heights
=================  =======  ===========================================================  ====================
``swissalti3d``    CH       swisstopo swissALTI3D 0.5 m (or 2 m), 1 km COG tiles found     EPSG:2056 / LN02
                            through the data.geo.admin.ch STAC API                         (EPSG:5728)
``ign-rgealti``    FR       IGN RGE ALTI 1 m, Géoplateforme raw-elevation WMS (wms-r)      EPSG:2154 /
                                                                                           NGF-IGN69 (5720)
``ign-lidarhd``    FR       IGN LiDAR HD MNT 0.5 m, same WMS; still being released in      EPSG:2154 /
                            batches, so RGE ALTI fills where it has no data yet            NGF-IGN69 (5720)
``es-mdt05``       ES       IGN/CNIG MDT05 (5 m, from PNOA LiDAR), IDEE WCS 2.0            EPSG:25830 /
                                                                                           REDNAP (5782)
``tirol-dtm``      AT       Land Tirol Geländemodell 0.5 m (lidar), WCS 2.0; only Tirol    EPSG:31254/31255 /
                            (the rest of Austria falls back to GLO-30)                     GHA (5778)
``kartverket-dtm`` NO       Kartverket DTM 1 m (national height model), geonorge WCS       EPSG:25833 /
                                                                                           NN2000 (5941)
``linz-dem1m``     NZ       LINZ national 1 m lidar DEM, open AWS bucket nz-elevation      EPSG:2193 /
                            (COG per 1:50k sheet); the contour 8 m DEM fills where no      NZVD2016 (7839)
                            lidar was flown
``gsi-dem5``       JP       GSI DEM5A/5B/5C (5 m) png tiles, DEM10B where 5 m is missing   reprojected to UTM /
                                                                                           JGD2011 (6695)
``glo30``          any      Copernicus DEM GLO-30 (providers.CopernicusDEMProvider)        reprojected to UTM /
                                                                                           EGM2008 (3855)
=================  =======  ===========================================================  ====================

Every provider has the ``acquire(spec, raw_cache) -> (GeoTIFF path, SourceRecord)`` shape of providers.py and
writes a GeoTIFF in a metric projected CRS with orthometric heights (tags ``vertical_crs`` and ``vertical_datum``).
GLO-30 is the far DEM everywhere and the near DEM wherever no finer source exists or the finer one fails; a GLO-30
near patch is tagged ``dem_fallback_1m_share`` = 0 so the build keeps its coarse-DEM camera clearance, as it does
for a US place without lidar.

Heights of different national datums and EGM2008 differ by up to a metre or two, and GLO-30 is a surface model
(forest canopy and buildings in the heights) where the national DEMs are bare earth; the near and far meshes are
not shifted to agree.

The country comes from the place catalogue; a custom point with no country is placed by the coarse polygons in
``AREAS`` (a point a polygon wrongly claims just falls back to GLO-30 when the national source has no data there).
"""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .utm import projected_crs, wgs84_utm_code

USER_AGENT = "GeoBlock/0.1 (https://github.com/jamiejk/geoblock-scenecompiler)"
NEAR_RADIUS_M = 400
FAR_RADIUS_M = 17000

COUNTRY_SOURCE = {"CH": "swissalti3d", "FR": "ign-lidarhd", "NO": "kartverket-dtm", "NZ": "linz-dem1m",
                  "JP": "gsi-dem5", "ES": "es-mdt05", "AT": "tirol-dtm"}
FALLBACK = "glo30"

# Coarse outlines (lon, lat) for custom points with no catalogue country. Checked in this order, so the Swiss outline
# wins over the French one around Geneva; neither is a border survey.
AREAS: dict[str, list[tuple[tuple[float, float], ...]]] = {
    "CH": [((6.02, 46.14), (7.04, 45.92), (7.66, 45.97), (7.86, 45.92), (8.40, 46.25), (8.92, 45.83),
            (9.05, 45.82), (9.27, 46.30), (10.10, 46.23), (10.47, 46.55), (10.49, 46.94), (9.60, 47.06),
            (9.53, 47.27), (9.67, 47.55), (8.55, 47.80), (7.60, 47.60), (6.85, 47.30), (6.10, 46.60),
            (5.96, 46.20))],
    "FR": [((-1.80, 43.40), (-4.80, 48.30), (-1.90, 49.70), (1.50, 51.10), (2.50, 51.10), (4.20, 49.90),
            (8.20, 49.00), (7.60, 47.60), (6.85, 47.30), (6.10, 46.60), (6.02, 46.14), (7.04, 45.92),
            (6.60, 45.10), (7.70, 44.10), (7.50, 43.80), (3.20, 42.40)),
           ((8.50, 41.30), (9.60, 41.30), (9.60, 43.05), (8.50, 43.05))],
    "NO": [((4.50, 58.00), (4.60, 62.50), (10.00, 64.50), (14.00, 67.50), (17.50, 69.60), (25.00, 71.20),
            (31.20, 70.30), (28.90, 69.00), (20.50, 69.00), (16.00, 67.50), (12.00, 64.00), (12.30, 61.00),
            (11.00, 59.00), (7.00, 57.90))],
    "AT": [((9.53, 47.27), (9.60, 47.06), (10.47, 46.87), (12.13, 46.94), (12.40, 46.65), (13.70, 46.52),
            (14.60, 46.40), (16.00, 46.70), (16.60, 47.70), (17.10, 48.00), (16.90, 48.70), (15.00, 49.00),
            (13.80, 48.75), (13.00, 47.80), (12.20, 47.60), (10.50, 47.55), (9.60, 47.55))],
    "ES": [((-9.40, 43.80), (-1.80, 43.40), (3.20, 42.40), (3.30, 41.80), (0.20, 39.70), (-0.80, 37.50),
            (-2.10, 36.70), (-5.60, 36.00), (-7.40, 37.20), (-7.00, 38.20), (-7.50, 39.70), (-6.90, 41.90),
            (-8.90, 41.90)),
           ((-18.20, 27.60), (-13.40, 27.60), (-13.40, 29.50), (-18.20, 29.50))],
    "NZ": [((166.00, -47.50), (178.70, -47.50), (178.70, -34.00), (166.00, -34.00))],
    "JP": [((122.90, 24.00), (132.00, 24.00), (142.00, 33.50), (146.00, 43.00), (145.80, 45.60), (141.50, 45.60),
            (139.30, 42.00), (135.50, 38.00), (131.00, 35.60), (129.60, 34.90), (128.90, 33.00))],
}


def _inside(lon: float, lat: float, ring) -> bool:
    hit = False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        if (y0 > lat) != (y1 > lat) and lon < x0 + (lat - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def country_at(lat: float, lon: float) -> str:
    """The country of ``AREAS`` holding the point, or "" (then only GLO-30 applies)."""
    for code, rings in AREAS.items():
        if any(_inside(lon, lat, list(ring)) for ring in rings):
            return code
    return ""


@dataclass(frozen=True)
class TerrainPlan:
    country: str
    near: str            # source id for the 400 m landing patch
    far: str             # source id for the 17 km DEM
    zone: int            # WGS 84 / UTM EPSG code; utm.projected_crs(zone) is the build's CRS
    near_resolution_m: float

    @property
    def projected_crs(self) -> str:
        return projected_crs(self.zone)


def plan(lat: float, lon: float, country: str | None = None) -> TerrainPlan:
    """Sources for a place outside the US: by the catalogue ``country``, else by ``AREAS``."""
    code = country if country else country_at(lat, lon)
    near = COUNTRY_SOURCE.get(code, FALLBACK)
    return TerrainPlan(code, near, FALLBACK, wgs84_utm_code(lat, lon), PROVIDERS[near].nominal_resolution_m)


# --- shared plumbing ----------------------------------------------------------------------------------------

def _client():
    import httpx
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=90, follow_redirects=True)


def _native_bounds(spec, crs: str, margin_m: float = 60) -> tuple[float, float, float, float]:
    from rasterio.warp import transform_bounds
    x0, y0, x1, y1 = spec.bounds
    return transform_bounds(spec.projected_crs, crs, x0 - margin_m, y0 - margin_m, x1 + margin_m, y1 + margin_m,
                            densify_pts=21)


def _cache_dir(raw_cache: Path, name: str, request: dict) -> Path:
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    directory = raw_cache / name / key
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write(raster: Path, data: np.ndarray, crs: str, transform, nodata: float, **tags) -> None:
    import rasterio
    data = np.asarray(data, dtype="float32")
    if data.ndim == 2:
        data = data[None]
    profile = {"driver": "GTiff", "height": data.shape[1], "width": data.shape[2], "count": 1, "dtype": "float32",
               "crs": crs, "transform": transform, "nodata": nodata, "compress": "deflate", "predictor": 3,
               "tiled": True}
    temp = raster.with_suffix(".part.tif")
    with rasterio.open(temp, "w", **profile) as dst:
        dst.write(data)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    os.replace(temp, raster)


def _locked(directory: Path, make: Callable[[Path], None]) -> Path:
    raster = directory / "source.tif"
    with (directory / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not raster.is_file():
            make(raster)
    return raster


def _record(raster: Path, **kw):
    import rasterio
    from .models import SourceRecord
    from .providers import sha256, utc_now
    with rasterio.open(raster) as src:
        tags = src.tags()
        return SourceRecord(horizontal_crs=src.crs.to_string(), vertical_crs=tags.get("vertical_crs"),
                            vertical_reference_verified=True,
                            native_pixel_spacing=tuple(abs(float(x)) for x in src.res),
                            native_pixel_spacing_units="metres", retrieval_utc=utc_now(), sha256=sha256(raster), **kw)


def _merge_cogs(uris: list[str], bounds, nodata: float, res: float | None = None):
    """Window-read and merge COGs (vsicurl) over ``bounds`` in their own CRS; returns (data, transform, crs)."""
    import rasterio
    from rasterio.merge import merge
    from .providers import grid_aligned
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff"):
        sources = []
        try:
            for uri in uris:
                try:
                    sources.append(rasterio.open(uri))
                except rasterio.errors.RasterioIOError:
                    continue
            if not sources:
                raise ValueError(f"none of {len(uris)} tiles could be opened (first: {uris[:1]})")
            grid = res or abs(sources[0].res[0])
            aligned = grid_aligned(bounds, (sources[0].bounds.left, sources[0].bounds.top), grid)
            src_nodata = sources[0].nodata if sources[0].nodata is not None else nodata
            data, transform = merge(sources, bounds=aligned, nodata=src_nodata, res=(grid, grid) if res else None)
            crs = sources[0].crs.to_string()
        finally:
            for s in sources:
                s.close()
    data = data.astype("float32")
    data[~np.isfinite(data) | (data == src_nodata)] = nodata
    return data, transform, crs


def _require_cover(data: np.ndarray, nodata: float, what: str, minimum: float = 0.98) -> None:
    share = float((data != nodata).mean())
    if share < minimum:
        raise ValueError(f"{what} covers only {100 * share:.0f}% of the landing area")


def _warp_to(source: Path, crs: str, bounds, res: float, raster: Path, **tags) -> None:
    """Reproject a DEM onto a ``res`` grid in ``crs`` over ``bounds`` (bilinear)."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    west, south, east, north = bounds
    width, height = math.ceil((east - west) / res), math.ceil((north - south) / res)
    transform = from_origin(west, north, res, res)
    out = np.full((height, width), np.nan, np.float32)
    with rasterio.open(source) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
        keep = {k: v for k, v in src.tags().items() if k in ("vertical_crs", "vertical_datum", "source_tiles")}
    out[~np.isfinite(out)] = -9999.0
    _write(raster, out, crs, transform, -9999.0, **{**keep, **tags})


# --- providers ----------------------------------------------------------------------------------------------

class SwissALTI3DProvider:
    """swisstopo swissALTI3D (bare earth, 0.5 m or 2 m) from its 1 km COG tiles. Tile file names carry the survey
    year, so the STAC API lists them for the bbox; the newest year of each tile is kept."""

    id = "swissalti3d"
    nominal_resolution_m = 0.5
    stac = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
    crs, vertical, datum = "EPSG:2056", "EPSG:5728", "LN02"

    def __init__(self, resolution: float = 0.5, client=None):
        if resolution not in (0.5, 2.0):
            raise ValueError("swissALTI3D comes at 0.5 m or 2 m")
        self.resolution = resolution
        self.nominal_resolution_m = resolution
        self.client = client

    def asset_urls(self, bbox_wgs84: tuple[float, float, float, float]) -> list[str]:
        """The newest tile per 1 km square over ``bbox_wgs84`` at this resolution (follows STAC paging)."""
        client = self.client or _client()
        newest: dict[str, tuple[str, str]] = {}
        url, params = self.stac, {"bbox": ",".join(f"{v:.6f}" for v in bbox_wgs84), "limit": 100}
        res = "0.5" if self.resolution == 0.5 else "2"
        try:
            while url:
                r = client.get(url, params=params)
                r.raise_for_status()
                page = r.json()
                for item in page.get("features", []):
                    _, year, square = item["id"].split("_")  # swissalti3d_2019_2617-1092
                    for key, asset in item.get("assets", {}).items():
                        if key.endswith(f"_{res}_2056_5728.tif") and (square not in newest or year > newest[square][0]):
                            newest[square] = (year, asset["href"])
                url = next((link["href"] for link in page.get("links", []) if link.get("rel") == "next"), None)
                params = None
        finally:
            if self.client is None:
                client.close()
        return [href for _, href in sorted(newest.values(), key=lambda v: v[1])]

    def acquire(self, spec, raw_cache: Path):
        from .coordinates import geographic_envelope
        bounds = _native_bounds(spec, self.crs)
        directory = _cache_dir(raw_cache, "swissalti3d", {"bounds": [round(v, 1) for v in bounds],
                                                          "res": self.resolution})

        def make(raster):
            urls = self.asset_urls(geographic_envelope(spec, margin_m=60))
            if not urls:
                raise ValueError("swissALTI3D has no tile here (outside Switzerland)")
            data, transform, crs = _merge_cogs(["/vsicurl/" + u for u in urls], bounds, -9999.0)
            _require_cover(data, -9999.0, "swissALTI3D")
            _write(raster, data, crs, transform, -9999.0, vertical_crs=self.vertical, vertical_datum=self.datum,
                   source_tiles=" ".join(u.rsplit("/", 1)[-1] for u in urls))

        raster = _locked(directory, make)
        return raster, _record(raster, provider="swisstopo", dataset=f"swissALTI3D {self.resolution:g} m",
                               nominal_resolution_m=self.resolution,
                               license="swisstopo open data (free use, attribution required)",
                               attribution="© swisstopo, swissALTI3D", source_uri=self.stac)


def _coverage_bytes(kind: str, content: bytes) -> bytes | None:
    """The image part of a WCS 2.0 ``multipart/related`` answer (GML description + GeoTIFF; ArcGIS's WCSServer sends
    this even for FORMAT=image/tiff, Tirol 18 Oct), or None when the answer has no image part."""
    import email
    msg = email.message_from_bytes(f"Content-Type: {kind}\r\n\r\n".encode() + content)
    for part in msg.walk():
        if not part.is_multipart() and part.get_content_type().startswith("image/"):
            return part.get_payload(decode=True)
    return None


class _GridServiceProvider:
    """A DEM served as a float GeoTIFF by an OGC GetMap / GetCoverage request in a national CRS, in chunks of at
    most ``max_px`` a side. The response's own georeferencing is not trusted: the grid is the requested one."""

    id = ""
    crs = vertical = datum = ""
    nominal_resolution_m = 1.0
    max_px = 2000
    nodata_values: tuple[float, ...] = (-99999.0, -9999.0, -32767.0, -3.4028234663852886e38)

    def __init__(self, client=None):
        self.client = client

    def url(self, bounds: tuple[float, float, float, float], width: int, height: int) -> str:
        raise NotImplementedError

    def chunks(self, bounds) -> list[tuple[tuple[float, float, float, float], int, int, int, int]]:
        """((west, south, east, north), width, height, row, col) requests tiling ``bounds`` on the 1 m grid."""
        res = self.nominal_resolution_m
        west, south, east, north = bounds
        west, south = math.floor(west / res) * res, math.floor(south / res) * res
        east, north = math.ceil(east / res) * res, math.ceil(north / res) * res
        width, height = round((east - west) / res), round((north - south) / res)
        out = []
        for row in range(0, height, self.max_px):
            for col in range(0, width, self.max_px):
                w, h = min(self.max_px, width - col), min(self.max_px, height - row)
                left, top = west + col * res, north - row * res
                out.append(((left, top - h * res, left + w * res, top), w, h, row, col))
        return out

    def acquire(self, spec, raw_cache: Path):
        import rasterio
        from rasterio.transform import from_origin
        bounds = _native_bounds(spec, self.crs)
        directory = _cache_dir(raw_cache, self.id, {"bounds": [round(v, 1) for v in bounds]})

        def make(raster):
            jobs = self.chunks(bounds)
            res = self.nominal_resolution_m
            height = max(r + h for _, _, h, r, _ in jobs)
            width = max(c + w for _, w, _, _, c in jobs)
            data = np.full((height, width), -9999.0, np.float32)
            client = self.client or _client()
            try:
                for box, w, h, row, col in jobs:
                    r = client.get(self.url(box, w, h))
                    r.raise_for_status()
                    kind = r.headers.get("content-type", "")
                    body = r.content
                    if kind.startswith("multipart/related"):
                        body = _coverage_bytes(kind, r.content)
                        if body is None:
                            raise ValueError(f"{self.id}: multipart answer with no image part: {r.text[:300]}")
                    elif "xml" in kind or "html" in kind or "text" in kind:
                        raise ValueError(f"{self.id}: service answered {kind}: {r.text[:300]}")
                    with rasterio.MemoryFile(io.BytesIO(body)) as mem, mem.open() as src:
                        nodata = src.nodata
                        # a WCS 2.0 answer is on the service's native grid, which can be a pixel off the request
                        part = src.read(1, out_shape=(h, w)).astype("float32") if src.shape != (h, w) else \
                            src.read(1).astype("float32")
                        if nodata is not None:
                            part[part == nodata] = -9999.0
                    part[~np.isfinite(part) | np.isin(part, self.nodata_values) | (part < -1000)] = -9999.0
                    data[row:row + h, col:col + w] = part[:h, :w]
            finally:
                if self.client is None:
                    client.close()
            _require_cover(data, -9999.0, self.id)
            west, north = jobs[0][0][0], jobs[0][0][3]
            _write(raster, data, self.crs, from_origin(west, north, res, res), -9999.0,
                   vertical_crs=self.vertical, vertical_datum=self.datum)

        raster = _locked(directory, make)
        return raster, _record(raster, **self.record_fields())

    def record_fields(self) -> dict:
        raise NotImplementedError


class IGNRGEAltiProvider(_GridServiceProvider):
    """IGN RGE ALTI 1 m (bare earth; 5 m where no lidar or correlation survey exists) from the Géoplateforme
    raw-elevation WMS, as float GeoTIFF in Lambert-93. Heights NGF-IGN69 (mainland; Corsica is IGN78)."""

    id = "ign-rgealti"
    crs, vertical, datum = "EPSG:2154", "EPSG:5720", "NGF-IGN69"
    service = "https://data.geopf.fr/wms-r/wms"
    layer = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"

    def url(self, bounds, width, height):
        w, s, e, n = bounds
        return (f"{self.service}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={self.layer}&STYLES="
                f"&CRS={self.crs}&BBOX={w:.2f},{s:.2f},{e:.2f},{n:.2f}&WIDTH={width}&HEIGHT={height}"
                "&FORMAT=image/geotiff")

    def record_fields(self):
        return dict(provider="IGN", dataset="IGN RGE ALTI 1 m (Géoplateforme)", nominal_resolution_m=1.0,
                    license="Licence Ouverte / Etalab 2.0", attribution="IGN, RGE ALTI", source_uri=self.service)


class IGNLidarHDProvider(IGNRGEAltiProvider):
    """IGN LiDAR HD MNT 0.5 m (bare earth from the national lidar programme) from the same raw-elevation WMS. RGE ALTI
    in the steep Alps is photo-derived in places (smooth, hatched cliff faces at the Aiguille du Midi); LiDAR HD is
    real lidar. Its MNT is still being released in batches: where it has no data the patch comes from RGE ALTI."""

    id = "ign-lidarhd"
    nominal_resolution_m = 0.5
    layer = "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"

    def acquire(self, spec, raw_cache: Path):
        try:
            return super().acquire(spec, raw_cache)
        except Exception as exc:  # noqa: BLE001 - not released here yet (or the layer failed): RGE ALTI, loudly
            print(f"terrain: ign-lidarhd failed ({type(exc).__name__}: {str(exc)[:160]}); near DEM from RGE ALTI",
                  flush=True)
            return IGNRGEAltiProvider(self.client).acquire(spec, raw_cache)

    def record_fields(self):
        return dict(provider="IGN", dataset="IGN LiDAR HD MNT 0.5 m (Géoplateforme)", nominal_resolution_m=0.5,
                    license="Licence Ouverte / Etalab 2.0", attribution="IGN, LiDAR HD", source_uri=self.service)


class _WCS20Provider(_GridServiceProvider):
    """A DEM coverage behind an OGC WCS 2.0 GetCoverage (subset by the native CRS's axes)."""

    service = coverage = ""
    axes = ("x", "y")

    def url(self, bounds, width, height):
        w, s, e, n = bounds
        x, y = self.axes
        return (f"{self.service}?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID={self.coverage}"
                f"&SUBSET={x}({w:.2f},{e:.2f})&SUBSET={y}({s:.2f},{n:.2f})&FORMAT=image/tiff")


class SpainMDTProvider(_WCS20Provider):
    """IGN/CNIG MDT05 (5 m bare earth from PNOA LiDAR) through the IDEE INSPIRE WCS in ETRS89 / UTM 30N. Heights
    REDNAP (Alicante; Canaries: their own tide gauges)."""

    id = "es-mdt05"
    crs, vertical, datum = "EPSG:25830", "EPSG:5782", "REDNAP"
    nominal_resolution_m = 5.0
    service = "https://servicios.idee.es/wcs-inspire/mdt"
    coverage = "Elevacion25830_5"

    def record_fields(self):
        return dict(provider="IGN España", dataset="MDT05 (CNIG, from PNOA LiDAR)", nominal_resolution_m=5.0,
                    license="CC BY 4.0", attribution="MDT05 cedido por © Instituto Geográfico Nacional",
                    source_uri=self.service)


class _TirolProvider(_WCS20Provider):
    vertical, datum = "EPSG:5778", "GHA (Gebrauchshöhen Adria)"
    nominal_resolution_m = 0.5
    service = "https://gis.tirol.gv.at/arcgis/services/Service_Public/terrain/MapServer/WCSServer"

    def record_fields(self):
        return dict(provider="Land Tirol", dataset="Geländemodell 0.5 m (Laserscanning)", nominal_resolution_m=0.5,
                    license="CC BY 4.0", attribution="Land Tirol - data.tirol.gv.at", source_uri=self.service)


class TirolDTMProvider:
    """Land Tirol's 0.5 m lidar terrain model through its WCS 2.0, published in two Gauss-Krüger zones: M28
    (EPSG:31254) west of about 11.76°E, M31 (EPSG:31255) east of it (East Tirol included). Only Tirol: elsewhere in
    Austria the coverage is empty and the near patch falls back to GLO-30."""

    id = "tirol-dtm"
    nominal_resolution_m = 0.5
    zones = {"M28": "EPSG:31254", "M31": "EPSG:31255"}

    def __init__(self, client=None):
        self.client = client

    def zone_provider(self, lon: float) -> _TirolProvider:
        name = "M28" if lon < 11.76 else "M31"
        p = _TirolProvider(self.client)
        p.id, p.crs, p.coverage = f"tirol-dtm-{name.lower()}", self.zones[name], f"Gelaendemodell_50cm_{name}"
        return p

    def acquire(self, spec, raw_cache: Path):
        return self.zone_provider(spec.longitude).acquire(spec, raw_cache)


class KartverketDTMProvider(_GridServiceProvider):
    """Kartverket's national terrain model DTM 1 m (bare earth, from the national lidar programme) through the
    geonorge WCS in UTM 33 (EUREF89). Heights NN2000."""

    id = "kartverket-dtm"
    crs, vertical, datum = "EPSG:25833", "EPSG:5941", "NN2000"
    service = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833"
    coverage = "nhm_dtm_topo_25833"

    def url(self, bounds, width, height):
        w, s, e, n = bounds
        return (f"{self.service}?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage&COVERAGE={self.coverage}"
                f"&CRS={self.crs}&RESPONSE_CRS={self.crs}&BBOX={w:.2f},{s:.2f},{e:.2f},{n:.2f}"
                f"&WIDTH={width}&HEIGHT={height}&FORMAT=GeoTIFF")

    def record_fields(self):
        return dict(provider="Kartverket", dataset="Kartverket DTM 1 m (Nasjonal høydemodell)",
                    nominal_resolution_m=1.0, license="CC BY 4.0", attribution="© Kartverket", source_uri=self.service)


# LINZ: the national 1 m lidar DEM is published per NZTopo50 map sheet (24 x 36 km), named like CB08.
NZ_SHEET_WIDTH, NZ_SHEET_HEIGHT = 24_000, 36_000
NZ_SHEET_LEFT, NZ_SHEET_TOP = 988_000, 6_234_000  # NZTM corner of the sheet grid; row 0 is "AS"
NZ_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"           # no I or O
NZ_FIRST_ROW = NZ_LETTERS.index("S")               # "AS" = the 17th two-letter row code


def nz_sheet(easting: float, northing: float) -> str:
    """The NZTopo50 sheet code holding an NZTM point, e.g. CB08 for Milford Sound."""
    col = math.floor((easting - NZ_SHEET_LEFT) / NZ_SHEET_WIDTH)
    row = math.floor((NZ_SHEET_TOP - northing) / NZ_SHEET_HEIGHT) + NZ_FIRST_ROW
    if not 0 <= col < 100 or not 0 <= row < len(NZ_LETTERS) ** 2:
        raise ValueError("outside the NZTopo50 sheet grid")
    return f"{NZ_LETTERS[row // len(NZ_LETTERS)]}{NZ_LETTERS[row % len(NZ_LETTERS)]}{col:02d}"


def nz_tile(easting: float, northing: float, scale: int) -> str:
    """The LINZ tile name for an NZTM point at a tile-index ``scale`` (1000, 5000, 10000 ...): sheet, scale, then
    the 1-based row and column of the tile inside the sheet (CB08_5000_0107)."""
    per_side = 50_000 // scale
    sheet = nz_sheet(easting, northing)
    col = math.floor((easting - NZ_SHEET_LEFT) % NZ_SHEET_WIDTH / (NZ_SHEET_WIDTH / per_side)) + 1
    row = math.floor((NZ_SHEET_TOP - northing) % NZ_SHEET_HEIGHT / (NZ_SHEET_HEIGHT / per_side)) + 1
    digits = 3 if per_side >= 100 else 2
    return f"{sheet}_{scale}_{row:0{digits}d}{col:0{digits}d}"


def nz_sheets(bounds: tuple[float, float, float, float]) -> list[str]:
    west, south, east, north = bounds
    out = []
    for y in np.arange(north, south - NZ_SHEET_HEIGHT, -NZ_SHEET_HEIGHT):
        for x in np.arange(west, east + NZ_SHEET_WIDTH, NZ_SHEET_WIDTH):
            px, py = min(x, east), max(y, south)
            name = nz_sheet(px, py)
            if name not in out:
                out.append(name)
    return out


class LINZDEMProvider:
    """LINZ 1 m lidar DEM (bare earth) from the open nz-elevation bucket, one COG per NZTopo50 sheet; holes (no
    lidar) are filled from the contour-derived 8 m DEM of the same bucket. No key. Heights NZVD2016."""

    id = "linz-dem1m"
    nominal_resolution_m = 1.0
    bucket = "https://nz-elevation.s3.ap-southeast-2.amazonaws.com"
    lidar = "new-zealand/new-zealand/dem_1m/2193"
    contour = "new-zealand/new-zealand-contour/dem_8m/2193"
    crs, vertical, datum = "EPSG:2193", "EPSG:7839", "NZVD2016"

    def __init__(self, bucket: str | None = None):
        self.bucket = (bucket or os.environ.get("GEOBLOCK_NZ_ELEVATION_ROOT") or self.bucket).rstrip("/")

    def urls(self, bounds, collection: str) -> list[str]:
        return [f"{self.bucket}/{collection}/{sheet}.tiff" for sheet in nz_sheets(bounds)]

    def acquire(self, spec, raw_cache: Path):
        bounds = _native_bounds(spec, self.crs)
        directory = _cache_dir(raw_cache, "linz_dem", {"bounds": [round(v, 1) for v in bounds], "root": self.bucket})

        def make(raster):
            data, transform, crs = _merge_cogs(["/vsicurl/" + u for u in self.urls(bounds, self.lidar)], bounds,
                                               -9999.0, res=1.0)
            hole = data == -9999.0
            filled = float(hole.mean())
            if filled:
                coarse, _, _ = _merge_cogs(["/vsicurl/" + u for u in self.urls(bounds, self.contour)], bounds,
                                           -9999.0, res=1.0)
                data[hole] = coarse[hole]
            _require_cover(data, -9999.0, "LINZ DEM")
            _write(raster, data, crs, transform, -9999.0, vertical_crs=self.vertical, vertical_datum=self.datum,
                   contour_filled_share=f"{filled:.3f}", source_tiles=" ".join(nz_sheets(bounds)))

        raster = _locked(directory, make)
        import rasterio
        with rasterio.open(raster) as src:
            filled = float(src.tags().get("contour_filled_share", 0))
        return raster, _record(raster, provider="LINZ",
                               dataset="LINZ New Zealand LiDAR 1 m DEM" + (
                                   f" ({100 * filled:.0f}% from the contour 8 m DEM)" if filled else ""),
                               nominal_resolution_m=8 if filled > 0.5 else 1, license="CC BY 4.0",
                               attribution="Sourced from the LINZ Data Service and licensed for reuse under CC BY 4.0",
                               source_uri=f"{self.bucket}/{self.lidar}/")


def gsi_decode(rgb: np.ndarray) -> np.ndarray:
    """GSI elevation PNG to metres: h = R*65536 + G*256 + B in 0.01 m, two's complement at 2^23; 2^23 itself is no
    data (NaN)."""
    rgb = rgb.astype(np.int64)
    h = rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]
    out = np.where(h < 2 ** 23, h, h - 2 ** 24).astype(np.float64) * 0.01
    out[h == 2 ** 23] = np.nan
    return out.astype(np.float32)


class GSIDEMProvider:
    """GSI (Geospatial Information Authority of Japan) elevation tiles: DEM5A (lidar), DEM5B and DEM5C (5 m) at
    zoom 15, then DEM10B (10 m) at zoom 14 for what none of them covers. Heights JGD2011 (Tokyo Bay mean sea level).
    The Web Mercator mosaic is reprojected onto a 2 m grid in the build's UTM zone."""

    id = "gsi-dem5"
    nominal_resolution_m = 5.0
    root = "https://cyberjapandata.gsi.go.jp/xyz"
    layers = (("dem5a_png", 15), ("dem5b_png", 15), ("dem5c_png", 15), ("dem_png", 14))
    vertical, datum = "EPSG:6695", "JGD2011"

    def __init__(self, client=None):
        self.client = client

    @staticmethod
    def tiles(bbox_wgs84, z: int) -> list[tuple[int, int]]:
        west, south, east, north = bbox_wgs84
        n = 2 ** z

        def tx(lon):
            return int((lon + 180) / 360 * n)

        def ty(lat):
            r = math.radians(lat)
            return int((1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * n)

        return [(x, y) for y in range(ty(north), ty(south) + 1) for x in range(tx(west), tx(east) + 1)]

    def url(self, layer: str, z: int, x: int, y: int) -> str:
        return f"{self.root}/{layer}/{z}/{x}/{y}.png"

    def mosaic(self, bbox_wgs84, layer: str, z: int, client):
        """(heights, Web Mercator transform) of one layer over the bbox; NaN where the layer has no tile."""
        from PIL import Image
        from rasterio.transform import from_origin
        tiles = self.tiles(bbox_wgs84, z)
        xs, ys = sorted({t[0] for t in tiles}), sorted({t[1] for t in tiles})
        out = np.full((len(ys) * 256, len(xs) * 256), np.nan, np.float32)
        for x, y in tiles:
            r = client.get(self.url(layer, z, x, y))
            if r.status_code == 404:
                continue
            r.raise_for_status()
            rgb = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
            out[(y - ys[0]) * 256:(y - ys[0] + 1) * 256, (x - xs[0]) * 256:(x - xs[0] + 1) * 256] = gsi_decode(rgb)
        size = 2 * math.pi * 6378137.0 / 2 ** z / 256
        left = xs[0] * 256 * size - math.pi * 6378137.0
        top = math.pi * 6378137.0 - ys[0] * 256 * size
        return out, from_origin(left, top, size, size)

    def acquire(self, spec, raw_cache: Path):
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling, reproject
        from .coordinates import geographic_envelope
        bounds = spec.bounds
        directory = _cache_dir(raw_cache, "gsi_dem", {"bounds": [round(v, 1) for v in bounds],
                                                      "crs": spec.projected_crs})

        def make(raster):
            bbox = geographic_envelope(spec, margin_m=60)
            res = 2.0
            west, south, east, north = bounds[0] - 60, bounds[1] - 60, bounds[2] + 60, bounds[3] + 60
            transform = from_origin(west, north, res, res)
            out = np.full((math.ceil((north - south) / res), math.ceil((east - west) / res)), np.nan, np.float32)
            used = []
            client = self.client or _client()
            try:
                for layer, z in self.layers:
                    if np.isfinite(out).all():
                        break
                    heights, merc = self.mosaic(bbox, layer, z, client)
                    if not np.isfinite(heights).any():
                        continue
                    part = np.full_like(out, np.nan)
                    reproject(heights, part, src_transform=merc, src_crs="EPSG:3857", src_nodata=np.nan,
                              dst_transform=transform, dst_crs=spec.projected_crs, dst_nodata=np.nan,
                              resampling=Resampling.bilinear)
                    take = ~np.isfinite(out) & np.isfinite(part)
                    if take.any():
                        used.append(f"{layer}:{take.mean():.3f}")
                    out[take] = part[take]
            finally:
                if self.client is None:
                    client.close()
            out[~np.isfinite(out)] = -9999.0
            _require_cover(out, -9999.0, "GSI DEM")
            _write(raster, out, spec.projected_crs, transform, -9999.0, vertical_crs=self.vertical,
                   vertical_datum=self.datum, layers=" ".join(used))

        raster = _locked(directory, make)
        with rasterio.open(raster) as src:
            layers = src.tags().get("layers", "")
        shares = dict(part.split(":") for part in layers.split())
        coarse = float(shares.get("dem_png", 0)) > 0.5  # mostly DEM10B
        return raster, _record(raster, provider="GSI", dataset=f"GSI elevation tiles ({layers})",
                               nominal_resolution_m=10 if coarse else 5,
                               license="GSI website terms (attribution; compatible with CC BY 4.0)",
                               attribution="国土地理院 (GSI Japan) 標高タイル", source_uri=self.root)


class GLO30Provider:
    """Copernicus DEM GLO-30 (providers.CopernicusDEMProvider) reprojected onto the build's UTM grid: 10 m for the
    near patch (still 30 m data), 25 m for the far DEM."""

    id = "glo30"
    nominal_resolution_m = 30.0

    def __init__(self, grid_m: float | None = None):
        self.grid_m = grid_m

    def acquire(self, spec, raw_cache: Path):
        from .providers import CopernicusDEMProvider
        source, record = CopernicusDEMProvider().acquire(spec, raw_cache)
        grid = self.grid_m or (10.0 if spec.radius_m <= 2000 else 25.0)
        x0, y0, x1, y1 = spec.bounds
        margin = 3 * grid
        bounds = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
        directory = _cache_dir(raw_cache, "glo30_utm", {"src": record.sha256, "crs": spec.projected_crs,
                                                        "bounds": [round(v, 1) for v in bounds], "grid": grid})
        raster = _locked(directory, lambda out: _warp_to(source, spec.projected_crs, bounds, grid, out))
        return raster, record.model_copy(update={"horizontal_crs": spec.projected_crs, "sha256": None})


PROVIDERS: dict[str, object] = {p.id: p for p in (SwissALTI3DProvider(), IGNRGEAltiProvider(), IGNLidarHDProvider(),
                                                   SpainMDTProvider(), TirolDTMProvider(), KartverketDTMProvider(), LINZDEMProvider(), GSIDEMProvider(),
                                                   GLO30Provider())}


def provider(source: str):
    if source not in PROVIDERS:
        raise ValueError(f"unknown DEM source {source!r}; one of {', '.join(PROVIDERS)}")
    return PROVIDERS[source]


def spec_for(lat: float, lon: float, zone: int, radius_m: float):
    from pyproj import Transformer
    from .models import GeoSceneSpec
    crs = projected_crs(zone)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    return GeoSceneSpec(project_id="touchdown-terrain", name="Touchdown terrain", longitude=lon, latitude=lat,
                        projected_crs=crs, vertical_crs="EPSG:3855", radius_m=radius_m,
                        origin_easting_m=x, origin_northing_m=y, location_source="place catalogue")


def _tag_fallback(near: Path, reason: str) -> None:
    import rasterio
    with rasterio.open(near, "r+") as dst:
        dst.update_tags(dem_fallback_1m_share="0.000", dem_fallback_reason=reason[:300])


def acquire(lat: float, lon: float, near_source: str, zone: int, cache: Path) -> tuple[Path, Path]:
    """(near, far) DEM GeoTIFFs for a landing outside the US, the shape terrain_route.acquire returns.

    ``near_source`` is a source id (``plan().near``); when it fails (no data here, server down) the near patch is
    GLO-30 instead, said loudly and tagged on the file. The far DEM is GLO-30."""
    near_spec, far_spec = spec_for(lat, lon, zone, NEAR_RADIUS_M), spec_for(lat, lon, zone, FAR_RADIUS_M)
    if near_source == FALLBACK:
        near, _ = GLO30Provider().acquire(near_spec, cache)
        _tag_fallback(near, "no finer national DEM for this country")
    else:
        try:
            near, _ = provider(near_source).acquire(near_spec, cache)
        except Exception as exc:  # noqa: BLE001 - any failure of the national source: fall back, loudly
            print(f"terrain: {near_source} failed ({type(exc).__name__}: {str(exc)[:200]}); near DEM from GLO-30",
                  flush=True)
            near, _ = GLO30Provider().acquire(near_spec, cache)
            _tag_fallback(near, f"{near_source}: {type(exc).__name__}: {exc}")
    far, _ = GLO30Provider().acquire(far_spec, cache)
    return near, far
