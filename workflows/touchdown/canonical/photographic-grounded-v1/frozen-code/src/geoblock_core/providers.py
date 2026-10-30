from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
import json
import os
from pathlib import Path
import tempfile
import fcntl

import httpx
import numpy as np
import rasterio
from rasterio.windows import from_bounds

from .coordinates import geographic_envelope
from .models import GeoSceneSpec, SourceRecord


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def grid_aligned(bounds: tuple[float, float, float, float], origin: tuple[float, float],
                 res: float) -> tuple[float, float, float, float]:
    """Expand ``bounds`` outward onto the source's own pixel grid (origin = its left, top), so a merge
    copies cells instead of resampling them half a pixel off: the same place then reads the same heights
    whatever window it was fetched in."""
    left, top = origin
    west, south, east, north = bounds
    return (left + math.floor((west - left) / res) * res, top - math.ceil((top - south) / res) * res,
            left + math.ceil((east - left) / res) * res, top - math.floor((top - north) / res) * res)


def crop_raster(source: str | Path, spec: GeoSceneSpec, output: Path,
                margin_m: float = 90) -> None:
    """Save a native-resolution subset with its original values and mask."""
    x0,y0,x1,y1=spec.bounds
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(str(source)) as src:
            from rasterio.warp import transform_bounds
            left,bottom,right,top=transform_bounds(spec.projected_crs,src.crs,
                x0-margin_m,y0-margin_m,x1+margin_m,y1+margin_m,densify_pts=21)
            window=from_bounds(left,bottom,right,top,transform=src.transform)
            window=window.round_offsets().round_lengths()
            data=src.read(1,window=window,boundless=True,masked=True)
            profile=src.profile.copy()
            profile.update(width=data.shape[1],height=data.shape[0],
                           transform=src.window_transform(window),compress="deflate",tiled=True,
                           nodata=src.nodata if src.nodata is not None else -9999)
            output.parent.mkdir(parents=True,exist_ok=True)
            with rasterio.open(output,"w",**profile) as dst:
                dst.write(data.filled(src.nodata if src.nodata is not None else -9999),1)
                dst.write_mask((~np.ma.getmaskarray(data)).astype("uint8")*255)
                dst.scales=src.scales
                dst.offsets=src.offsets
                dst.update_tags(**src.tags())


class LocalGeoTIFFProvider:
    def __init__(self, path: Path, *, dataset: str, vertical_crs: str | None,
                 vertical_verified: bool, license: str | None = None,
                 attribution: str | None = None, acquisition_dates: str | None = None,
                 source_uri: str | None = None, zero_is_nodata: bool = False):
        self.path=path.expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.dataset=dataset
        self.vertical_crs=vertical_crs
        self.vertical_verified=vertical_verified
        self.license=license
        self.attribution=attribution
        self.acquisition_dates=acquisition_dates
        self.source_uri=source_uri
        self.zero_is_nodata=zero_is_nodata

    def acquire(self, spec: GeoSceneSpec, raw_cache: Path) -> tuple[Path, SourceRecord]:
        digest=sha256(self.path)
        with rasterio.open(self.path) as src:
            if src.crs is None:
                raise ValueError("local GeoTIFF has no horizontal CRS")
            from rasterio.warp import transform_bounds
            b=transform_bounds(spec.projected_crs,src.crs,*spec.bounds,densify_pts=21)
            if not (src.bounds.left < b[2] and b[0] < src.bounds.right and
                    src.bounds.bottom < b[3] and b[1] < src.bounds.top):
                raise ValueError("local GeoTIFF does not intersect scene area")
            record=SourceRecord(provider="LocalGeoTIFF",dataset=self.dataset,
                horizontal_crs=src.crs.to_string(),vertical_crs=self.vertical_crs,
                vertical_reference_verified=self.vertical_verified,
                native_pixel_spacing=tuple(abs(float(x)) for x in src.res),
                native_pixel_spacing_units="metres" if src.crs.is_projected else "degrees",
                nodata_policy="zero_as_unknown" if self.zero_is_nodata else "source_mask_only",
                license=self.license,attribution=self.attribution,
                acquisition_dates=self.acquisition_dates,
                source_uri=self.source_uri or self.path.as_uri(),
                retrieval_utc=utc_now(),sha256=digest)
        return self.path,record


class USGS3DEPProvider:
    """USGS 3DEP 1/3 arc-second seamless DEM, read by window from the public bucket.

    Tiles are 1 x 1 degree, named by their north-west corner, so no catalogue
    query is needed. Only the requested window of each tile is transferred.
    Heights are NAVD88 metres; coverage is limited to the conterminous US,
    where that vertical reference is documented for the seamless product.
    """

    default_root="https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"
    conus=(-125.0,24.0,-66.0,50.0)

    def __init__(self, tile_root: str | None = None, read_resolution_deg: float | None = None):
        self.tile_root=(tile_root or os.environ.get("GEOBLOCK_3DEP_ROOT") or self.default_root).rstrip("/")
        # Coarser than native (1/10800 deg) reads from the COG overviews: wide regional scenes stay small.
        self.read_resolution_deg=read_resolution_deg

    @staticmethod
    def tile_names(west: float, south: float, east: float, north: float) -> list[str]:
        names=[]
        for top in range(math.floor(south)+1,math.ceil(north)+1):
            for left in range(math.floor(west),math.ceil(east)):
                names.append(f"n{top:02d}w{-left:03d}")
        return names

    def tile_uri(self, name: str) -> str:
        path=f"{self.tile_root}/{name}/USGS_13_{name}.tif"
        return "/vsicurl/"+path if path.startswith(("http://","https://")) else path

    def acquire(self, spec: GeoSceneSpec, raw_cache: Path) -> tuple[Path, SourceRecord]:
        west,south,east,north=geographic_envelope(spec)
        w0,s0,e0,n0=self.conus
        if not (w0<=west and east<=e0 and s0<=south and north<=n0):
            raise ValueError("USGS 3DEP seamless terrain is only used inside the conterminous US "
                             "(NAVD88 heights); choose another provider for this place.")
        names=self.tile_names(west,south,east,north)
        request={"root":self.tile_root,"bbox":[round(v,7) for v in (west,south,east,north)]}
        if self.read_resolution_deg:
            request["resolution_deg"]=self.read_resolution_deg
        key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        directory=raw_cache/"usgs3dep13"/key
        directory.mkdir(parents=True,exist_ok=True)
        raster=directory/"source.tif"
        uris=[self.tile_uri(name) for name in names]
        with (directory/".lock").open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not raster.is_file():
                from rasterio.merge import merge
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                                  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
                    sources=[]
                    missing=[]
                    try:
                        for uri in uris:
                            try:
                                sources.append(rasterio.open(uri))
                            except rasterio.errors.RasterioIOError:
                                missing.append(uri)  # open sea (or outside the product) has no tile
                        if not sources:
                            raise ValueError(f"No USGS 3DEP 1/3 arc-second tile at {', '.join(missing)}")
                        nodata=sources[0].nodata if sources[0].nodata is not None else -999999.0
                        res=(self.read_resolution_deg,)*2 if self.read_resolution_deg else None
                        grid=res[0] if res else abs(sources[0].res[0])
                        aligned=grid_aligned((west,south,east,north),(sources[0].bounds.left,sources[0].bounds.top),grid)
                        data,transform=merge(sources,bounds=aligned,nodata=nodata,res=res)
                        profile={"driver":"GTiff","height":data.shape[1],"width":data.shape[2],
                                 "count":1,"dtype":data.dtype,"crs":sources[0].crs,
                                 "transform":transform,"nodata":nodata,
                                 "compress":"deflate","predictor":3,"tiled":True}
                    finally:
                        for source in sources:
                            source.close()
                if not np.isfinite(data).any() or np.all(data==nodata):
                    raise ValueError("USGS 3DEP returned no valid elevation for this area")
                if missing:
                    data[data==nodata]=0.0  # a missing tile is sea; wide regional scenes reach offshore
                temp=raster.with_suffix(".part.tif")
                with rasterio.open(temp,"w",**profile) as dst:
                    dst.write(data)
                    dst.update_tags(vertical_crs="EPSG:5703",vertical_datum="NAVD88",
                                    source_tiles=",".join(names),
                                    sea_filled_tiles=",".join(uri.rsplit("/",1)[-1] for uri in missing))
                os.replace(temp,raster)
        with rasterio.open(raster) as src:
            record=SourceRecord(provider="USGS3DEP",
                dataset="USGS 3DEP 1/3 arc-second DEM (seamless, current)",
                horizontal_crs=src.crs.to_string(),vertical_crs="EPSG:5703",
                vertical_reference_verified=True,nominal_resolution_m=10,
                native_pixel_spacing=tuple(abs(float(x)) for x in src.res),
                native_pixel_spacing_units="degrees",
                license="Public domain (U.S. Geological Survey)",
                attribution="U.S. Geological Survey, 3D Elevation Program (3DEP), The National Map",
                acquisition_dates="Seamless composite; source survey dates vary by location",
                source_uri=" ".join(uri.removeprefix("/vsicurl/") for uri in uris),
                retrieval_utc=utc_now(),sha256=sha256(raster))
        return raster,record


class OpenTopographyProvider:
    endpoint="https://portal.opentopography.org/API/globaldem"

    def __init__(self, api_key: str | None = None):
        self.api_key=api_key or os.environ.get("OPENTOPOGRAPHY_API_KEY")

    def acquire(self, spec: GeoSceneSpec, raw_cache: Path) -> tuple[Path, SourceRecord]:
        bbox=geographic_envelope(spec)
        normalized=[round(v,7) for v in bbox]
        request={"dataset":"GEDTM30","bbox":normalized,"format":"GTiff"}
        key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        directory=raw_cache/"opentopography"/key
        directory.mkdir(parents=True,exist_ok=True)
        raster=directory/"source.tif"
        lockfile=directory/".lock"
        with lockfile.open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not raster.is_file():
                if not self.api_key:
                    raise ValueError("GEDTM30 is not cached; configure OPENTOPOGRAPHY_API_KEY")
                params={"demtype":"GEDTM30","west":normalized[0],"south":normalized[1],
                        "east":normalized[2],"north":normalized[3],
                        "outputFormat":"GTiff","API_Key":self.api_key}
                with tempfile.NamedTemporaryFile(dir=directory,suffix=".part",delete=False) as tmp:
                    temp=Path(tmp.name)
                    try:
                        with httpx.Client(timeout=120,follow_redirects=True) as client:
                            with client.stream("GET",self.endpoint,params=params) as response:
                                if response.status_code in (401,403):
                                    raise ValueError("OpenTopography rejected the API key or access level")
                                if response.status_code != 200:
                                    raise ValueError(f"OpenTopography returned HTTP {response.status_code}")
                                for chunk in response.iter_bytes():
                                    tmp.write(chunk)
                        tmp.flush()
                        with rasterio.open(temp) as check:
                            if check.count != 1 or check.crs is None:
                                raise ValueError("OpenTopography response is not a georeferenced single-band DEM")
                        os.replace(temp,raster)
                    finally:
                        temp.unlink(missing_ok=True)
        with rasterio.open(raster) as src:
            record=SourceRecord(provider="OpenTopography",dataset="GEDTM30 (API response; version unverified)",
                horizontal_crs=src.crs.to_string(),vertical_crs="EPSG:3855",
                vertical_reference_verified=True,nominal_resolution_m=30,
                native_pixel_spacing=tuple(abs(float(x)) for x in src.res),
                native_pixel_spacing_units="metres" if src.crs.is_projected else "degrees",
                license="CC BY 4.0",
                attribution="GEDTM30 (2025), distributed by OpenTopography, DOI 10.5069/G9BV7DT1",
                acquisition_dates="2006–2015 source surveys; API release requires inspection",
                source_uri="https://doi.org/10.5069/G9BV7DT1",sha256=sha256(raster),
                retrieval_utc=utc_now())
        return raster,record


class CopernicusDEMProvider:
    """Copernicus DEM GLO-30 (about 30 m, worldwide), read by window from the AWS open-data bucket.

    Used for places outside the conterminous US. Tiles are 1 x 1 degree, named by their south-west
    corner; open ocean has no tile and is filled as sea level. Heights are EGM2008 metres. It is a
    surface model, so forest canopy and buildings are in the heights.
    """

    default_root="https://copernicus-dem-30m.s3.amazonaws.com"
    tile_list_name="tileList.txt"
    native_deg=1/3600

    def __init__(self, tile_root: str | None = None, read_resolution_deg: float | None = None):
        self.tile_root=(tile_root or os.environ.get("GEOBLOCK_COPDEM_ROOT") or self.default_root).rstrip("/")
        self.read_resolution_deg=read_resolution_deg

    @staticmethod
    def tile_names(west: float, south: float, east: float, north: float) -> list[str]:
        names=[]
        for bottom in range(math.floor(south),math.ceil(north)):
            for left in range(math.floor(west),math.ceil(east)):
                lat=f"{'N' if bottom>=0 else 'S'}{abs(bottom):02d}_00"
                lon=f"{'E' if left>=0 else 'W'}{abs(left):03d}_00"
                names.append(f"Copernicus_DSM_COG_10_{lat}_{lon}_DEM")
        return names

    def tile_uri(self, name: str) -> str:
        path=f"{self.tile_root}/{name}/{name}.tif"
        return "/vsicurl/"+path if path.startswith(("http://","https://")) else path

    def existing_tiles(self, raw_cache: Path) -> set[str] | None:
        """Names of the land tiles, from the bucket's own list (cached); None if it cannot be read."""
        cached=raw_cache/"copernicus_glo30"/self.tile_list_name
        if not cached.is_file():
            source=f"{self.tile_root}/{self.tile_list_name}"
            try:
                if source.startswith(("http://","https://")):
                    with httpx.Client(timeout=60,follow_redirects=True) as client:
                        response=client.get(source)
                        response.raise_for_status()
                        text=response.text
                else:
                    text=Path(source).read_text()
            except (httpx.HTTPError,OSError):
                return None
            cached.parent.mkdir(parents=True,exist_ok=True)
            cached.write_text(text)
        return {line.strip() for line in cached.read_text().splitlines() if line.strip()}

    def acquire(self, spec: GeoSceneSpec, raw_cache: Path) -> tuple[Path, SourceRecord]:
        west,south,east,north=geographic_envelope(spec)
        names=self.tile_names(west,south,east,north)
        land=self.existing_tiles(raw_cache)
        present=[name for name in names if land is None or name in land]
        if not present:
            raise ValueError("Copernicus DEM has no land tile here (open sea)")
        resolution=self.read_resolution_deg or self.native_deg
        request={"root":self.tile_root,"bbox":[round(v,7) for v in (west,south,east,north)],
                 "resolution_deg":resolution}
        key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        directory=raw_cache/"copernicus_glo30"/key
        directory.mkdir(parents=True,exist_ok=True)
        raster=directory/"source.tif"
        uris=[self.tile_uri(name) for name in present]
        with (directory/".lock").open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not raster.is_file():
                from rasterio.merge import merge
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                                  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
                    sources=[]
                    try:
                        for uri in uris:
                            try:
                                sources.append(rasterio.open(uri))
                            except rasterio.errors.RasterioIOError as exc:
                                raise ValueError(f"No Copernicus DEM tile at {uri}") from exc
                        # Tiles north of 50 degrees are narrower in longitude; one grid for all.
                        aligned=grid_aligned((west,south,east,north),(sources[0].bounds.left,sources[0].bounds.top),
                                             resolution)
                        data,transform=merge(sources,bounds=aligned,nodata=-32767.0,
                                             res=(resolution,resolution))
                    finally:
                        for source in sources:
                            source.close()
                data=data.astype("float32")
                sea=data==-32767.0
                if sea.all():
                    raise ValueError("Copernicus DEM returned no land for this area")
                data[sea]=0.0  # no tile is open sea
                profile={"driver":"GTiff","height":data.shape[1],"width":data.shape[2],
                         "count":1,"dtype":"float32","crs":"EPSG:4326","transform":transform,
                         "nodata":-32767.0,"compress":"deflate","predictor":3,"tiled":True}
                temp=raster.with_suffix(".part.tif")
                with rasterio.open(temp,"w",**profile) as dst:
                    dst.write(data)
                    dst.update_tags(vertical_crs="EPSG:3855",vertical_datum="EGM2008",
                                    source_tiles=",".join(present),sea_filled_tiles=",".join(
                                        name for name in names if name not in present))
                os.replace(temp,raster)
        with rasterio.open(raster) as src:
            record=SourceRecord(provider="CopernicusDEM",source_type="DSM",
                dataset="Copernicus DEM GLO-30 (2021 release, surface model)",
                horizontal_crs="EPSG:4326",vertical_crs="EPSG:3855",
                vertical_reference_verified=True,nominal_resolution_m=30,
                native_pixel_spacing=tuple(abs(float(x)) for x in src.res),
                native_pixel_spacing_units="degrees",nodata_policy="missing_tile_is_sea_level",
                license="Copernicus DEM licence: free for any use, including commercial, with attribution",
                attribution="Copernicus DEM GLO-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
                            "2014-2018 provided under COPERNICUS by the European Union and ESA",
                acquisition_dates="TanDEM-X acquisitions 2011-2015",
                source_uri=" ".join(uri.removeprefix("/vsicurl/") for uri in uris),
                retrieval_utc=utc_now(),sha256=sha256(raster))
        return raster,record


class USGS1mDEMProvider:
    """USGS 3DEP 1 m bare-earth DEM of one lidar project, read by window from its 10 km tiles.

    For the small landing patch an eye-level shot ends on, where the 10 m seamless DEM (meshed at 20 m)
    is too coarse. Heights are NAVD88 metres in the project's NAD83 / UTM zone.
    """

    def __init__(self, project: str, zone: int, root: str | None = None, fallback_below: float = 0.8):
        from .lidar import DEM1M_ROOT
        self.project=project
        self.zone=zone
        self.fallback_below=fallback_below  # 1 m share of the square below which the whole square is 1/3" (0: raise)
        self.root=(root or os.environ.get("GEOBLOCK_DEM1M_ROOT") or DEM1M_ROOT).rstrip("/")

    def tile_uris(self, west: float, south: float, east: float, north: float) -> list[str]:
        uris=[]
        for x in range(int(west//10_000),int(east//10_000)+1):
            for y in range(math.ceil(south/10_000),math.ceil(north/10_000)+1):
                path=f"{self.root}/{self.project}/TIFF/USGS_1M_{self.zone}_x{x}y{y}_{self.project}.tif"
                uris.append("/vsicurl/"+path if path.startswith(("http://","https://")) else path)
        return uris

    @staticmethod
    def seamless_fill(spec: GeoSceneSpec, raw_cache: Path, transform, shape: tuple[int, int], crs: str) -> np.ndarray:
        """The 1/3 arc-second DEM on the 1 m grid, for the parts of the square the project does not reach."""
        from rasterio.warp import Resampling, reproject
        wider=spec.model_copy(update={"radius_m":spec.radius_m+200})  # the 1 m read reaches 60 m past the square
        source,_=USGS3DEPProvider().acquire(wider,raw_cache)
        out=np.full(shape,np.nan,dtype="float32")
        with rasterio.open(source) as src:
            reproject(rasterio.band(src,1),out,dst_transform=transform,dst_crs=crs,dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
        if np.isnan(out).any():
            raise ValueError("the 1/3 arc-second DEM does not cover the gaps in the 1 m project")
        return out

    def acquire(self, spec: GeoSceneSpec, raw_cache: Path) -> tuple[Path, SourceRecord]:
        from rasterio.warp import transform_bounds
        crs=f"EPSG:{26900+self.zone}"
        x0,y0,x1,y1=spec.bounds
        west,south,east,north=transform_bounds(spec.projected_crs,crs,x0-60,y0-60,x1+60,y1+60,densify_pts=21)
        request={"root":self.root,"project":self.project,"zone":self.zone,
                 "bounds":[round(v,2) for v in (west,south,east,north)]}
        key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        directory=raw_cache/"usgs3dep1m"/key
        directory.mkdir(parents=True,exist_ok=True)
        raster=directory/"source.tif"
        with (directory/".lock").open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not raster.is_file():
                from rasterio.merge import merge
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                                  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
                    sources=[]
                    try:
                        for uri in self.tile_uris(west,south,east,north):
                            try:
                                sources.append(rasterio.open(uri))
                            except rasterio.errors.RasterioIOError:
                                continue  # the project does not reach this tile
                        if not sources:
                            raise ValueError(f"3DEP 1 m project {self.project} has no tile here")
                        nodata=sources[0].nodata if sources[0].nodata is not None else -999999.0
                        aligned=grid_aligned((west,south,east,north),(sources[0].bounds.left,sources[0].bounds.top),
                                             abs(sources[0].res[0]))
                        data,transform=merge(sources,bounds=aligned,nodata=nodata)
                    finally:
                        for source in sources:
                            source.close()
                valid=np.isfinite(data)&(data!=nodata)
                filled=0.0
                fallback=None
                if valid.mean()<0.95:
                    # A project boundary can clip the square. Keep it if the middle (where the person stands and
                    # the camera holds) is all lidar, and fill the rest from the 1/3" DEM: same NAVD88 heights.
                    rows,cols=valid.shape[1:]
                    middle=valid[0,3*rows//8:5*rows//8,3*cols//8:5*cols//8]
                    data=data.astype("float32")
                    seamless=self.seamless_fill(spec,raw_cache,transform,data.shape[1:],crs)
                    if valid.mean()<self.fallback_below or not middle.all():
                        # Most places have no 1 m lidar (Battery Spencer, 18 Oct: the San Francisco project's edge
                        # runs through the landing, 26% covered): use the 1/3" (~10 m) DEM for the whole square
                        if not self.fallback_below:
                            raise ValueError(f"3DEP 1 m project {self.project} covers only {100*valid.mean():.0f}% of "
                                             "the landing area")
                        fallback=float(valid.mean())
                        data[0]=seamless
                        filled=1.0
                    else:
                        data[~valid]=seamless[~valid[0]]
                        filled=float(1-valid.mean())
                profile={"driver":"GTiff","height":data.shape[1],"width":data.shape[2],"count":1,
                         "dtype":data.dtype,"crs":crs,"transform":transform,"nodata":nodata,
                         "compress":"deflate","predictor":3,"tiled":True}
                temp=raster.with_suffix(".part.tif")
                with rasterio.open(temp,"w",**profile) as dst:
                    dst.write(data)
                    dst.update_tags(vertical_crs="EPSG:5703",vertical_datum="NAVD88",project=self.project,
                                    seamless_filled_share=f"{filled:.3f}",
                                    **({"dem_fallback_1m_share":f"{fallback:.3f}"} if fallback is not None else {}))
                os.replace(temp,raster)
        with rasterio.open(raster) as src:
            from .lidar import survey_year
            filled=float(src.tags().get("seamless_filled_share",0))
            fallback=src.tags().get("dem_fallback_1m_share")
            record=SourceRecord(provider="USGS3DEP1m",
                dataset=(f"USGS 3DEP 1/3 arc-second DEM on the 1 m grid (10 m fallback: 1 m project {self.project} "
                         f"covers {100*float(fallback):.0f}% of the landing square)") if fallback is not None else
                        f"USGS 3DEP 1 m DEM, lidar project {self.project}"
                        +(f" ({100*filled:.0f}% at the edge filled from the 1/3 arc-second DEM)" if filled else ""),
                horizontal_crs=crs,vertical_crs="EPSG:5703",vertical_reference_verified=True,
                nominal_resolution_m=10 if fallback is not None else 1,native_pixel_spacing=tuple(abs(float(x)) for x in src.res),
                native_pixel_spacing_units="metres",license="Public domain (U.S. Geological Survey)",
                attribution="U.S. Geological Survey, 3D Elevation Program 1 m DEM",
                acquisition_dates=str(survey_year(self.project) or "see project metadata"),
                source_uri=f"{self.root}/{self.project}/TIFF/",retrieval_utc=utc_now(),sha256=sha256(raster))
        return raster,record
