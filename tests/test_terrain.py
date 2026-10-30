import json

import numpy as np
import pytest
import rasterio
import trimesh
from rasterio.transform import from_origin
from pyproj import Transformer

from geoblock_core.models import GeoSceneSpec, TerrainAsset
from geoblock_core.processing import compare_assets, compile_terrain
from geoblock_core.providers import LocalGeoTIFFProvider


def scene():
    lon,lat=-4.31,51.565
    x,y=Transformer.from_crs(4326,27700,always_xy=True).transform(lon,lat)
    return GeoSceneSpec(project_id="test",name="Known slope",longitude=lon,latitude=lat,
        origin_easting_m=x,origin_northing_m=y,radius_m=30,
        location_source="test coordinate")


def raster(path,spec):
    x0,y0,x1,y1=spec.bounds
    rows,cols=np.indices((64,64))
    height=(rows*0.2+cols*0.1-5).astype("float32")
    height[24:30,24:30]=-9999
    with rasterio.open(path,"w",driver="GTiff",width=64,height=64,count=1,
        dtype="float32",crs=spec.projected_crs,
        transform=from_origin(x0-2,y1+2,1,1),nodata=-9999) as dst:
        dst.write(height,1)


def test_compiler_preserves_extent_holes_and_heights(tmp_path):
    spec=scene()
    source=tmp_path/"source.tif"
    raster(source,spec)
    provider=LocalGeoTIFFProvider(source,dataset="synthetic",vertical_crs="EPSG:5701",
                                   vertical_verified=True)
    out=tmp_path/"out"
    asset=compile_terrain(spec,provider,out,tmp_path/"raw","A",1,5)
    assert asset.height_alignment=="verified"
    assert asset.bounds_local_m==(-30,-30,30,30)
    assert asset.mesh_triangles>0
    with rasterio.open(out/"terrain_metric.tif") as ds:
        assert ds.bounds.left==pytest.approx(spec.bounds[0])
        assert ds.bounds.right==pytest.approx(spec.bounds[2])
        assert ds.read(1,masked=True).mask[23:29,23:29].any()
        assert ds.read(1,masked=True).min()<0
    assert (out/"terrain.glb").stat().st_size>100
    mesh=next(iter(trimesh.load(out/"terrain.glb",force="scene").geometry.values()))
    assert np.mean(mesh.face_normals[:,1])>0.5
    assert mesh.bounds[0,0]==pytest.approx(-30)
    assert mesh.bounds[1,0]==pytest.approx(30)
    assert TerrainAsset.model_validate_json((out/"metadata.json").read_text()).source.sha256
    assert compare_assets(asset,asset)["matched"]


def test_unknown_vertical_reference_blocks_matched_export(tmp_path):
    spec=scene()
    source=tmp_path/"source.tif"
    raster(source,spec)
    provider=LocalGeoTIFFProvider(source,dataset="unknown",vertical_crs=None,
                                   vertical_verified=False)
    with pytest.raises(ValueError,match="unverified"):
        compile_terrain(spec,provider,tmp_path/"blocked",tmp_path/"raw","B",1,5)
    preview=compile_terrain(spec,provider,tmp_path/"preview",tmp_path/"raw","B",1,5,
                            exploratory=True)
    assert preview.height_alignment=="exploratory_unaligned"
    assert not compare_assets(preview,preview)["matched"]


def test_scale_and_zero_as_unknown_are_applied_before_resampling(tmp_path):
    spec=scene()
    source=tmp_path/"scaled.tif"
    x0,y0,x1,y1=spec.bounds
    raw=np.full((64,64),120,dtype="int16")
    raw[:20,:]=0
    with rasterio.open(source,"w",driver="GTiff",width=64,height=64,count=1,
        dtype="int16",crs=spec.projected_crs,
        transform=from_origin(x0-2,y1+2,1,1),nodata=-32768) as dst:
        dst.write(raw,1)
        dst.scales=(0.1,)
    provider=LocalGeoTIFFProvider(source,dataset="scaled",vertical_crs="EPSG:5701",
                                   vertical_verified=True,zero_is_nodata=True)
    out=tmp_path/"scaled-out"
    asset=compile_terrain(spec,provider,out,tmp_path/"raw","A",1,5)
    with rasterio.open(out/"terrain_metric.tif") as ds:
        a=ds.read(1,masked=True)
        assert a.mask[5,10]
        assert float(a[30,10])==pytest.approx(12.0,abs=0.01)
    assert asset.source.nodata_policy=="zero_as_unknown"
