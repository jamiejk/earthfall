from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer
from pyproj.transformer import TransformerGroup
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
import trimesh

from .coordinates import validate_scene_coordinates, vertical_transformer
from .models import GeoSceneSpec, SourceRecord, TerrainAsset
from .providers import crop_raster, sha256


def _grid(spec: GeoSceneSpec, spacing: float):
    width=2*spec.radius_m/spacing
    if not math.isclose(width,round(width),abs_tol=1e-8):
        raise ValueError("area width must be divisible by raster and mesh spacing")
    n=int(round(width))
    x0,y0,x1,y1=spec.bounds
    return n,from_origin(x0,y1,spacing,spacing)


def _bilinear_source(pixels: np.ndarray, transform, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Sample source pixel centres, requiring all contributing pixels to be valid."""
    edge_col,edge_row=(~transform)*(sx,sy)
    col=edge_col-0.5
    row=edge_row-0.5
    rows,cols=pixels.shape
    inside=(col>=0)&(row>=0)&(col<=cols-1)&(row<=rows-1)
    col=np.clip(col,0,cols-1)
    row=np.clip(row,0,rows-1)
    c0=np.floor(col).astype("int32"); r0=np.floor(row).astype("int32")
    c1=np.minimum(c0+1,cols-1); r1=np.minimum(r0+1,rows-1)
    q00=pixels[r0,c0]; q01=pixels[r0,c1]; q10=pixels[r1,c0]; q11=pixels[r1,c1]
    valid=inside&np.isfinite(q00)&np.isfinite(q01)&np.isfinite(q10)&np.isfinite(q11)
    tx=col-c0; ty=row-r0
    out=q00*(1-tx)*(1-ty)+q01*tx*(1-ty)+q10*(1-tx)*ty+q11*tx*ty
    out[~valid]=np.nan
    return out.astype("float32")


def _normalize_raster(crop: Path, spec: GeoSceneSpec, source: SourceRecord,
                      output: Path, spacing: float, exploratory: bool):
    n,affine=_grid(spec,spacing)
    if CRS.from_user_input(source.horizontal_crs)==CRS.from_user_input(spec.projected_crs):
        horizontal_operation="identity: source and project horizontal CRS agree"
        horizontal_accuracy=0.0
        inverse_horizontal=None
    else:
        group=TransformerGroup(spec.projected_crs,source.horizontal_crs,
                               always_xy=True,allow_ballpark=False)
        if not group.best_available or not group.transformers:
            missing=sorted({grid.short_name for op in group.unavailable_operations
                            for grid in op.grids if not grid.available})
            raise ValueError("required horizontal transformation grids are unavailable: "+", ".join(missing))
        inverse_horizontal=group.transformers[0]
        horizontal_operation=inverse_horizontal.description
        horizontal_accuracy=float(inverse_horizontal.accuracy)
    with rasterio.open(crop) as src:
        raw=src.read(1,masked=True)
        scale,offset=src.scales[0],src.offsets[0]
        physical=raw.astype("float32")*scale+offset
        if source.nodata_policy=="zero_as_unknown":
            physical=np.ma.masked_where(np.asarray(raw)==0,physical)
        pixels=physical.filled(np.nan).astype("float32")
        metric=np.full((n,n),np.nan,dtype="float32")
        if inverse_horizontal is None:
            reproject(pixels,metric,src_transform=src.transform,src_crs=src.crs,
                      dst_transform=affine,dst_crs=spec.projected_crs,
                      src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.bilinear)
            source_mask=(~np.ma.getmaskarray(physical)).astype("uint8")
            target_mask=np.zeros((n,n),dtype="uint8")
            reproject(source_mask,target_mask,src_transform=src.transform,src_crs=src.crs,
                      dst_transform=affine,dst_crs=spec.projected_crs,
                      src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
            metric[target_mask==0]=np.nan
        else:
            x0,y0,x1,y1=spec.bounds
            xs=x0+(np.arange(n,dtype="float64")+0.5)*spacing
            for row0 in range(0,n,128):
                row1=min(row0+128,n)
                ys=y1-(np.arange(row0,row1,dtype="float64")+0.5)*spacing
                xx,yy=np.meshgrid(xs,ys)
                sx,sy=inverse_horizontal.transform(xx,yy)
                metric[row0:row1]=_bilinear_source(pixels,src.transform,sx,sy)
    if not np.isfinite(metric).any():
        raise ValueError("source has no valid coverage in project area")
    if not source.vertical_reference_verified or not source.vertical_crs:
        if not exploratory:
            raise ValueError("source vertical datum is unverified; use explicit exploratory preview only")
        operation="unverified source heights; no vertical transformation"
        alignment="exploratory_unaligned"
        accuracy=None
    elif source.vertical_crs==spec.vertical_crs:
        operation="identity: source and project vertical CRS agree"
        alignment="verified"
        accuracy=0.0
    else:
        transformer=vertical_transformer(source.horizontal_crs,source.vertical_crs,
                                         spec.projected_crs,spec.vertical_crs)
        inverse=Transformer.from_crs(spec.projected_crs,source.horizontal_crs,
                                     always_xy=True,allow_ballpark=False)
        accuracy=float(transformer.accuracy)
        operation=transformer.description
        x0,y0,x1,y1=spec.bounds
        xs=x0+(np.arange(n,dtype="float64")+0.5)*spacing
        for row0 in range(0,n,128):
            row1=min(row0+128,n)
            ys=y1-(np.arange(row0,row1,dtype="float64")+0.5)*spacing
            xx,yy=np.meshgrid(xs,ys)
            block=metric[row0:row1]
            valid=np.isfinite(block)
            if valid.any():
                sx,sy=inverse.transform(xx[valid],yy[valid])
                _,_,heights=transformer.transform(sx,sy,block[valid].astype("float64"))
                if not np.all(np.isfinite(heights)):
                    raise ValueError("vertical operation produced invalid heights")
                block[valid]=np.asarray(heights,dtype="float32")
        alignment="verified"
    profile={"driver":"GTiff","height":n,"width":n,"count":1,"dtype":"float32",
             "crs":spec.projected_crs,"transform":affine,"nodata":-9999.0,
             "compress":"deflate","predictor":3,"tiled":True}
    with rasterio.open(output,"w",**profile) as dst:
        dst.write(np.where(np.isfinite(metric),metric,-9999).astype("float32"),1)
        dst.write_mask(np.isfinite(metric).astype("uint8")*255)
        dst.update_tags(vertical_crs=spec.vertical_crs if alignment=="verified" else "UNVERIFIED",
                        source_vertical_crs=source.vertical_crs or "UNKNOWN",
                        vertical_operation=operation,height_alignment=alignment)
    return metric,operation,accuracy,alignment,horizontal_operation,horizontal_accuracy


def _sample_nodes(values: np.ndarray, spacing: float, bounds: tuple[float,float,float,float],
                  node_spacing: float):
    n=values.shape[0]
    nodes=int(round((bounds[2]-bounds[0])/node_spacing))+1
    xs=np.linspace(bounds[0],bounds[2],nodes,dtype="float64")
    ys=np.linspace(bounds[1],bounds[3],nodes,dtype="float64")
    cols=np.clip((xs-bounds[0])/spacing-0.5,0,n-1)
    rows=np.clip((bounds[3]-ys)/spacing-0.5,0,n-1)
    cc,rr=np.meshgrid(cols,rows)
    c0=np.floor(cc).astype("int32"); r0=np.floor(rr).astype("int32")
    c1=np.minimum(c0+1,n-1); r1=np.minimum(r0+1,n-1)
    q00=values[r0,c0]; q01=values[r0,c1]; q10=values[r1,c0]; q11=values[r1,c1]
    valid=np.isfinite(q00)&np.isfinite(q01)&np.isfinite(q10)&np.isfinite(q11)
    tx=cc-c0; ty=rr-r0
    heights=(q00*(1-tx)*(1-ty)+q01*tx*(1-ty)+q10*(1-tx)*ty+q11*tx*ty)
    heights[~valid]=np.nan
    return xs,ys,heights


def _export_mesh(values: np.ndarray, spec: GeoSceneSpec, raster_spacing: float,
                 mesh_spacing: float, output: Path) -> tuple[int,int]:
    xs,ys,h=_sample_nodes(values,raster_spacing,spec.bounds,mesh_spacing)
    width=len(xs)
    xx,yy=np.meshgrid(xs-spec.origin_easting_m,ys-spec.origin_northing_m)
    valid=np.isfinite(h)
    # glTF is Y up. This rotation maps local east/north/up to east/up/-north.
    verts=np.column_stack((xx.ravel(),np.nan_to_num(h).ravel(),-yy.ravel()))
    ids=np.arange(width*width,dtype="int32").reshape(width,width)
    a=ids[:-1,:-1]; b=ids[:-1,1:]; c=ids[1:,:-1]; d=ids[1:,1:]
    # Row index increases northward. These orders produce upward-facing normals.
    first=np.stack((a,b,c),axis=-1).reshape(-1,3)
    second=np.stack((b,d,c),axis=-1).reshape(-1,3)
    faces=np.concatenate((first,second))
    keep=np.all(valid.ravel()[faces],axis=1)
    faces=faces[keep]
    if not len(faces):
        raise ValueError("mesh grid has no complete valid triangles")
    used=np.unique(faces)
    remap=np.full(len(verts),-1,dtype="int32")
    remap[used]=np.arange(len(used),dtype="int32")
    mesh=trimesh.Trimesh(vertices=verts[used],faces=remap[faces],process=False)
    output.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
    return len(used),len(faces)


def _preview(values: np.ndarray, spec: GeoSceneSpec, source: SourceRecord,
             alignment: str, output: Path) -> None:
    valid=np.isfinite(values)
    lo,hi=np.percentile(values[valid],[2,98])
    if hi<=lo: hi=lo+1
    scaled=np.nan_to_num(np.clip((values-lo)/(hi-lo),0,1),nan=0)
    rgb=np.empty((*values.shape,3),dtype="uint8")
    rgb[...,0]=(55+175*scaled).astype("uint8")
    rgb[...,1]=(82+130*scaled).astype("uint8")
    rgb[...,2]=(75+92*scaled).astype("uint8")
    rgb[~valid]=(29,45,65)
    image=Image.fromarray(rgb,mode="RGB").resize((900,900),Image.Resampling.BILINEAR)
    canvas=Image.new("RGB",(960,1040),(245,245,241))
    canvas.paste(image,(30,75))
    draw=ImageDraw.Draw(canvas)
    draw.text((30,20),f"{spec.name} — {source.dataset}",fill=(15,20,25))
    draw.text((30,45),f"North ↑    {int(spec.radius_m*2)} m square    {alignment}",fill=(15,20,25))
    draw.text((30,990),f"Valid: {100*valid.mean():.1f}%    Elevation: {values[valid].min():.1f} to {values[valid].max():.1f} m",fill=(15,20,25))
    draw.line((40,955,340,955),fill=(0,0,0),width=4)
    draw.text((42,962),f"{spec.radius_m*2/3:g} m",fill=(15,20,25))
    canvas.save(output)


def compile_terrain(spec: GeoSceneSpec, provider, output_dir: Path,
                    raw_cache: Path, variant: str, raster_spacing: float,
                    mesh_spacing: float, exploratory: bool=False) -> TerrainAsset:
    validate_scene_coordinates(spec)
    if raster_spacing<=0 or mesh_spacing<=0:
        raise ValueError("spacings must be positive")
    output_dir.mkdir(parents=True,exist_ok=True)
    source_path,source=provider.acquire(spec,raw_cache)
    crop=output_dir/"terrain_source_crop.tif"
    metric_path=output_dir/"terrain_metric.tif"
    mesh_path=output_dir/"terrain.glb"
    preview=output_dir/"preview_topdown.png"
    crop_raster(source_path,spec,crop,margin_m=max(90,2*raster_spacing))
    values,operation,accuracy,alignment,horizontal_op,horizontal_accuracy=_normalize_raster(
        crop,spec,source,metric_path,raster_spacing,exploratory)
    count,triangles=_export_mesh(values,spec,raster_spacing,mesh_spacing,mesh_path)
    _preview(values,spec,source,alignment,preview)
    attribution=output_dir/"ATTRIBUTION.md"
    attribution.write_text(f"# Terrain attribution\n\n{source.attribution or source.dataset}\n\nLicense: {source.license or 'not verified'}\nSource: {source.source_uri}\n",encoding="utf-8")
    names=[crop.name,metric_path.name,mesh_path.name,preview.name,attribution.name]
    asset=TerrainAsset(project_id=spec.project_id,variant=variant,source=source,
        project_crs_wkt=CRS.from_user_input(spec.projected_crs).to_wkt(),
        horizontal_operation=horizontal_op,horizontal_operation_accuracy_m=horizontal_accuracy,
        origin_easting_m=spec.origin_easting_m,origin_northing_m=spec.origin_northing_m,
        bounds_local_m=(-spec.radius_m,-spec.radius_m,spec.radius_m,spec.radius_m),
        common_vertical_crs=spec.vertical_crs,vertical_operation=operation,
        vertical_operation_accuracy_m=accuracy,height_alignment=alignment,
        raster_spacing_m=raster_spacing,mesh_spacing_m=mesh_spacing,
        mesh_vertices=count,mesh_triangles=triangles,
        coverage_fraction=float(np.isfinite(values).mean()),
        elevation_min_m=float(np.nanmin(values)),elevation_max_m=float(np.nanmax(values)),
        files_sha256={name:sha256(output_dir/name) for name in names})
    (output_dir/"metadata.json").write_text(asset.model_dump_json(indent=2)+"\n",encoding="utf-8")
    return asset


def compare_assets(a: TerrainAsset,b: TerrainAsset) -> dict:
    problems=[]
    if a.project_id!=b.project_id: problems.append("project IDs differ")
    if a.project_crs_wkt!=b.project_crs_wkt: problems.append("project CRS differs")
    if a.common_vertical_crs!=b.common_vertical_crs: problems.append("common vertical CRS differs")
    if a.height_alignment!="verified" or b.height_alignment!="verified":
        problems.append("one or both source heights are unverified")
    for field in ("origin_easting_m","origin_northing_m"):
        if abs(getattr(a,field)-getattr(b,field))>0.01: problems.append(field+" differs by over 1 cm")
    if max(abs(x-y) for x,y in zip(a.bounds_local_m,b.bounds_local_m))>0.01:
        problems.append("local XY bounds differ by over 1 cm")
    return {"matched":not problems,"problems":problems,
            "variants":[a.variant,b.variant],
            "sources":[a.source.dataset,b.source.dataset],
            "coverage_fraction":[a.coverage_fraction,b.coverage_fraction],
            "elevation_ranges_m":[[a.elevation_min_m,a.elevation_max_m],
                                   [b.elevation_min_m,b.elevation_max_m]],
            "height_operation_accuracy_m":[a.vertical_operation_accuracy_m,
                                           b.vertical_operation_accuracy_m]}
