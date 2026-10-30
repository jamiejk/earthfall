from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pydantic import ValidationError
from pyproj import Transformer, datadir

from .coordinates import geographic_envelope, validate_scene_coordinates
from .models import GeoSceneSpec, TerrainAsset
from .processing import compare_assets, compile_terrain
from .providers import LocalGeoTIFFProvider, OpenTopographyProvider


def _parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(prog="geoblock",description="Compile matched metric terrain assets")
    commands=parser.add_subparsers(dest="command",required=True)
    init=commands.add_parser("init",help="Create a canonical project specification from WGS84 coordinates")
    init.add_argument("--project-id",required=True)
    init.add_argument("--name",required=True)
    init.add_argument("--longitude",type=float,required=True)
    init.add_argument("--latitude",type=float,required=True)
    init.add_argument("--centre-longitude",type=float)
    init.add_argument("--centre-latitude",type=float)
    init.add_argument("--location-source",required=True)
    init.add_argument("--projected-crs",default="EPSG:27700")
    init.add_argument("--vertical-crs",default="EPSG:5701")
    init.add_argument("--radius-m",type=float,default=1500)
    init.add_argument("--output",type=Path,required=True)
    inspect=commands.add_parser("inspect",help="Inspect a scene spec and source envelope")
    inspect.add_argument("--spec",type=Path,required=True)
    compile_cmd=commands.add_parser("compile",help="Compile one terrain variant")
    compile_cmd.add_argument("--spec",type=Path,required=True)
    compile_cmd.add_argument("--variant",required=True)
    compile_cmd.add_argument("--source",choices=("local","gedtm30"),required=True)
    compile_cmd.add_argument("--raster",type=Path)
    compile_cmd.add_argument("--dataset",default="Local DTM")
    compile_cmd.add_argument("--source-vertical-crs")
    compile_cmd.add_argument("--vertical-verified",action="store_true")
    compile_cmd.add_argument("--license")
    compile_cmd.add_argument("--attribution")
    compile_cmd.add_argument("--acquisition-dates")
    compile_cmd.add_argument("--source-uri")
    compile_cmd.add_argument("--zero-is-nodata",action="store_true")
    compile_cmd.add_argument("--proj-grid-dir",type=Path)
    compile_cmd.add_argument("--raster-spacing-m",type=float,required=True)
    compile_cmd.add_argument("--mesh-spacing-m",type=float,required=True)
    compile_cmd.add_argument("--exploratory",action="store_true")
    compile_cmd.add_argument("--output-dir",type=Path,required=True)
    compile_cmd.add_argument("--raw-cache",type=Path,default=Path("data/cache/raw"))
    compare=commands.add_parser("compare",help="Validate paired terrain metadata")
    compare.add_argument("a",type=Path)
    compare.add_argument("b",type=Path)
    compare.add_argument("--output",type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args=_parser().parse_args(argv)
    try:
        if args.command=="init":
            if (args.centre_longitude is None)!=(args.centre_latitude is None):
                raise ValueError("provide both centre coordinates or neither")
            centre_lon=args.centre_longitude if args.centre_longitude is not None else args.longitude
            centre_lat=args.centre_latitude if args.centre_latitude is not None else args.latitude
            x,y=Transformer.from_crs("EPSG:4326",args.projected_crs,
                                     always_xy=True,allow_ballpark=False).transform(
                                         centre_lon,centre_lat)
            spec=GeoSceneSpec(project_id=args.project_id,name=args.name,
                longitude=args.longitude,latitude=args.latitude,
                projected_crs=args.projected_crs,vertical_crs=args.vertical_crs,
                radius_m=args.radius_m,origin_easting_m=x,origin_northing_m=y,
                location_source=args.location_source,
                origin_source=f"manual provisional centre WGS84 ({centre_lon}, {centre_lat})")
            validate_scene_coordinates(spec)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(spec.model_dump_json(indent=2)+"\n",encoding="utf-8")
            result={"spec":str(args.output),"bounds":spec.bounds}
        elif args.command=="inspect":
            spec=GeoSceneSpec.model_validate_json(args.spec.read_text())
            result={"project_id":spec.project_id,"bounds_projected_m":spec.bounds,
                    "fetch_envelope_wgs84":geographic_envelope(spec),
                    "roundtrip_error_m":validate_scene_coordinates(spec)}
        elif args.command=="compile":
            if args.proj_grid_dir:
                if not args.proj_grid_dir.is_dir():
                    raise ValueError("--proj-grid-dir does not exist")
                datadir.append_data_dir(str(args.proj_grid_dir.resolve()))
            spec=GeoSceneSpec.model_validate_json(args.spec.read_text())
            if args.source=="local":
                if args.raster is None: raise ValueError("--raster is required for a local source")
                if args.vertical_verified and not args.source_vertical_crs:
                    raise ValueError("--source-vertical-crs is required with --vertical-verified")
                provider=LocalGeoTIFFProvider(args.raster,dataset=args.dataset,
                    vertical_crs=args.source_vertical_crs,vertical_verified=args.vertical_verified,
                    license=args.license,attribution=args.attribution,
                    acquisition_dates=args.acquisition_dates,source_uri=args.source_uri,
                    zero_is_nodata=args.zero_is_nodata)
            else:
                provider=OpenTopographyProvider()
            asset=compile_terrain(spec,provider,args.output_dir,args.raw_cache,args.variant,
                                  args.raster_spacing_m,args.mesh_spacing_m,args.exploratory)
            result={"metadata":str(args.output_dir/"metadata.json"),
                    "height_alignment":asset.height_alignment,
                    "coverage_fraction":asset.coverage_fraction,
                    "mesh_vertices":asset.mesh_vertices,"mesh_triangles":asset.mesh_triangles}
        else:
            a=TerrainAsset.model_validate_json(args.a.read_text())
            b=TerrainAsset.model_validate_json(args.b.read_text())
            result=compare_assets(a,b)
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2))
        return 0 if args.command!="compare" or result["matched"] else 2
    except (ValueError,FileNotFoundError,ValidationError) as exc:
        print(json.dumps({"error":str(exc)}),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
