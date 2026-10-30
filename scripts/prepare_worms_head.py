"""Fetch only the published COG windows and pinned PROJ grids for the example."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import httpx
import rasterio

from geoblock_core.models import GeoSceneSpec
from geoblock_core.providers import crop_raster, sha256


SOURCES = {
    "gedtm30_v20250619": {
        "url": "https://s3.opengeohub.org/global/edtm/filtered.dtm_edtm_m_30m_s_20060101_20151231_go_epsg.4326.3855_v20250619.tif",
        "filename": "worms_head_gedtm30_v20250619_subset.tif",
        "margin_m": 120,
    },
    "welsh_lidar": {
        "url": "https://dmwproductionblob.blob.core.windows.net/cogs/lidar/wales_dtm_32bit_cog.tif",
        "filename": "worms_head_lidar_subset.tif",
        "margin_m": 90,
    },
}
GRIDS = {
    "uk_os_OSGM15_GB.tif": "bf0ac81f1c0c33332693d7d606f335b793c3c2995b8e69eeb1350bd94adcde3b",
    "uk_os_OSTN15_NTv2_OSGBtoETRS.tif": "5d6ed64d2119952c4c559fa1fccbc594b6520fc3ec3ef2fc10be13202c4384fa",
    "us_nga_egm08_25.tif": "4191d471eefebf24091b56dbc604353cb3b8cf8cc70e448bb9ae56a272bef17a",
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    spec = GeoSceneSpec.model_validate_json((root/"examples/worms_head.json").read_text())
    data_dir = root/"data"
    data_dir.mkdir(exist_ok=True)
    for name,info in SOURCES.items():
        path=data_dir/info["filename"]
        if not path.is_file():
            crop_raster("/vsicurl/"+info["url"],spec,path,margin_m=info["margin_m"])
        with rasterio.open(path) as ds:
            record={"source_uri":info["url"],"subset_sha256":sha256(path),
                    "crs":ds.crs.to_string(),"bounds":list(ds.bounds),
                    "pixel_spacing":list(ds.res),"scale":ds.scales[0],
                    "offset":ds.offsets[0],"nodata":ds.nodata,
                    "vertical_datum_tag":ds.tags().get("VERTICAL_DATUM")}
        path.with_suffix(".json").write_text(json.dumps(record,indent=2)+"\n")
        print(name,path,record["subset_sha256"])
    grid_dir=data_dir/"proj"
    grid_dir.mkdir(exist_ok=True)
    with httpx.Client(timeout=120,follow_redirects=True) as client:
        for name,expected in GRIDS.items():
            path=grid_dir/name
            if path.is_file() and sha256(path)==expected:
                continue
            temp=path.with_suffix(".part")
            digest=hashlib.sha256()
            try:
                with client.stream("GET","https://cdn.proj.org/"+name) as response:
                    response.raise_for_status()
                    with temp.open("wb") as fh:
                        for chunk in response.iter_bytes():
                            fh.write(chunk)
                            digest.update(chunk)
                if digest.hexdigest()!=expected:
                    raise ValueError(f"PROJ grid checksum mismatch: {name}")
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
            print("grid",name,expected)
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (ValueError, httpx.HTTPError, rasterio.errors.RasterioError) as exc:
        print(str(exc),file=sys.stderr)
        raise SystemExit(2)
