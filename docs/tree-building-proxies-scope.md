# Tree and building proxies in the blocking: scope

Checked 6 October 2026 from the cloud container, at Mather Point (36.0617, -112.1077). Scoping only; nothing here is wired in yet.

## Why

The blocking has bare 10 m terrain (`USGS3DEPProvider`, 1/3 arc-second). Trees on the South Rim are invented by the keyframe painter and LTX, so they change size and slide between frames. A tree silhouette in the layout video gives LTX something to hold, the same way the terrain does.

## Data, as tested

| Source | Mather Point result | Licence | Headless access | Coverage |
| --- | --- | --- | --- | --- |
| 3DEP LiDAR point cloud, EPT `AZ_GrandCanyonNP_1_2019` | Covers Mather Point. ~20 points/m², ground classified (class 2). Vegetation, buildings **and cliff rock** are all class 1. | US public domain | Plain HTTPS range reads from `s3-us-west-2.amazonaws.com/usgs-lidar-public/` with `laspy`+`lazrs`. Same code shape as `scripts/prepare_measured_buildings.py`. Works from the cloud container (tnmaccess does not). | US only; survey dates and classes vary by project. |
| Meta/WRI global canopy height, 1.2 m (`dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/{quadkey9}.tif`) | Tile `023102000` covers it. Trees 4–7 m (p50/p90), max 15 m. | CC BY 4.0 | COG, windowed read with `rasterio` `/vsicurl/`. Reachable from the cloud. | Global. Model-derived from 2009–2020 satellite imagery. |
| Overture buildings, release `2026-09-23.1` | 86 footprints within ~1 km of Mather Point (28 with height), 813 in Grand Canyon Village (332 with height). No roof shapes. Sources: OSM, Microsoft ML, Esri Community Maps. | ODbL (OSM-derived rows) / CDLA Permissive for ML rows; carry attribution | GeoParquet on S3, read with `pyarrow.dataset` and a bbox filter (duckdb's spatial extension download is blocked from the cloud container, pyarrow works). | Global footprints; heights patchy. |
| OSM via Overpass | Not testable: blocked by the cloud proxy. Overture already contains the OSM footprints. | ODbL | Would work on the desktop. | Global |

### The canopy sources disagree, and lidar has a canyon-specific trap

On a 300 m box around Mather Point, on ground under 25° slope only:

- Lidar canopy height model (points normalised against a ground TIN): 38% of cells over 2 m, median tree 5.2 m.
- Meta canopy model: 14.5% over 2 m, median 4.0 m.
- Overlap of the two canopy masks (IoU): 0.29.

On steep ground the lidar reads 65% "canopy", which is cliff rock the ground classifier left as class 1. So a lidar canopy model on canyon terrain must be masked by ground slope (and ideally by multiple returns) before use. I have not checked either source against aerial photos yet, so I can't say which cover figure is right. Pinyon-juniper on the rim being sparser than 38% is plausible, and Meta is known to under-report sparse woodland, so the truth is likely between them. The first build step includes that check against NAIP.

## How it becomes Blender proxies

1. `geoblock_core/canopy.py`: for the scene bbox, build a 1 m canopy height model (lidar EPT where a 3DEP project covers the site, else Meta), mask steep slopes, find tree tops as local maxima with a height-scaled window, and size each crown from its watershed segment. Output `trees.json` (x, y, height, crown radius, source) cached per place, with attribution.
2. In `blender_render.py` and `blender_earth_zoom.py`: drop each tree onto the **imported 10 m mesh** by ray-casting down (not onto lidar ground, or trees float or sink where the 10 m mesh misses the rim), and instance one low-poly proxy (cone over short trunk, or a squashed ellipsoid for juniper) scaled per tree. A dark green material in each palette so the painter reads them as trees.
3. Only within a near-field radius of the camera path (start at ~1.5 km). Beyond that, a tree is under a pixel at 1920 wide; paint far canopy into the terrain colour as a darker tint from the same raster.
4. Buildings: Overture footprints in the bbox, height from Overture where present, else measured from lidar returns inside the footprint (port of the Woodside logic, including its "reject if uncertain" rule), extruded as flat-roofed blocks in a neutral grey.

Everything is driven by lat/lon and radius; no per-shot steps.

## Cost

- Data, once per place: a 1 km radius of lidar is ~60 M points, several hundred MB of LAZ, a few minutes to fetch and process. Meta tiles are tens of MB.
- Blender: about 1,500–5,000 instanced trees for a 1.5 km near field (469 crowns in the 300 m test box). Instances are cheap in EEVEE; expect seconds per clip, not minutes. Not measured.
- ComfyUI/LTX: no change to the graph or VRAM.
- The real cost is risk, not time. Proxies might be painted as clay or bushes (the depth ControlNet strength 1.0 test gave clay looks), or klein might crowd them with extra trees. Needs one A/B render to know.

## What to build first

1. Trees only, on the rim reveal: `canopy.py` + proxies + `--trees` flag, unit tests on a synthetic canopy raster, and a NAIP overlay check that the tree tops land on real trees (this also settles lidar vs Meta).
2. One A/B blocking clip with and without trees, then a klein + LTX 2.5 render of each, queued by the render thread when the GPU is free.
3. Only if trees hold in motion: buildings, then Earth zoom near-field trees around the hiker.

Buildings come second because the Grand Canyon and Earth zoom shots barely show any (the visitor centre is ~500 m back from the rim); they matter for town locations.

A side finding: the same EPT ground points give a 1 m bare-earth surface from the cloud, which could later replace the 10 m mesh near the rim edge. Out of scope here.
