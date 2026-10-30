# Measured tree proxies in the blocking

*6–7 October 2026. Follows [tree-building-proxies-scope.md](tree-building-proxies-scope.md).*

Trees on the South Rim were invented by the keyframe painter and LTX, so they swelled and slid between
frames. `--trees` stands a low-poly proxy on every measured tree near the camera path, so the layout video
carries real tree positions and sizes for LTX to follow.

```
geoblock-run --name "Mather Point, Grand Canyon" --latitude 36.0617 --longitude -112.1078 \
  --preset "Rim reveal" --heading 5 --palette grand-canyon --trees ...
```

It works for the Earth zoom too (`--preset "Earth zoom" --trees`): trees cover the landing, kept clear of
the low camera path and of the stand-in. In the Earth zoom the trees fade in as the camera comes down
(none above 1000 m over the ground, all below 450 m), because from higher up the few-km tree area reads as
one dark patch on the imagery.

`geoblock-run --trees` measures the trees in its own environment first (`uv run --extra comfy --extra buildings`)
and caches them where the ComfyUI node looks, so ComfyUI itself needs no lidar packages; use
`--no-tree-prefetch` to let the node fetch instead. `--trees` (or `--trees auto`) uses USGS 3DEP lidar where a survey covers the place, else the Meta/WRI
canopy map; `--trees lidar` / `--trees meta` force one. The Render Blocking Video node has the same choice
as its `trees` input. Needs `laspy[lazrs]` and `shapely` (`uv sync --extra buildings`) for lidar.

## How it works

1. **Area.** `trees.near_field_bounds`: the frames within 300 m of the ground (or of the path's lowest
   point), plus 1 km and where they look, capped at 3.5 km a side around the lowest camera position. From
   higher up a tree is a pixel or two wide.
2. **Canopy height model, 1 m.** `lidar.project_for` finds the survey in hobu's boundary index of the
   public EPT bucket; `lidar.read_points` streams only the octree nodes in the box. Ground (class 2) is
   gridded and gap-filled; every vegetation point (classes 3–5, or class 1 when the survey doesn't split
   vegetation) is measured against the ground under it. Cells on ground steeper than 35° are dropped,
   because canyon surveys leave cliff rock in class 1 and it reads as canopy. Heights over 45 m are dropped.
3. **Tree tops.** Local maxima of the lightly smoothed CHM, at least 2 m tall; tops inside a taller
   tree's crown are merged into it. Crown radius is how far the canopy stays above half the tree's height.
4. **Proxies.** In Blender each tree is dropped onto the terrain mesh as rendered (ray cast, not the lidar
   ground, so nothing floats where the 10–20 m mesh misses the rim). Under 12 m: a rounded
   pinyon/juniper crown down to the ground. 12 m and over: an open, round-topped ponderosa crown high on
   a trunk (a clean cone was copied literally by klein as a Christmas-tree plantation). Every crown is
   lumped and leaned at random (±25%) so no two match. All trees are one mesh, dark green, with the same
   distance haze as the terrain. The depth pass includes them. Trees thin out over the outer 500 m of the
   measured area (at most a fifth of its shorter side), so it doesn't show as a hard rectangle from the air.

Cached per place and area in `<scene>/trees/trees-<key>.json` (local east/north metres, height, crown
radius, source and attribution), and the lidar tiles in `data/cache/raw/ept/`.

## Which canopy source (checked at Mather Point)

NAIP aerial photos were not reachable from the cloud container (the USGS image server, Planetary Computer
and the requester-pays NAIP buckets are all blocked), so two other checks were used:

- **The lidar's own intensity image.** Trees show as dark crowns in it. The lidar canopy crowns line up
  with them one for one; the Meta map is a blur that keeps only the larger crowns
  (`/mnt/project-files/trees/canopy_check_rim.jpg`, `canopy_check_plateau.jpg`).
- **Sentinel-2 NDVI** (5 June 2024, 0% cloud, 10 m), an independent sensor. Correlation between NDVI and
  canopy cover per 10 m cell, on ground under 30°:

| Area | Lidar | Meta |
|---|---|---|
| Rim, 300 m box | 0.62 | 0.59 |
| Plateau south of Mather Point, 400 m box | 0.33 | 0.16 |

Lidar cover on flat ground is 34–49% against Meta's 14–32%. Lidar is the better source where it exists;
Meta is the global fallback and will under-draw sparse woodland. The lidar is from 2019.

Holes in a survey (a missing source tile left a 0.4 × 0.9 km rectangle with no points just north of Mather
Point) are filled from the Meta map with `--trees auto`; Meta draws fewer trees, so a filled hole stays a
little sparser.

Known limits: buildings in class-1-only surveys can become "trees" (few near Mather Point); dead trees
count as trees; small trees on cliffs are dropped with the rock.

## Cloud test renders (CPU, software EEVEE, previews only)

- `/mnt/project-files/trees/blocking_ab_cloud_cpu.jpg`: rim reveal at heading 5°, one frame a second,
  without (left) and with (right) trees. 172 k trees in the near field; the plateau the camera skims is
  dense ponderosa per the lidar, then the rim, then open canyon where nothing changes.
- `/mnt/project-files/trees/earthzoom_landing_ab_cloud_cpu.jpg`: the low part of the Earth zoom (frames
  56–242), without and with trees. 81.5 k trees; one tree was removed for standing in the camera's way.
  The globe and HLS imagery were flat stand-ins because GIBS is blocked from the cloud container, and the
  dark frames are an artefact of rendering only every twelfth frame.

The camera clears the tallest treetop under the rim reveal path by 19.6 m.

## Cost

Mather Point rim reveal (2.8 × 3.5 km near field): about 600 MB of LAZ fetched once, about 5 min from the
cloud container, under 2 GB of memory (the points stream node by node); 172 k trees, about 6 M vertices,
built into one mesh in a few seconds. GPU render time with trees is not measured yet.
