# City buildings in the Earth zoom

*7 October 2026. Follows [tree-building-proxies-scope.md](tree-building-proxies-scope.md).*

The Earth zoom can now land on a city. `--buildings` stands a flat-roofed block on every building footprint
within 7 km of the landing, at its measured height, so the keyframe painter and LTX keep the city's real
street grid and skyline instead of inventing one.

```
uv run --extra comfy --extra buildings geoblock-run --place san-francisco-twin-peaks --preset "Earth zoom" \
  --duration 10 --ltx --keyframes flux-klein --ltx-version 2.5 --landing-lidar auto
```

`--buildings` defaults to `auto`: on for catalog places measured as cities (their catalog entry has
`coverage.city`), off elsewhere. `--buildings` alone forces it on; `--buildings off` turns it off. It only
applies to the Earth zoom. The Render Blocking Video node has the same switch as its `buildings` input.

## The first city: Twin Peaks, San Francisco

`san-francisco-twin-peaks` stands at the Christmas Tree Point lookout (moved 22 m by the catalog's sightline
check) and looks down Market Street to the Financial District, heading 47°. Picked over the other cities
checked because of its building heights and its ground:

| City, 5 km square around the view | Overture buildings | With a height | Over 100 m |
|---|---|---|---|
| San Francisco (Twin Peaks to downtown) | 33,585 | 96% | 80 |
| New York (Midtown) | 17,507 | 99% | 483 |
| Chicago (Loop) | 8,739 | 82% | 247 |
| Seattle (Kerry Park to downtown) | 14,262 | 75% | 54 |
| Rio de Janeiro (Sugarloaf to Botafogo) | 21,377 | 44% | 5 |
| Hong Kong (Victoria Peak to Central) | 7,552 | 3% (43% with floor counts) | 88 |

New York has the best heights, but it is flat and its classic views are from rooftops, which the stand-in
can't stand on yet. San Francisco has a hilltop overlook the Earth zoom already handles, 3DEP 10 m terrain
plus a 1 m lidar landing patch, NAIP, hills and the bay for the descent, and a skyline in the view.

Within 7 km of the landing: 175,371 buildings (163,862 measured heights, 865 from floor counts, 10,644 at the
7 m default), median 7 m, tallest 326 m (Salesforce Tower), 192 over 100 m.

## How it works

1. **Read.** `buildings.fetch_buildings` reads Overture's `buildings/building` and `building_part` GeoParquet
   straight from the public S3 bucket over HTTP range requests (pyarrow), only the row groups whose
   bounding boxes overlap the area. A row-group index per release is cached in
   `data/cache/raw/overture/<release>/`. San Francisco takes about 10 s once the index exists.
2. **Footprints.** Outer rings only, projected to the scene's UTM CRS, simplified to 0.5 m, under 15 m²
   dropped. A building with parts (a tower on a podium) is drawn as its parts, each with its own height and
   `min_height`.
3. **Heights.** Overture `height`, else `num_floors` × 3.2 m, else 7 m. Roof shapes are not modelled.
4. **Blocks.** In Blender each block stands on the terrain mesh as rendered (ray cast at every corner): walls
   from a metre under the lowest corner, roof at the measured height over the median ground. Blocks that
   would reach within 15 m of the camera below 1 km, or the stand-in, are left out. One mesh, about 3 M
   vertices for San Francisco.
5. **Look.** Roofs wear the same draped HLS/NAIP imagery as the ground, so from high up the city reads the
   same with or without blocks; walls are pale concrete with a quarter of the imagery colour. Hidden above
   20 km.
6. **Prompts.** The catalog measures the city (`coverage.city`), and the place's look gains a sentence from
   it: "a dense city of real buildings, mostly 2–4 storey houses and apartment blocks packed along a street
   grid, rising to a cluster of downtown skyscrapers up to 326 m tall". The keyframe prompts also tell klein
   that each pale block is one real building to paint at that footprint and height; the LTX prompt says the
   buildings keep their places and heights instead (it has no Image 1).

Cached per scene in `<scene>/buildings/buildings-<key>.json` and `.npz`. `geoblock-run` prefetches them in
its own environment (`--extra buildings`: pyarrow and shapely), so ComfyUI needs neither.

Attribution (added to the render manifest): Overture Maps Foundation buildings, release 2026-09-23.1:
© OpenStreetMap contributors (ODbL), Microsoft and Esri Community Maps contributors (CDLA Permissive 2.0).
ODbL needs that credit wherever the video is shown.

## Known limits

- Blocking time: the first GPU run spent 31 min in Blender, almost all of it in frames 21–55 (about 65 s a
  frame). The buildings are not the cause: `scripts/time_building_render.py` on frames 30–32 took 276 s as
  is, 276 s without block shadows, 279 s with triangulated roofs and 266 s with no buildings at all (each
  including scene setup). The slow stretch is the mid-altitude part of every Earth zoom (cloud deck sheets
  or the regional terrain are the suspects), not the city.
- Flat roofs, no roof shapes, no bridges. The Golden Gate and Bay bridges are not buildings in Overture, so
  the painter has to invent them from the imagery.
- Overture heights are mostly OpenStreetMap's. In San Francisco those came from a city lidar import, so they
  are good; elsewhere they can be patchy (Hong Kong, Rio above).
- Other catalog places marked "city" (Victoria Peak, Sugarloaf) are measured only on the next full catalog
  build; until then `auto` leaves their buildings off.
- `--trees` in a city would count roofs as trees where the lidar leaves buildings unclassified; not tried.
