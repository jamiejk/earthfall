# Woodside address flyover test

**Current blocking rule:** every shot must have measured height for its target. The early terrain-only, textured-drape and flat-guide renders below are historical comparisons; the current renderer and H3 queue reject them as new blocking controls. The active Woodside control is the measured-building model described at the end of this document.

This second GeoBlock fixture tests an address-centred scene away from the Welsh coast. The address is [Woodside Library, 3140 Woodside Road, Woodside, CA 94062](https://smcl.org/locations/1W/). An [OpenStreetMap address point](https://www.openstreetmap.org/node/368169993) resolved through Nominatim is at 37.4291056, -122.2580225. It is a location marker, **not** a surveyed building footprint.

## Source selection

- The [USGS TNM Access API](https://apps.nationalmap.gov/tnmaccess/) returned two 1 m 3DEP products for the 2.4 km square. The 2025-published `CA_CaliforniaGaps_B23` tile has 100% valid coverage in the requested crop; `CA_SantaClaraCounty_2020_A20` has 0% there. The script selected by measured crop coverage, then publication date. The USGS catalogue identifies the 1 m DEM as a LiDAR-derived bare-earth model in NAD83 / UTM 10N metres with NAVD88 metre heights. The address-point ground height sampled from the selected crop is about 119.84 m NAVD88.
- [USGS NAIP Imagery](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer) supplied 660 m and 2.4 km aerial crops. The service's catalog `identify` result at the address names image `m_3712238_se_10_060_20220518`, acquisition date **18 May 2022**, nominal 0.6 m. [USGS describes CONUS NAIP imagery as public-domain](https://www.usgs.gov/national-map-supporting-themes). The aerial imagery is a place reference, not a height source.
- The terrain grid and aerial image come from different acquisitions; the clay mesh deliberately contains no building or tree surfaces. The gold pin and halo are schematic annotations.

The selected source URLs, candidate coverage audit, coordinates, licences and checksums are recorded in `data/woodside_library_3dep_1m.json`, `data/woodside_library_naip_manifest.json`, and `outputs/woodside_library_3dep/h3_textured_reference_manifest.json`. The local raster and video outputs are ignored build artifacts; rerun the commands below to recreate them.

## Historical flat-model commands (archived)

These commands document the earlier experiment and **will not run against the current renderer or H3 queue**: measured building heights are now required.

```bash
uv run python scripts/prepare_usgs_3dep_scene.py \
  --name 'Woodside Library flyover' \
  --address '3140 Woodside Road, Woodside, CA 94062' \
  --latitude 37.4291056 --longitude -122.2580225 \
  --radius-m 1200 --project-id woodside-library \
  --spec examples/woodside_library.json \
  --raster data/woodside_library_3dep_1m.tif \
  --location-source 'OpenStreetMap node 368169993 via Nominatim'

uv run geoblock compile --spec examples/woodside_library.json \
  --variant USGS_3DEP_1m --source local \
  --raster data/woodside_library_3dep_1m.tif \
  --dataset 'USGS 3DEP 1 m DEM; CA_CaliforniaGaps_B23' \
  --source-vertical-crs EPSG:5703 --vertical-verified \
  --license 'USGS public domain' \
  --attribution 'USGS 3D Elevation Program, The National Map' \
  --source-uri 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/CA_CaliforniaGaps_B23/TIFF/USGS_1M_10_x56y415_CA_CaliforniaGaps_B23.tif' \
  --raster-spacing-m 1 --mesh-spacing-m 5 \
  --output-dir outputs/woodside_library_3dep

uv run python scripts/fetch_usgs_naip_references.py \
  --spec examples/woodside_library.json --output-dir data --name woodside_library

blender -b --factory-startup --python scripts/render_address_flyover.py -- \
  --terrain outputs/woodside_library_3dep/terrain.glb \
  --height-m 119.841705 \
  --address '3140 Woodside Road, Woodside, CA 94062' \
  --blend outputs/woodside_library_3dep/flyover.blend \
  --frames-dir outputs/woodside_library_3dep/flyover_frames --render

blender -b --factory-startup --python scripts/render_address_flyover.py -- \
  --terrain outputs/woodside_library_3dep/terrain.glb \
  --texture data/woodside_library_naip_drape_4096.png \
  --height-m 119.841705 \
  --address '3140 Woodside Road, Woodside, CA 94062' \
  --blend outputs/woodside_library_3dep/flyover_textured.blend \
  --frames-dir outputs/woodside_library_3dep/textured_frames --render

ffmpeg -framerate 24 -start_number 1 \
  -i outputs/woodside_library_3dep/flyover_frames/frame_%04d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
  outputs/woodside_library_3dep/flyover_clean.mp4
ffmpeg -framerate 24 -start_number 1 \
  -i outputs/woodside_library_3dep/textured_frames/frame_%04d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
  outputs/woodside_library_3dep/woodside_library_textured_flyover.mp4
python scripts/create_geoblock_workflow.py --fixture woodside_library
python scripts/queue_woodside_h3.py --take textured
```

Both clay and textured flyovers are eight seconds, 1280 × 720, 24 fps. The textured version is a 2022 NAIP aerial drape on a 1 m source DEM compiled to a 5 m mesh. The near-address aerial patch retains its native detail in a 4096-pixel drape; the outer portion comes from the wider NAIP crop. This shows roads and vegetation in their aerial positions, but trees and buildings remain flat texture rather than surveyed 3D objects. The H3 pack retimes the selected control to 124 frames at 864 × 480 for a five-second test and pairs it with the two NAIP crops. The Comfy canvas is `examples/woodside_library_comfy.json`; its first four GeoBlock nodes now select Woodside by name and prepare this fixture. This is still a two-location prototype, not an arbitrary-address downloader inside ComfyUI.

## Test result

The first native H3 pass used the grey clay control plus the two NAIP stills. It preserved the camera motion but largely ignored the still-image appearance, leaving brown, mostly untextured terrain. A second pass used the aerial-draped control and the same stills. That output preserved the visible roads and wooded pattern across six sampled frames and removed the schematic pin; the sampled frames show a continuous arc without an obvious cut. It remains a stylized reconstruction: the model has not supplied verifiable 3D tree or building geometry, and current site conditions may differ from the May 2022 photograph.

- Grounded textured Blender video: `outputs/woodside_library_3dep/woodside_library_textured_flyover.mp4` (8 s, 1280 × 720, 24 fps).
- MiniMax H3 textured test: `/run/media/jamie/Data/Image Creation/Comfy/output/video/WoodsideLibrary_H3_textured_00001_.mp4` (5.17 s, 864 × 480, 24 fps). The local GPU render completed successfully with prompt ID `783c9117-7a28-47ba-8ff2-4b85bd0284f2`.
- The CPU-only Comfy preparation flow on port 8189 completed successfully with prompt ID `8ec30100-658e-4e07-a14f-aa333c2bcef5`, producing a scene preview, two aerial images, motion reference, prompt and frame count. The full H3 graph was run separately on a GPU-enabled Comfy process, which has been stopped after completion.

For the next test, the main improvement is to add separate 3D features (building footprints/heights and trees) on top of the DEM rather than asking H3 to infer them from a flat aerial drape.

## Clear-line style experiment, 4 October 2026

Two original clear-line Woodside illustrations were supplied in addition to the two NAIP place images. The first H3 run kept the aerial-draped video as the motion reference and explicitly described it as planning geometry, not the final visual style. It completed successfully (prompt `b53d223b-fe6a-49ed-b5b3-c1941d5d93e1`) at 124 frames, 864 × 480, 24 fps. The output is `outputs/woodside_library_3dep/woodside_h3_clearline_v1.mp4`. It removed the gold pin and preserved the camera path and recognizable scene, but retained the appearance of aerial texture on a model. The style images influenced the palette more than the linework.

An exploratory second H3 pass used a posterized and edge-outlined version of the first result as its video guide and placed the style images first in reference order. It also completed (prompt `60c02676-3008-415b-84a3-c1ca7d2c5881`), saved as `outputs/woodside_library_3dep/woodside_h3_clearline_v2_illustrated.mp4`. This produced flatter colour but lost road/building detail and did not achieve the crisp, consistent ink contours of the style images. Both results have 124 frames and last 5.17 seconds. The second pass was an experiment recorded in `outputs/woodside_library_3dep/h3_clearline_illustrated_manifest.json`; the current H3 queue no longer offers either flat control.

The tested outcome is that prompt instructions and two style stills did **not** override the dense visual content of the reference video. Before another costly H3 pass, the control should be redesigned directly in Blender as a marker-free, simplified motion/geometry render rather than filtering a textured or generated video. Then compare that control with the aerial-draped one using the same place images, style images, shot settings and seed. Human review remains necessary for geographic and style fidelity.

## Marker-free Blender guide comparison

The next control was rendered directly in Blender, with no address marker, from the same 3DEP terrain, camera keys and a 32-colour/2048-pixel simplification of the NAIP drape. Flat workbench lighting removed shadows and cavity shading. Its archived commands follow; the current height gate rejects them:

```bash
python scripts/make_flat_map_texture.py \
  --input data/woodside_library_naip_drape_4096.png \
  --output data/woodside_library_flat_guide_2048.png
blender -b --factory-startup --python scripts/render_address_flyover.py -- \
  --terrain outputs/woodside_library_3dep/terrain.glb \
  --height-m 119.841705 \
  --address '3140 Woodside Road, Woodside, CA 94062' \
  --blend outputs/woodside_library_3dep/flat_guide.blend \
  --frames-dir outputs/woodside_library_3dep/flat_guide_frames \
  --no-marker --flat-guide --render
python scripts/queue_woodside_h3.py --take flatguide --style clear-line
```

The first controlled H3 pass changed only the video reference and output filename relative to the earlier aerial-draped clear-line graph; prompt, four stills, seed, sampler, 124 frames and 864 × 480 dimensions stayed fixed. It completed with prompt `0304c6cd-304a-43e4-8e42-2606a7384d65`. The result (`outputs/woodside_library_3dep/woodside_h3_clearline_flatguide_with_pin.mp4`) shows clearer illustrated trees and roofs, but H3 invented a bright map pin even though the guide video contained none. The old prompt's negative mention of a "gold locator pin" is a plausible trigger, not a proven cause.

A second flat-guide H3 pass kept the same control and stills but replaced the prompt with positive instructions that never mention a pin. It completed with prompt `90fd180e-286c-462e-b100-c84ba7167d9c`; `outputs/woodside_library_3dep/woodside_h3_clearline_flatguide_pinfree.mp4` has no visible pin in six sampled frames. The appearance is less stable: early frames retain more illustrated form, while later frames flatten into map-like colour patches. The old `flatguide` take and legacy prompt are no longer available in the queue.

This experiment establishes that simplifying the motion reference can improve the style, but a flat terrain drape still does not provide 3D trees or buildings. H3 may also follow negative prompt nouns literally. The next engineering step is a genuinely schematic Blender scene with separate proxy trees/buildings and roads, or a dedicated video stylisation stage with temporal consistency checks. The current H3 result is a useful test, not a finished clear-line scene.

## Measured-height close shot

All building shots now require a structure-height source. The 3DEP 1 m DEM remains the ground model, but it cannot qualify on its own. The [USGS LPC point cloud](https://registry.opendata.aws/usgs-lidar/) for `CA_CaliforniaGaps_3_B23` supplies above-ground returns through its public EPT mirror. [OpenStreetMap](https://www.openstreetmap.org/copyright) supplies building footprints, not measured heights. The loader checks classified ground returns against the DEM before estimating each roof height from elevated returns within an inset footprint. Because this LPC tile groups roofs and vegetation in class 1, it requires a dense and spatially coherent roof surface; uncertain buildings are omitted. The output gives approximate flat-roof blocks, not surveyed roof shapes.

For a 200 m neighborhood, 52 of 66 OSM building footprints qualified. The building footprint containing the address (`way 382886306`) has a 4.35 m estimated block height above the DEM. Sampled LPC ground points differed from the 3DEP DEM by a median -0.002 m. Fourteen buildings failed the roof checks and are absent from the block scene. These results and every rejection reason are in `outputs/woodside_library_3dep/measured_buildings.json`.

```bash
uv run --extra buildings python scripts/prepare_measured_buildings.py --radius-m 200
blender -b --factory-startup --python scripts/render_address_flyover.py -- \
  --terrain outputs/woodside_library_3dep/terrain.glb \
  --height-m 119.841705 \
  --address '3140 Woodside Road, Woodside, CA 94062' \
  --buildings outputs/woodside_library_3dep/measured_buildings.json \
  --close-shot --no-marker \
  --blend outputs/woodside_library_3dep/blocking_measured.blend \
  --frames-dir outputs/woodside_library_3dep/blocking_measured_frames --render
ffmpeg -framerate 24 -start_number 1 \
  -i outputs/woodside_library_3dep/blocking_measured_frames/frame_%04d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
  outputs/woodside_library_3dep/woodside_library_measured_blocking_flyover.mp4
```

The Comfy preparation workflow is `examples/woodside_library_comfy.json`. Its scene node always checks measured height, and its Woodside Plan Shot offers only **Close measured-building orbit**. The route rejects a missing or unaligned height source, missing target roof evidence, or less than 70% usable nearby footprints. The 8-second Blender guide retimes to a 124-frame H3 input.

Two earlier H3 clear-line tests used a measured-height guide with an aerial ground drape and four image slots. The first retained the old wide NAIP images. It generated a broad school-dominated panorama and lost the intended close framing (`outputs/woodside_library_3dep/woodside_h3_clearline_close_measured_v1.mp4`, prompt `7a9a15f1-c790-4f13-b098-3fba68c4aacb`). For the second, `scripts/make_woodside_close_crop.py` derived a checked 220 m NAIP crop from the existing 660 m image and paired it with that guide's opening frame. This produced distinct illustrated roofs, facades and parallax in the nearby buildings (`outputs/woodside_library_3dep/woodside_h3_clearline_close_measured_tight.mp4`, prompt `5f66764f-4dfd-4dca-b297-89ddc777fd98`). The second shot still carries mottled aerial texture on the ground and drifts to a higher view toward the end. These are historical comparisons, not outputs of the current undraped blocking guide.

```bash
uv run python scripts/make_woodside_close_crop.py
python scripts/queue_woodside_h3.py --take closemeasuredtight --style clear-line \
  --submit --url http://127.0.0.1:8190
```

The crop source, bounds, opening still, checksums and image roles are recorded in `data/woodside_library_close_reference_manifest.json`. The direct queue script and Comfy preparation node both use these close references. H3 was run on the project's GPU-enabled ComfyUI v0.38.2 installation; inference used 12 steps and produced 124 frames at 864 × 480, 24 fps. The temporary GPU process was stopped after these takes.

The blocking policy was then tightened for **every** scene. The Woodside Blender renderer now requires the measured-building manifest even for wide camera routes, and it refuses a route wider than the height dataset's measured coverage. The current blocking guide omits the aerial drape entirely so photographed roofs cannot masquerade as geometry. Worm's Head qualifies as a natural-landform scene only after a sampled landmark-height and local-relief check; its relative promontory relief is measured even though its absolute vertical datum remains exploratory. Historical flat-model videos and H3 results remain for comparison, but new queue commands cannot use them. The undraped guide has not yet been sent through H3.
