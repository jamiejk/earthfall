# GeoBlock SceneCompiler

For Touchdown Earth-to-landmark flights, use the [canonical photographic-grounded-v1 workflow](docs/touchdown-canonical-workflow.md), accepted against the complete Bamburgh run on 27 October 2026. Create is integrated with that generation chain through `touchdown.canonical_build`, with automatic Qwen layout/review steps and preserved evidence. The specification distinguishes the accepted assistant-reviewed control from the additional app automation layer. The GeoBlock measured-terrain workflow below is a separate path.

**Real places, real camera moves.** Pick a landmark, and GeoBlock fetches measured terrain around it, flies a drone-style camera preset through that terrain, renders the move in headless Blender, and hands the blocking video to MiniMax H3 in ComfyUI. The geometry of the shot comes from survey data rather than the video model's guess. Everything runs as one ComfyUI graph, driven through the [Comfy SDK](https://docs.comfy.org/development/api-development/sdks).

![The GeoBlock graph in ComfyUI](docs/grand_canyon_canvas.png)

*Rim-reveal preset over USGS 3DEP terrain at Mather Point, Grand Canyon (low-resolution test render):*

![Rim reveal blocking frames](docs/grand_canyon_rim_reveal_frames.png)

**Current status:** Natural landforms in the conterminous US work end to end up to the blocking video: place → USGS 3DEP 1/3 arc-second terrain (NAVD88) → measured-relief check → camera preset with terrain clearance → Blender render. The H3 stage has produced a first Grand Canyon rim reveal on a local RTX 4090 without place photos; see [the first-run review](docs/grand-canyon-h3-first-run.md) for what holds and what does not. Buildings are out of scope until they pass the project's shape gate. See [the competition plan](docs/COMPETITION_PLAN.md) for what comes next.

## Quickstart: any US landform

1. Install GeoBlock: `uv sync --extra comfy` (Python 3.12).
2. Link `comfy_geoblock_prototype/` into ComfyUI's `custom_nodes/`, and install its `requirements.txt` into ComfyUI's Python. Install [Blender](https://www.blender.org/download/) 4.2+ and put it on `PATH`, or set `GEOBLOCK_BLENDER` to its executable.
3. Start ComfyUI, then put the Comfy API v2 in front of it: `comfy-api-proxy --comfyui http://127.0.0.1:8188 --port 8189`.
4. Render a blocking shot through the Comfy SDK:

```bash
export COMFY_BASE_URL=http://127.0.0.1:8189   # unset it and set COMFY_API_KEY for Comfy Cloud
uv run geoblock-run --name "Mather Point, Grand Canyon" --latitude 36.0617 --longitude -112.1078 \
  --preset "Rim reveal" --duration 10
```

Or search by name instead of coordinates: `--search "Half Dome" --result-number 1`. Add `--h3 --photo view1.jpg --photo view2.jpg` to generate the final video with MiniMax H3 Ref2VA, uploading your licensed place photos as SDK assets. Or let GeoBlock find them: `--auto-photos 3` fetches openly licensed Wikimedia Commons photos near the coordinates (CC0, CC BY, CC BY-SA or public domain; one per author; `--heading` favours views titled in that direction) and writes their attribution to `outputs/place_photos/`. Look at the picks and drop any with `--exclude-photo <page id>`. On the canvas, the **Place Photos (Wikimedia Commons)** node does the same. Outputs (control video, route map, opening frame and, with `--h3`, the generated video) land in `outputs/runs/<job id>/`.

Or pick one of 45 measured landing places worldwide: `--list-places` shows them, `--place-info gornergrat` shows one place's terrain, lidar, imagery and licences, and `--place gornergrat` builds the shot there, facing its named view. Terrain outside the conterminous US comes from Copernicus GLO-30. See [the place catalog](docs/place-catalog.md).

To work interactively, drop [`examples/grand_canyon_blocking_api.json`](examples/grand_canyon_blocking_api.json) or [`examples/grand_canyon_h3_api.json`](examples/grand_canyon_h3_api.json) onto the ComfyUI canvas. These are the same graphs `geoblock-run` submits.

| Preset | Move |
| --- | --- |
| Rim reveal | Skim low toward the edge, then fly out over the drop as the view opens |
| Orbit | Circle the landmark while keeping it centred |
| Flyover | Pass straight over the landmark, looking ahead and down |
| Push-in | Approach the landmark from out over the low ground |
| Pull-back rise | Start close on the landmark and pull away while climbing |

Routes aim toward the deepest measured drop unless you give a heading, lift smoothly wherever they would come within the minimum clearance of the ground, and use MiniMax H3 frame lengths (17n + 5 at 24 fps, at most 15 s per clip). Longer routes report overlapping segments for clip chaining.

## Running in the cloud

**RunPod (whole graph on one GPU pod).** Start a pod with a network volume of at least 80 GB mounted at `/workspace`, set a random `GEOBLOCK_PROXY_TOKEN` in the pod's environment, expose HTTP port 8189, and run [`scripts/runpod/bootstrap.sh`](scripts/runpod/bootstrap.sh) on it. It installs ComfyUI at a tested commit, the GeoBlock nodes, Blender as the `bpy` module and the pinned H3 Ref2VA weights, then starts a token-protected `comfy-api-proxy`. ComfyUI itself stays on the pod's loopback; reach its UI through an SSH tunnel. Then, from your machine:

```bash
export GEOBLOCK_PROXY_TOKEN=<same token>
uv run geoblock-run --blocking-url https://<pod-id>-8189.proxy.runpod.net \
  --name "Mather Point, Grand Canyon" --latitude 36.0617 --longitude -112.1078 \
  --engine cycles --h3 --photo view1.jpg
```

Use `--engine cycles` on pods: Cycles renders on the GPU through CUDA/OptiX, while EEVEE and Workbench need an OpenGL/EGL context that pod containers may not provide.

**Comfy Cloud (H3 stage only).** Comfy Cloud runs its own node set, so the GeoBlock nodes and Blender stay on a self-hosted server. `--h3-url cloud` splits the shot: the blocking video renders on `--blocking-url`, then the control video, opening frame and photos are uploaded to Comfy Cloud as SDK assets and only the H3 graph runs there, authenticated with `COMFY_API_KEY`. That key is never sent to a self-hosted target. If Comfy Cloud names the H3 model files differently, pass `--h3-unet`, `--h3-clip` and `--h3-vae`.

## Development fixtures

Worm's Head and Woodside Library were the first experiments. They remain for reproducibility and the author's own film work; the sections below document them. For the decisions and experiments that led here, read [the project story](docs/PROJECT_STORY.md). The [implementation plan](docs/GeoBlock_Implementation_Plan.md) is background design material. Generated videos, source rasters and model weights are not in this repository. See [landmark selection](docs/landmark-selection.md), [reproduction and asset notes](docs/REPRODUCING.md), and [pipeline status](docs/programmatic-flyover.md).

### Setup

Use Python 3.12. `uv sync --extra test` creates an isolated GIS environment; it does not modify ComfyUI or Blender. The resulting `uv.lock` pins tested Python dependencies. Blender is a separate executable and is not needed for the terrain milestone. Run `uv run python scripts/prepare_worms_head.py` to acquire only the two example COG windows and three checksum-verified PROJ grids in the ignored `data/` directory.

### CLI

Create a scene specification with a confirmed WGS84 coordinate:

```bash
uv run geoblock init --project-id worms-head --name "Worm's Head" \
  --longitude -4.3241 --latitude 51.5643 \
  --centre-longitude -4.31 --centre-latitude 51.565 \
  --location-source "OpenStreetMap way 1493808602; landmark reference" \
  --output examples/worms_head.json
uv run geoblock inspect --spec examples/worms_head.json
```

For a local DTM with a documented vertical CRS:

```bash
uv run geoblock compile --spec examples/worms_head.json --variant B_lidar \
  --source local --raster data/worms_head_lidar_subset.tif \
  --dataset "Welsh Government LiDAR DTM 2020–2023" \
  --source-vertical-crs EPSG:5701 --vertical-verified \
  --license OGL --attribution "Contains Welsh Government LiDAR data" \
  --raster-spacing-m 1 --mesh-spacing-m 5 --output-dir outputs/B_lidar
```

Only use `--vertical-verified` when the source's height datum has been confirmed outside the GeoTIFF; an EPSG:27700 horizontal CRS alone is insufficient. Without verified documentation, use `--exploratory` and omit `--vertical-verified`. Exploratory assets cannot pass the matched pair gate.

For GEDTM30, put the individual OpenTopography key in `OPENTOPOGRAPHY_API_KEY` in your local environment. No key enters a project spec, cache metadata or exported workflow:

```bash
uv run geoblock compile --spec examples/worms_head.json --variant A_global \
  --source gedtm30 --raster-spacing-m 30 --mesh-spacing-m 30 \
  --output-dir outputs/A_global
```

The EGM2008-to-ODN transformation requires PROJ grids. The compiler identifies any missing grid and stops rather than substituting an approximate vertical shift. Install the named grids in an isolated PROJ data directory or configure PROJ network access, then record their versions and hashes with the final run.

The included example uses a local window from the documented GEDTM30 v20250619 COG. Its Int32 samples are decimetres, so the 0.1 raster scale is applied. Exact-zero cells are conservatively marked unknown because the source leaves extensive ocean at zero; this may also remove true zero-height land. The newer v1.2 Float32 COG was excluded after its 0.1 header scale conflicted with its metre-valued samples at this location. See [the source check](docs/source-check.md).

```bash
uv run geoblock compile --spec examples/worms_head.json --variant A_global \
  --source local --raster data/worms_head_gedtm30_v20250619_subset.tif \
  --dataset "GEDTM30 v20250619 (publisher COG)" \
  --source-vertical-crs EPSG:3855 --vertical-verified --zero-is-nodata \
  --license "CC BY 4.0" --raster-spacing-m 30 --mesh-spacing-m 30 \
  --proj-grid-dir data/proj --output-dir outputs/A_global
```

```bash
uv run geoblock compare outputs/A_global/metadata.json outputs/B_lidar/metadata.json \
  --output outputs/pair_report.json
```

The project centre is a starting search point. Preview and landmark inspection must fix the final area and camera before any production pair. A large fraction of this 3 km coastal square is naturally ocean and remains nodata. Mesh triangles are emitted only where all three sampled nodes have measured elevation.

For an exploratory Blender scene of the 1 m western ridge crop with sea, run:

```bash
uv run python scripts/prepare_landform_height_evidence.py \
  --terrain-dir outputs/B_lidar_1m_zoom --spec examples/worms_head.json \
  --target-easting 239154.9012823891 --target-northing 187680.64672494138
blender -b --factory-startup --python scripts/create_blender_sea_scene.py
```

This writes `outputs/B_lidar_1m_zoom/sea_scene.blend` and `sea_preview.png`. The sea object's height is an editable staging value (default `0` in the LiDAR raster's unverified height frame); pass `-- --water-height 1.0` to render another level. It is not a measured tide or a certified ODN water level.

The 450 m crop only shows the western ridge. For the entire Worm's Head, its intertidal causeway area, and the Rhossili headland, use the wider 5 km scene spec in `examples/worms_head_full.json` and the 2 m mesh in `outputs/B_lidar_full_2m/`. Rebuild the Blender overview with:

```bash
uv run python scripts/prepare_landform_height_evidence.py
blender -b --factory-startup --python scripts/create_blender_sea_scene.py -- \
  --terrain outputs/B_lidar_full_2m/terrain.glb \
  --blend outputs/B_lidar_full_2m/sea_scene_full.blend \
  --preview outputs/B_lidar_full_2m/sea_preview_full.png \
  --view full --water-height -3.5
```

The low water setting is illustrative in the unverified LiDAR height frame. The DTM has gaps across the narrow tidal crossing, so a continuous low-tide path cannot be certified from this source alone.

For a perspective render looking west over Worm's Head from an elevated camera above the Rhossili headland edge, change the final command to `--view lookout` and write to `sea_scene_lookout.blend` and `sea_preview_lookout.png`. The lookout is deliberately above the measured ground to keep the foreground cliff from hiding the promontory.

The 15-second clay flyover starts from that lookout, passes the Outer Head over the sea, turns 180 degrees, and heads back toward Rhossili. The revised route uses a wider sea arc, keeps the promontory in view during the turn, and eases the return before the camera reaches the headland edge. Its camera and 360-frame render are generated with:

```bash
blender -b --factory-startup --python scripts/render_flyover_reference.py -- --render
ffmpeg -framerate 24 -start_number 1 \
  -i outputs/B_lidar_full_2m/flyover_v3_frames/frame_%04d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
  outputs/B_lidar_full_2m/flyover_reference_v3_h3.mp4
```

The MP4 is 1280 × 720 at 24 fps, without audio. It fits [MiniMax H3's reference-video limit of 2–15 seconds per clip](https://www.minimax.io/news/minimax-h3-open-source). The original `flyover_reference_h3.mp4` remains the first-take baseline, and `flyover_reference_v2_h3.mp4` preserves the wider turn with the faster return for comparison (`--route wide`). The route and sea level are visual references, not navigational or surveyed-tide data.

### ComfyUI GeoBlock prototype

The product-facing starting point is [the landmark-selection canvas](examples/landmark_selection_comfy.json): `Search Landmarks → Choose Landmark → Review`. The search text is supplied by the user, results are numbered and identified by OSM feature and coordinates, and the chosen place carries a requested coverage radius. It is not a fixed list of landmarks. Search does **not** claim that a 3D model exists there; the scene stage must validate source geometry before rendering. Regenerate the canvas with `python scripts/create_landmark_selection_workflow.py`. See [landmark selection](docs/landmark-selection.md) for the geocoder and coverage contract.

The [map-resource survey](docs/map-resource-survey.md) compares land DTMs, DSMs, bathymetry, imagery, and vectors, and proposes a source picker with coverage and vertical-datum checks. [This example source plan](examples/worms_head_source_plan.json) illustrates a multi-source Worm's Head scene; it is a design example, not an executable composite yet.

The working preparation graph is [examples/geoblock_prototype_comfy.json](examples/geoblock_prototype_comfy.json). It contains five GeoBlock nodes and preview outputs:

`Resolve Place → Terrain Scene → Plan Shot → Build References → Show Report`

The first node takes a place name or exact coordinates and a coverage radius. The terrain node attaches the corresponding prepared terrain scene and preview. The shot node selects a supported route and converts the requested 5–15 second duration to a valid H3 frame count. The reference node retimes the control MP4, checks the hashes of two licensed Commons photos or public-domain NAIP aerial crops, and emits ComfyUI `VIDEO`, `IMAGE`, `STRING` prompt, and `INT` frame-count outputs. Connect the video through `GetVideoComponents` to `MiniMaxH3ReferenceToVideo`'s `ref_videos.ref_video_0` input, and connect the image, prompt, and frame outputs to the corresponding H3 inputs. The example graph stops before H3 inference, so it can be queued cheaply to inspect the scene and references.

The fixture preparation graphs support the two prepared examples. Their legacy **Resolve Fixture** node remains for reproducible tests; it is not the product search. The **Discover Geometry Sources** node gives an alphabetical starting list of landform datasets, 3D city model datasets and catalogs, including the [Open 3D City Model Catalog](https://catalog.open3d.city/). Dataset listings are candidates, not a guarantee of complete buildings or a renderable shot. Source-to-scene selection and per-camera coverage checks remain to be implemented. The fixture scene node does not substitute terrain from another place. Worm's Head sea height and vertical alignment remain exploratory; camera routes are fixed rather than editable waypoints.

Every 3D blocking shot now requires measured height at its target. Worm's Head has measured terrain relief at the promontory; Woodside's experimental control adds lidar-height building volumes over its ground DEM. Flat terrain, aerial drapes and OSM footprints alone are ineligible as blocking controls. For an architecture-led demo, the project additionally requires validated roof/building geometry; Woodside's extrusions do not yet pass that shape requirement. The Woodside graph [examples/woodside_library_comfy.json](examples/woodside_library_comfy.json) uses the measured-height orbit for experimentation. See [docs/woodside-library-test.md](docs/woodside-library-test.md) for the source audit and reproducible Blender guide.

The [Woodside Library test](docs/woodside-library-test.md) adds a second prepared fixture at a public address in California. It selects a valid USGS 3DEP 1 m DEM tile, compiles a NAVD88 terrain scene, renders an eight-second clay address flyover, and pairs that motion with two USGS NAIP aerial crops. Open [the Woodside Comfy canvas](examples/woodside_library_comfy.json) to review the place → terrain → shot → references flow. The library marker is schematic; no building mesh is claimed. The node still needs a general source-discovery/download adapter before it can handle arbitrary addresses.

The Woodside clear-line experiment uses [two original style references](assets/style_references/woodside_clearline/manifest.json), available through `GeoBlockClearLineStyle` and a staged four-image H3 graph. New H3 jobs use `python scripts/queue_woodside_h3.py --take closemeasuredtight --style clear-line`, which rejects the older flat controls. The [test results](docs/woodside-library-test.md) retain those historical comparisons. The [programmatic pipeline status](docs/programmatic-flyover.md) separates automated stages from general-address support and visual QA.

To install in the newer ComfyUI checkout, link `comfy_geoblock_prototype/` into that checkout's `custom_nodes/` directory and restart it. The example canvas can be regenerated with `python scripts/create_geoblock_workflow.py`; its visual overview can be regenerated with `uv run python scripts/draw_geoblock_flow.py`. The local prototype and Woodside canvas are saved in that ComfyUI's `user/default/workflows/` directory. A CPU-only ComfyUI process can inspect reference preparation; H3 inference needs the GPU-enabled process.

### MiniMax H3 reference pack

The downloaded local Ref2VA weights are in the ignored `models/MiniMax-H3/` directory. To repeat the exact four-file ComfyUI download and check its hashes, run `bash scripts/download_h3_ref2va.sh`. The pinned [Comfy-Org checkpoint](https://huggingface.co/Comfy-Org/MiniMax-H3) consists of the pruned INT8 Ref2VA diffusion model (21.0 GB), the NVFP4-AWQ Qwen3-VL encoder (15.7 GB), and video/audio VAEs (5.8 GB together). The original [MiniMax BF16 Ref2VA checkpoint](https://huggingface.co/MiniMaxAI/MiniMax-H3) is 144.1 GB and does not fit this machine's 24 GB GPU. The INT8 ComfyUI path was validated with the Worm's Head and Woodside H3 renders described below.

The image collector mines public Wikimedia Commons files by category, text search, and geographic radius. It filters JPEGs by resolution and explicit reusable licence metadata, ranks candidates, and records the Commons page, author, licence, dimensions, coordinates, and SHA-256. Seven visually reviewed page IDs are pinned in `examples/worms_head_imagery.json`, so rerunning the pipeline selects the same views even if search ranking changes:

```bash
uv run python scripts/mine_commons_references.py --limit 7
uv run python scripts/validate_h3_pack.py
uv run python scripts/stage_h3_inputs.py
```

The ignored `data/reference_images/worms_head/` directory contains the seven JPEGs, `manifest.json` with the full candidate audit, `ATTRIBUTION.md`, and `h3_reference_pack.json` with image roles, the Blender motion reference, and the shot prompt. The pack is intended for H3's Reference to Video workflow: use the seven photos as place references and the MP4 for camera motion. [H3 accepts up to nine reference images and up to three reference clips with a 15-second total video-reference limit](https://huggingface.co/MiniMaxAI/MiniMax-H3). The photos show the causeway at different tide levels; their coordinates can identify the photographed subject rather than the camera, so neither their GPS nor a single tide state should override the measured terrain or the low-tide route design. Keep the generated output labelled as a stylized reconstruction and retain the Commons attribution when using the photos.

The H3 queue scripts also accept `--veda` for an optional [Veda sparse-attention node](https://github.com/veda-sparse/Veda-on-ComfyUI) in ComfyUI 0.38.0+. It preserves dense attention for geographic image/video references and saves separately named outputs. Our Ref2VA settings are outside the predictor's training configuration, so compare a Veda render with the same-seed dense output before treating it as a production speedup. See [pipeline notes](docs/programmatic-flyover.md#optional-veda-h3-attention).

ComfyUI v0.38.2 is installed separately at `/run/media/jamie/Data/Image Creation/Comfy` for Linux, using `/home/jamie/comfyui-venv` with PyTorch 2.11 CUDA 13.0. Its ignored `extra_model_paths.yaml` points to the GeoBlock H3 weights and the older portable ComfyUI model library, without duplicating model files. `stage_h3_inputs.py` copies the checked photos and MP4 to `Comfy/input/geoblock_worms_head/`. Start this installation with `bash scripts/start_comfy_h3.sh` and open `http://127.0.0.1:8188`. A 124-frame native Ref2VA preview completed with two photographs and a motion clip. For the slower follow-up with moving breakers, the broad turn, and the eased return, run `python scripts/queue_h3_preview.py --take fixed --submit`. This writes `outputs/h3_fixed_workflow_api.json`, stages a 175-frame version of the revised clay reference, and saves a separate `WormsHead_H3_fixed` video without replacing earlier takes. The intermediate `--take next` version remains available for comparison; it develops a hard visual jump near its end.

The 175-frame fixed take still fades toward a different viewpoint in its last 15 frames. The review cut keeps the first 160 frames (6.67 seconds), which cover the outward flight, wide turn, and the start of the return without that transition:

```bash
ffmpeg -i "/run/media/jamie/Data/Image Creation/Comfy/output/video/WormsHead_H3_fixed_00001_.mp4" \
  -frames:v 160 -an -c:v libx264 -crf 18 -pix_fmt yuv420p \
  "/run/media/jamie/Data/Image Creation/Comfy/output/video/WormsHead_H3_review.mp4"
```

## Licence

GeoBlock's code and documentation are licensed under the [GNU General Public License v3.0 or later](LICENSE). ComfyUI and Blender are both GPL, and the custom nodes and Blender scripts here run inside them, so GPL keeps the whole pipeline in one compatible licence.

Geographic data fetched by the pipeline keeps its own terms, recorded in each scene's `ATTRIBUTION.md` and metadata: USGS 3DEP is public domain, Welsh Government LiDAR is OGL, Wikimedia Commons photos carry their individual licences, and OpenStreetMap search results are ODbL.
