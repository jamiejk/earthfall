# Reproducing the GeoBlock prototype

This repository contains source code, small scene specifications, ComfyUI workflow examples, and two original Woodside style-reference images. It does **not** contain the acquired rasters and point clouds, derived meshes or videos, Wikimedia Commons photographs, MiniMax H3 weights, or the Veda predictor. Those live under ignored `data/`, `outputs/`, and `models/` directories on the development machine. An example canvas is not a self-contained render until its referenced assets have been rebuilt or supplied.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/) for the GIS compiler and tests.
- Blender and FFmpeg for building the 3D blocking shots and motion-reference MP4s.
- ComfyUI 0.38.0 or newer, a suitable GPU, and the separately obtained MiniMax H3 Ref2VA weights for local video generation. The ComfyUI node uses its own Python environment; install its dependencies there.
- Access and rights to each source dataset and model. Check the model licence and the source-specific attribution/licence before downloading or publishing derived work. `scripts/download_h3_ref2va.sh` pins the four Comfy-Org files but does not grant model-use rights.

## Code and lightweight validation

From the repository root:

```bash
uv sync --extra test
uv run pytest -q
uv run geoblock inspect --spec examples/worms_head.json
```

The tests and `inspect` do not require a GPU or downloaded terrain. The CLI commands in [README](../README.md) show how to prepare the two Worm's Head terrain variants and run their height-datum comparison. The [source check](source-check.md) records the vertical-reference uncertainty that prevents treating the present Welsh LiDAR crop as a certified sea-level datum.

## ComfyUI preparation graph

After preparing a fixture's terrain and reference assets, link the node into a ComfyUI checkout and restart ComfyUI:

```bash
export COMFY_ROOT=/path/to/ComfyUI
ln -s "$(pwd)/comfy_geoblock_prototype" "$COMFY_ROOT/custom_nodes/geoblock_prototype"
python scripts/create_geoblock_workflow.py
```

Load `examples/landmark_selection_comfy.json` to try the user-driven `Search Landmarks → Choose Landmark` flow. It searches only when queued, displays numbered results, and passes the selected landmark's coordinates and OSM feature ID onward. It does not yet fetch 3D geometry. The two fixture canvases, `examples/geoblock_prototype_comfy.json` and `examples/woodside_library_comfy.json`, remain for reproducing earlier renders: their five-node preparation flow loads a prepared scene, plans a fixed route, checks references, and exposes image/video/prompt outputs. It stops before H3 generation. The fixture scene loader does not download geometry for arbitrary chosen landmarks.

For H3, set up the compatible weights with `scripts/download_h3_ref2va.sh` and point ComfyUI's `extra_model_paths.yaml` at the resulting `models/MiniMax-H3` directories. Run `python scripts/queue_h3_preview.py --help` or `python scripts/queue_woodside_h3.py --help` for the staged API graphs. Neither queue script submits without `--submit`. The optional `--veda` flag needs the [Veda ComfyUI node](https://github.com/veda-sparse/Veda-on-ComfyUI) and its predictor; compare same-seed dense and Veda outputs because this project's Ref2VA settings are outside that predictor's training setup.

## Source and result boundaries

Worm's Head has a measured landform relief example, but the sea plane is an illustrative height and the tidal causeway has source gaps. Woodside has a USGS 3DEP bare-earth terrain, NAIP place images, OSM footprints and lidar-estimated building heights. Its flat-roof extrusions establish volume only; they do not pass the project's proposed building-shape requirement for an architecture-led demo. See the [map-resource survey](map-resource-survey.md) for candidate LoD2/mesh sources and the [pipeline status](programmatic-flyover.md) for the unimplemented general-address stages.

The generated PNG style references in `assets/style_references/woodside_clearline/` are illustrative, not geographic ground truth. Their manifest records the source frame and generation prompt. Wikimedia Commons photos are mined separately, with attribution kept in the ignored local pack. No Google or Apple photogrammetry tiles are part of this repository or its supported input pipeline.
