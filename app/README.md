# Earthfall app

The public app is branded **Earthfall**. Existing `touchdown` script names, environment variables, API/data paths and frozen recipe identifiers remain stable. The header and favicon use an acid-green tile with the figure rotated head-down.

## Canonical workflow diagram — 27 October 2026

The **How it works** navigation link opens `/#about`, an accessible HTML flow diagram of
[photographic-grounded-v1](../docs/touchdown-canonical-workflow.md): photographic flight → destination-grounded
regional texture/outpainting/orbital reference → regenerated opening and frame-matched assembly. It uses the
site's existing styles, stacks vertically on small screens and requires no diagram library or network service.
Create now queues the canonical generation chain. Its automatic Qwen layout/review layer replaces
the assistant-authored layout and inspections used in the accepted control. Uncertain or failed
checks hold a take; normal use does not require intermediate user approvals. Approximate geography
and validation limits are stated on both workflow pages.

## Studio layout and browser updates

The example player occupies the full first row; the **Create** form sits below it. The first form
stage is **Choose a place**. Studio CSS and JavaScript URLs carry content hashes so fresh HTML does
not reuse an older browser-cached script that cannot initialize the example selector. Update those
hashes when changing either asset. The reviewed example requests bypass cache and time out after
eight seconds, then use installed legacy examples or an explicit empty state.

## Reviewed example selector

The player and the example cards under it use the catalog in `app/web/reviewed-examples.json`: Bamburgh,
Eilean Donan, Chambord and Pena. Selection is independent of the new-film landmark controls. Only installed
gallery URLs appear; the previous studio examples are a fallback when the films are not installed.

The films themselves are tracked in `app/examples/`, and `app/server/reviewed_examples.json` records each one's
title, caption and location-photo credits. `touchdown-app` runs the installer on every start; to install into
another app data directory by hand:

```bash
.venv/bin/python scripts/install_earthfall_examples.py --gallery /path/to/app-data/gallery
```

The installer verifies existing film hashes, copies new clips, creates posters (a frame 60% through each film)
and writes the public catalog; it makes no generation calls and does not alter the source films. MP4s continue to use the
existing `/api/gallery/` routes and their passcode policy, rather than public static-file serving.

## Technical Workflow tab

`/static/workflow.html` is a separate navigation page for judges. A local SVG dependency graph shows
20 stages and 26 connections, including fresh source preparation, model generation, custom Python,
Qwen arrival/approach/layout/intermediate checks, hold branches and final assembly. Selecting a node opens
its scripts, exact model filenames, captured loader settings and ComfyUI node IDs/classes.
`workflow-graph.json` describes the canonical generation chain and current automatic app orchestration; it is not a live job monitor
or an executable ComfyUI workflow. Its graph data can be downloaded, and connections also have a text view.
All assets are served locally with no external diagram library. Create uses `touchdown.canonical_build`; the historical control and frozen inference files remain unchanged.

## Photographic studio — 22 October 2026

The default page is now the photographic studio: a large example player, a grouped dropdown of the modern
seven wonders, other landmarks and city views, an adjacent custom-place field with global search, portrait
and location-reference uploads, and ending/framing controls. A tumbling stick figure is the header icon and
favicon. There is no map-coverage restriction on a photographic setup.

**Create video** saves the setup and submits it to a photographic-capable GPU worker. **Save setup** remains
available independently. Worker heartbeats control availability; an offline worker does not become a false
“ready” flag. The queue distinguishes photographic jobs from terrain jobs, so an older terrain-only worker
cannot claim a photographic setup. Existing terrain APIs and job links remain available.

Run the studio without starting a GPU worker:

```bash
PYTHONPATH=app TOUCHDOWN_DATA=/tmp/touchdown-studio \
  .venv/bin/python -m server.cli --server-only --no-browser --port 8014
```

Open `http://127.0.0.1:8014`. Set `TOUCHDOWN_PASSCODE` before sharing access beyond your own machine. Existing
passcode protection applies to studio setup APIs, uploaded photos and example files.

- `GET /api/studio/config` returns photographic destinations, example availability, upload limits and creative
  choices. `render_available` reflects a photographic worker heartbeat within the last 45 seconds.
- `GET /api/studio/search?q=...` searches globally through the existing rate-limited, cached Nominatim client.
  Search only runs on explicit submission. Selecting a result automatically searches for location photos.
- `POST /api/studio/location-photos` accepts JSON `destination` and optional `retry`. It reuses installed
  references for the three existing examples, otherwise searches Wikimedia Commons by coordinates and name
  through `geoblock_core.place_photos`. It retrieves up to three openly licensed photographs and returns
  image URLs, author/licence/source credits and source/sanitized SHA-256 hashes. Results are cached under
  `<data>/location-photos/`; unsuccessful searches are cached for ten minutes and can be retried explicitly.
  A custom name without coordinates is resolved only on Find photos or Save; ambiguous names require a
  more specific search result. Typing alone makes no geocoder or photo-search requests. Discovery failures
  explain the upload fallback, and found photos remain unreviewed references.
- `POST /api/studio/setups` accepts multipart `destination_json`, `ending`, `framing`, optional `wearing`,
  `subject` and up to four `references`. Both accept JPEG, PNG, WebP and HEIC/HEIF (including `.hif`), up to
  12 MB each by default. Files are decoded, oriented, resized to a maximum side of 2048 pixels and re-encoded
  without metadata as JPEG or PNG. The existing 40-megapixel decoding limit still applies. Animated WebP
  imports its first frame; HEIF imports its primary image. For a revision, `base` identifies a previous setup and `retained` is a
  JSON array of its reference filenames. The portrait can be reused without another upload. Uploaded location
  photos cannot be retained across destination changes. `reference_set` binds an automatic selection to its
  canonical destination; `automatic` is a JSON array of selected sanitized SHA-256 hashes. Omit it to use all
  found photos, or send `[]` to use none. Saving also discovers references if the client has not done so.
  Selected automatic photos are copied into the immutable setup alongside their credits and hashes; reopening
  and revising do not require the discovery cache or the remote originals to remain available.
- `GET /api/studio/setups/{id}` restores the setup; the browser link is `/#setup={id}`. Saved revisions are
  independent snapshots under `<data>/setups/<id>/`, with sanitized images and a `setup.json`. No uploaded
  portrait/reference is published to the gallery. Keep the saved link to reopen the setup.
- `GET /api/studio/setups/{id}/images/{name}` only serves the exact recorded subject/reference filenames.
- `POST /api/studio/images/normalize` returns a sanitized JPEG/PNG preview. This lets HEIC/HEIF uploads display
  in browsers that cannot decode them. Temporary originals/previews are removed after the response is prepared,
  and the endpoint uses the same passcode, upload-size and pixel limits as saved uploads. Saving waits for
  previews to finish; stale conversion responses cannot overwrite newer portrait/location selections.
- `POST /api/studio/setups/{id}/render` queues a photographic job. Repeated submissions reuse an active take
  (or a finished take unless `?retry=true`). Offline returns 503; missing/changed references return 422/409.
  A held/failed/cancelled take can be tried again without altering its saved inputs.
- `GET /api/studio/availability` reports worker availability. The page refreshes it every 15 seconds.
- Job links (`/#job=...`) show seven photographic stages, previews, cancellation and the delivered MP4.
  `held` is a terminal review state which frees the queue; a generated candidate may be viewed/downloaded as
  a **review take**, without being presented as accepted. The original setup and previous attempts remain.
  `GET /api/jobs/{id}/review` provides the checks. Acceptance requires a review matching the uploaded video hash.
- Setups with fewer than two available references say `needs_location_photos`; otherwise they say
  `needs_reference_review`. Neither state claims a visual check passed. Twirl and arms-open require full-body
  or three-quarter framing. Some motion options are explicitly labelled as trials.

`app/server/studio_destinations.json` defines the dropdown's 21 places in three groups. The server resolves
preset IDs against this catalogue so they survive saving/reopening with their canonical place and coordinates.
The custom field can also be used directly without waiting for search. Changing place clears old location
references while retaining the person photo. Presets and resolved custom locations fetch photos automatically;
users can inspect credits, remove a pick or add their own images. A destination's presence in the dropdown
does not mean a reviewed reference set or example exists.

`app/server/studio_landmarks.json` is the explicit allowlist of source-photo metadata and modern example films.
The three small poster JPEGs are bundled; the previous reference thumbnails remain available as assets. The
films and full location reference photos remain in their
existing `outputs/touchdown/` folders. Missing examples produce an empty state, and missing installed source
photos trigger Commons discovery. There is no recursive scan of experiment folders. Bundled card photo credits are in
`app/web/img/landmarks/ATTRIBUTION.md` and accessible in the interface.

The remaining integration work is recorded in
[`docs/review/web-app-redesign-implementation-2026-10-22.md`](../docs/review/web-app-redesign-implementation-2026-10-22.md).

## Photographic GPU worker

Install the project's app, touchdown and Comfy SDK dependencies on the GPU host and configure its existing
ComfyUI install with FLUX 3 Klein, LTX 2.5 and the SDK proxy. The evaluated local Qwen3.5 4B build must already
be installed; the worker preflight does not download models. The web server needs no GPU dependencies.

```bash
export TOUCHDOWN_SERVER=http://127.0.0.1:8014  # or the NAS HTTPS address
export TOUCHDOWN_WORKER_TOKEN=...             # same private token as the server
export TOUCHDOWN_CANONICAL=1
export TOUCHDOWN_PHOTOGRAPHIC=0  # historical photographic jobs only
export TOUCHDOWN_PYTHON=/absolute/path/to/project/.venv/bin/python
export TOUCHDOWN_OUTPUT=/absolute/path/to/studio-renders
export TOUCHDOWN_COMFY_SERVER_CMD="/absolute/path/to/comfy-venv/bin/python main.py --listen 127.0.0.1 --port 8192 --reserve-vram 1.5 --disable-auto-launch"
export TOUCHDOWN_COMFY_SERVER_CWD=/absolute/path/to/ComfyUI
export TOUCHDOWN_FLUX3_FREE_URL=http://127.0.0.1:8192
export TOUCHDOWN_COMFY_URL=http://127.0.0.1:8193
PYTHONPATH=app .venv/bin/python -m worker
```

The worker downloads only the saved setup's exact allowlisted inputs and verifies their hashes. The isolated
4B model selects 2–4 references and describes the setting, connected ground, appearance and regional palette.
FLUX creates a full-body landing photograph with visible shoes, regardless of the final crop. Required
arrival checks are bound to the actual image, model/checker build, brief, allowed support and evidence pixels.
Two arrival takes are allowed; an unresolved check holds the job before video generation.

The canonical scene preparation uses the historical photographic high-view instruction
(approximately two kilometres); it does not impose a separate 30,000-foot phase.
The native portrait retains the saved ending/framing choices. The frozen split
flight generates parent, repaired anchor and descent with Prompt Relay and the
canonical preset, without Layout conditioning or flight latent refinement.

Create now runs **photographic-generic-orbit-v1** (28 October 2026): after the flight it finds the
flight's first full-frame cloud frame J and its zoom rate there, builds a camera-only proxy of a cached
generic cloud-covered Earth from low orbit (`assets/touchdown/opening/`), with no cloud added, and
generates a 49-frame LTX opening on Layout To Render 0.3 with flight frames J+8…J pinned at frame 0,
so LTX grows one cloud bank into the flight's densest white cloud frame J
(`scripts/touchdown/generic_opening.py`; `TOUCHDOWN_OPENING_MODE` selects the older modes). Delivery reverses generated
frames 48…9 into the first 40 frames and then plays the flight from frame J, so the film is
40 + (flight − J) frames. No crossfade or interpolation is added. The opening does not show the
destination from orbit. The layout, texture, outpaint, projection and orbit stages and their checks are
skipped; `TOUCHDOWN_CANONICAL_RECIPE=photographic-grounded-v1` restores them.

Nine automatic Qwen checkpoints record evidence and decisions. Failed/unclear
findings hold the take. A completed result requires the whole chain and a final
review bound to the delivered video hash. Source credits, graphs, code snapshots,
model records and decisions remain in the job output. The optional detail pass
is a separate source-bound child job.

ComfyUI is owned and stopped between generation/checker stages; another process
using its port is left untouched. Each Fill pass releases its runtime before
Qwen loads. The worker maintains progress and availability heartbeats during
long calls, and cancellation stops its owned process group. See the full
[canonical integration specification](../docs/touchdown-canonical-workflow.md).

## Previous terrain app and worker

The sections below describe the older terrain workflow and its retained APIs/worker. They do not describe the
current canonical creation path; its saved photographic setups render through the canonical worker described above.

A web interface for the Touchdown Earth-zoom pipeline: pick a real place, upload a photo of a person, and get a
clip that dives from orbit, through the clouds, over the measured terrain, and lands on that person. The app
does not generate anything itself. It writes a manifest and runs the existing build
(`python -m touchdown.build`, in `scripts/touchdown/`) on your own GPU through ComfyUI, using open-weight models
only (FLUX 3 Klein stills, LTX 2.5 video) plus Blender. It calls no paid APIs.

There are two ways to run it:

1. **Local (the main way):** `touchdown-app` runs the web app and a worker in one process on the GPU machine,
   and opens `http://127.0.0.1:8000`.
2. **Hosted demo (optional):** the web app runs on another always-on machine (for example a NAS in Docker, made
   public through a Cloudflare Tunnel), and a worker on the GPU machine pulls jobs from it over HTTPS.

The front end is plain HTML, CSS and JavaScript (`app/web/`) with no build step, so a native shell could wrap it
later.

## What v1 does, and what it doesn't

- **Places:** a searchable list of the place catalogue (`src/geoblock_core/data/places.json`). The 24 US places
  that have a USGS 3DEP 1 m lidar DEM and NAIP imagery in the catalogue can be picked; the other catalogue places
  are listed but disabled ("Coming soon: needs terrain data" outside the US, or no 1 m lidar / no NAIP). Monument
  Valley, Grand Canyon (Mather Point) and Yosemite (Tunnel View) keep their hand-written manifests in
  `shots/flux3/`. Every other place gets a manifest generated on the worker by `touchdown.manifest_gen` (see
  "Generated manifests" below). Each place is marked **Tested** (it has had a real build: the hand-written three,
  listed in `app/tested_places.json`) or **Untested** (selectable, with a warning). After a real build of another
  place, run `touchdown-app mark-tested <place id>` on the machine that runs the server; it writes
  `<data>/tested_places.json`.
- **Custom US location (experimental):** search a place name or click the map. The server checks the point with
  the catalogue builder's own test, `geoblock_core.lidar.dem1m_projects_at` (a USGS 1 m DEM tile under the point;
  lower 48 only, because NAIP is). Points without 1 m lidar are refused: the pipeline's ~10 m fallback
  (`project = "3dep13"` in `terrain_route.acquire`) is not offered. The name comes from the search result, a reverse
  lookup, or "Custom place", and can be edited. Nothing has been built this way yet.
- **Place search** uses OpenStreetMap Nominatim, but through the server (`/api/geocode`), not from the browser: a
  browser cannot set the identifying User-Agent Nominatim's usage policy asks for. The server sends one, makes at
  most one request a second, caches answers, and only asks when the user presses Search or clicks the map (no
  search-as-you-type, which the policy forbids). `TOUCHDOWN_GEOCODER_URL` points it elsewhere (for example your own
  Nominatim). If search fails, clicking the map still works.
- **Person:** a JPEG, PNG, WebP or HEIC/HEIF upload (12 MB limit by default). The server re-encodes it, which drops EXIF, GPS and
  all other metadata, and scales it to at most 2048 px. An optional "what they're wearing" field becomes
  `[shot] person` ("a person wearing ..."). Left empty, the preset's own description is used, and that
  describes the test subject, not your photo. Nothing looks at the photo to describe it yet:
  `app/worker/describe.py` (`describe_person`) is the one place a local captioner would plug in.
- **Run:** queue position, then a live stage list (server-sent events, with polling as a fallback). The orbit,
  cloud and end stills appear as previews once the stills stage finishes. At the end you get a video player and a
  download button. You can cancel a job, whether it is queued or running.
- **Gallery:** lists the mp4s in `<data>/gallery/`. It starts out empty, because no finished Touchdown clips are
  committed in this repo. On start the server copies in any mp4s it finds under
  `docs/review/fakeout/touchdown/flux3-v14/` (or `TOUCHDOWN_GALLERY_SEED`). Run
  `touchdown-app publish <job id> --title "..."` to add a finished job. User results are never listed
  automatically; only someone who has a job's link can see it.
- **One build at a time.** A build takes about 15 to 25 minutes on an RTX 4090. Other jobs wait in a queue
  (10 at most by default).

## Local mode

### Install

The pieces below are in the order you need them, and each one says what to type. The app itself installs in
seconds; the model weights and the ComfyUI server are the slow parts, and nothing here is free of disk or time.

**1. This repository, Python 3.12 and [uv](https://docs.astral.sh/uv/).**

```bash
git clone https://github.com/jamiejk/geoblock-scenecompiler && cd geoblock-scenecompiler
uv sync --locked --extra test --extra touchdown --extra app
```

That one line is what CI runs: `--extra test` is pytest, `--extra touchdown` is OpenCV (the pipeline needs it) and
`--extra app` is FastAPI (the web app). The pipeline's ComfyUI jobs also want `comfy-sdk`
(`uv sync --extra comfy`), but that often lives in another Python, ComfyUI's own venv; see "If the pipeline's
OpenCV and comfy_sdk live in different Pythons" below.

**2. `torch` and `sentence-transformers`, in the Python that runs the build.**

The view ranker (`scripts/touchdown/view_rank.py`) and the photo ranker (`scripts/touchdown/photo_op.py`) import
them. They are not in `pyproject.toml` on purpose: `torch` is a multi-gigabyte download whose CUDA build has to
match your driver, and the pipeline's Python is not always the project's venv.

```bash
uv pip install torch sentence-transformers              # in this project's venv
/path/to/ComfyUI/venv/bin/python -m pip install torch sentence-transformers   # or in ComfyUI's venv
```

`sentence-transformers` also pulls `torchvision`, and the ranker reads the preview images with Pillow and OpenCV,
so a separate environment needs those too:

```bash
uv venv --python 3.12 ~/.venvs/td-embed
uv pip install --python ~/.venvs/td-embed/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install --python ~/.venvs/td-embed/bin/python sentence-transformers pillow opencv-python-headless
export TOUCHDOWN_EMBED_PYTHON=~/.venvs/td-embed/bin/python     # the build then uses it for the rankers
```

The CPU build of torch is enough: the ranker scores a handful of preview images. Without `TOUCHDOWN_EMBED_PYTHON`
the build uses its own Python, which then needs `torch` and `sentence-transformers` installed there.

**3. The `google/embeddinggemma-2` model, in the local Hugging Face cache.**

The build runs the ranker with `HF_HUB_OFFLINE=1` (`scripts/touchdown/build.py`), so the model has to be cached
already: a build never downloads it.

```bash
hf download google/embeddinggemma-2        # or: huggingface-cli download google/embeddinggemma-2
```

15 of the 27 US catalogue places need it, `grand-canyon-mather-point` among them: they have no curated
`view_toward`, so `scripts/touchdown/manifest_gen.py` turns on `embedding_rank` and the build ranks the measured
candidate views with this model. The other 12 have a curated heading and never load it. The ranker can run in
its own Python (`TOUCHDOWN_EMBED_PYTHON`, with its packages on `TOUCHDOWN_EMBED_DEPS`), which is how the desktop
keeps torch out of the build's Python.

**4. ComfyUI, with the models the pipeline asks it for.**

- **ComfyUI v0.41.3 or newer** (FLUX 3's nodes are not in older releases), with the custom nodes the pipeline
  uses: GeoBlock and ComfyUI-LTXVideo. The manifests expect a `comfy-api-proxy` in front of ComfyUI.
- **FLUX 3 Image Klein** (the stills): `flux3-image-klein-bf16.safetensors` (diffusion_models),
  `qwen3vl_4b_bf16.safetensors` (text_encoders) and `flux2-vae.safetensors` (vae). Those are the names the
  pipeline asks ComfyUI for; `scripts/download_flux3_klein.sh` fetches BFL's INT8 package under different names,
  so rename them or point ComfyUI's `extra_model_paths.yaml` at that folder.
- **FLUX.2 [klein] 9B fp8** (the alternative keyframe painter): `flux-2-klein-9b-fp8.safetensors` with
  `qwen_3_8b_fp8mixed.safetensors`.
- **LTX 2.5**: `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`,
  `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`, the layout-to-render IC-LoRA
  (`ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors`), the video VAE (`ltx-2.5-video-vae-bf16.safetensors`)
  and the x2 latent upscaler (`ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`).
- **The sound, if you want it from the model:** the default sound stage is procedural wind, so no model is needed.
  The optional Foley V2A windows need the gated `ltx-2.3-foley-v2a.json` workflow next to the manifest,
  `ltx-2.3-22b-lora-foley-v2a-1.0.safetensors` and the LTX 2.3 checkpoint `ltx-2.3-22b-dev-fp8.safetensors`
  (which carries the audio VAE).

The ComfyUI address is hard-coded per manifest: every `shots/flux3/*.toml` has `[stills] flux3_url` on
`http://127.0.0.1:8193` and `flux3_free_url` on `http://127.0.0.1:8192`, and the build's `--url` defaults to
`8193`. Stock ComfyUI listens on `8188`, so either edit those two lines in the manifests, or run your servers on
8193 and 8192 and set `TOUCHDOWN_COMFY_URL` for the build's own `--url`.

**The build needs the API proxy, not plain ComfyUI.** It submits through `comfy-sdk`, which speaks the Comfy API v2
endpoints that plain ComfyUI does not serve: against a bare server the build fails with `HTTP 405: Method Not
Allowed` from the SDK's asset upload. Run one in front of ComfyUI and point the build at it:

```bash
python -m comfy_api_proxy.cli run --comfyui http://127.0.0.1:8188 --host 127.0.0.1 --port 8190
export TOUCHDOWN_COMFY_URL=http://127.0.0.1:8190        # the build's --url
```

`TOUCHDOWN_FLUX3_URL` and `TOUCHDOWN_FLUX3_FREE_URL` do the same job for the stills' own address, which the
manifests otherwise hard-code.

**Without the gated FLUX 3 weights, draw the stills with FLUX.2 [klein] instead.** FLUX 3 Image Klein is a gated
Hugging Face package; `TOUCHDOWN_STILLS=klein` drops those stills from the job manifest and draws them with
`flux-2-klein-9b-fp8.safetensors`, which is open weights. The stills are then 1344x768 rather than 2K, and the
outline conditioning (the thing that holds the skyline) stays on.

```bash
export TOUCHDOWN_STILLS=klein
```

**5. Blender and ffmpeg on `PATH`, and a 24 GB GPU.**

**The LTX 2.5 stack peaks around 22.7 GB on a 24 GB card, so nothing else may be resident.** A build takes about
15 to 25 minutes on an RTX 4090.

The card is needed twice, by two different models: the stills (FLUX 3 Klein, about 10 GB with its text encoder) and
the video (the LTX 2.5 transformer, 21.5 GB). ComfyUI's VRAM paging holds the weights it has loaded, and asking
it to unload them does not give the memory back, so the video stage can fail with `VRAM Allocation failed` or
`Fault failed: 2` on a 24 GB card. The way out is to restart the server in between, which the build can do when it
owns the server:

```bash
export TOUCHDOWN_COMFY_SERVER_CMD="/path/to/comfyui-venv/bin/python main.py --listen 127.0.0.1 --port 8192 --reserve-vram 1.0"
export TOUCHDOWN_COMFY_SERVER_CWD="/path/to/ComfyUI"
export TOUCHDOWN_COMFY_SERVER_URL="http://127.0.0.1:8192"
export TOUCHDOWN_COMFY_FREE_URL="http://127.0.0.1:8192"   # the plain server, for the model unload
```

The build then starts ComfyUI for the stills, stops it (the FLUX 3 pages go with it), starts it again for the
video stages, and stops it at the end. The build talks to ComfyUI through `comfy-api-proxy`, so the proxy has to be
running in front of that address; the proxy survives the restarts.

Without those variables the build uses the ComfyUI it is given and prints a warning: on a 24 GB card the video
stages may not fit.

Two more things that decide whether a build survives on a shared card:

- **Leave ComfyUI's dynamic VRAM on.** Starting it with `--disable-smart-memory` and a large `--reserve-vram`
  leaves nothing to offload to, and the 241-frame opener dies with `Allocation on device 0 would exceed allowed
  memory`. `--reserve-vram 1.5` with the default smart memory survived a card that had 13 GB taken by the desktop.
- **Use the draft quality when the card is shared.** The job's `quality` (the app's quality selector) picks
  960x544 without the 2x pass, which is what the 241-frame opener needs to fit. The 2x pass at the end is the
  expensive one.
- **The LTX 2.5 text encoder belongs on the CPU.** It is 15.4 GB and the transformer 21.5 GB, and the transformer
  needs the card. The pipeline's graph asks for `CLIPLoader(device="cpu")` for the LTX encoder (it runs once per
  job, before sampling), which leaves the whole card for the transformer.
- **Restart ComfyUI between builds on a shared machine.** After a long build it holds the models in pinned RAM —
  48 GB on this setup, with the swap full and the desktop fighting for the rest. A restart gives all of it back.

**FLUX 3, when you have the weights.** FLUX 3 Image Klein needs **ComfyUI v0.41.3 or newer** (v0.38.2's
`CLIPLoader` has no `flux3` type) and the package from
`black-forest-labs/flux-3-image-klein-comfy` (`scripts/download_flux3_klein.sh`; accept the licence there, not on
`flux-3-image-klein-base`, which is the same model in the diffusers layout). Two servers, as the manifests assume:

```bash
# the FLUX 3 server, and the proxy in front of it
python main.py --listen 127.0.0.1 --port 8192 --reserve-vram 1.5      # in a ComfyUI v0.41.3 checkout
python -m comfy_api_proxy.cli run --comfyui http://127.0.0.1:8192 --host 127.0.0.1 --port 8193 \
    --comfyui-base-dir "/path/to/ComfyUI"
```

The manifests then point the stills at 8193 and ask 8192 to unload afterwards, and the video stages run on the
build's own `--url`. On a 24 GB card use the int8 weights:

```bash
export TOUCHDOWN_FLUX3_UNET=flux3-image-klein-int8-convrot.safetensors
export TOUCHDOWN_FLUX3_CLIP=qwen3vl_4b_int8_convrot.safetensors
```

**6. Internet access for the terrain stage**, which downloads free USGS 3DEP relief and NAIP imagery. The data
covers the US only.

None of steps 2 to 6 is needed to try the interface: with `TOUCHDOWN_BUILD_CMD` pointing at the mock build
(`app/worker/fake_build.py`) the app writes the same folders and log lines in a few seconds and makes a
test-pattern video, with no GPU, no ComfyUI and no models. See "Trying the interface without a GPU" below.

### Steps

After step 1 (the clone and the `uv sync`), this is the whole local run:

```bash
# 1. Check the pipeline itself (no media is made; it lists the stages it would run):
cd scripts && PYTHONPATH=.:.. uv run python -m touchdown.build ../shots/flux3/monument-valley.toml \
    --descent 45 --url http://127.0.0.1:8193 --dry-run && cd ..

# 2. Start the app (opens a browser tab; --no-browser to skip):
uv run touchdown-app
```

Then open <http://127.0.0.1:8000>, pick a place, add a photo, and press **Make my Touchdown**.

By default the server listens on 127.0.0.1 only. `--host 0.0.0.0` makes it reachable from your LAN; set
`TOUCHDOWN_PASSCODE` if you do that.

If the pipeline's OpenCV and comfy_sdk live in different Pythons (see the docstring of
`scripts/touchdown/build.py`), point the app at them:

```bash
TOUCHDOWN_PYTHON=/path/to/ComfyUI/venv/bin/python \
TOUCHDOWN_BUILD_CMD="{python} -m touchdown.build {manifest} --descent {descent} --url {url} --comfy-python $PWD/.venv/bin/python" \
uv run touchdown-app
```

### Trying the interface without a GPU

A stand-in build writes the same folders and log lines in a few seconds and makes a test-pattern video:

```bash
TOUCHDOWN_BUILD_CMD="{python} {app}/worker/fake_build.py {manifest}" FAKE_STAGE_SECONDS=2 uv run touchdown-app
```

## Generated manifests

`scripts/touchdown/manifest_gen.py` writes a FLUX 3 manifest for a catalogue place or a custom point:

```bash
cd scripts && python -m touchdown.manifest_gen horseshoe-bend --subject ../shots/subjects/test-subject-red-coat.jpg \
    --person "a person in a yellow jacket" -o /tmp/horseshoe.toml          # --naip: NAIP ground words (network)
python -m touchdown.manifest_gen --list                                    # what can be generated, and why not
```

The per-place words (scene, region, approach, swoop view and target, ambient, orbit region, end and opener-end
prompts, swoop prompt, view and route queries) come from templates per terrain family (desert, canyon, mountain,
valley, coast, volcano, forest, hills, city), filled from the catalogue's name, region, kind, `view_toward` and
WorldCover land cover, and from NAIP ground words when the worker can measure them (`TOUCHDOWN_NAIP_WORDS=0` turns
that off). The scene type (landmark, urban, nature) is decided from the catalogue too and written to
`[stills] scene_type`. Everything that is not per place is copied from `shots/flux3/monument-valley.toml`. The
terrain table names the catalogue's newest 1 m project and lets the build search the landing: with the catalogue
view heading when there is one, else `heading = -1` (view search with the existing embedding ranker, as the Grand
Canyon manifest does). A custom point's manifest carries a `[place]` table, which `touchdown.build` registers so
`find_place` knows it.

The worker writes the generated manifest to `<output>/app-places/<place>.toml` and then copies it into the job
like a preset.

## How a job runs

The worker (in-process in local mode, or remote):

1. Copies the preset manifest to `<output>/app-<preset>-<job id>/job.toml`. It sets `[shot] subject` to the
   uploaded photo (saved next to it as `subject.jpg`), `[shot] person` to the description, `[shot] name` to
   `app-<preset>-<job id>`, and `output` to the absolute output folder. The few paths the build resolves
   against the manifest's folder are made absolute. Nothing else changes.
2. Runs `python -m touchdown.build <job.toml> --descent 45 --url http://127.0.0.1:8193`, the same command as by
   hand, from `scripts/` with `PYTHONPATH=scripts:<repo>`. `TOUCHDOWN_BUILD_CMD` can change it.
3. Follows progress from the folders the build already writes. `<root>/<stage>/<key>/` appears when a stage
   starts and gets `done.json` when it finishes. `[stage] cached` lines in the output mark reused stages. It
   uploads `orbit.png`, `cloud.png` and `end.png` from the stills folder as small JPEG previews. Then it uploads
   the `touchdown.mp4` named on the build's `Built ...` line (the sound stage's output). If the build fails, the
   worker marks the job failed and sends the last 80 lines of its output.
4. Cancelling a running job stops only the build this worker started, by its PID. On Linux and macOS the build
   is started in its own session, so the signal goes to that process group: the build and the Blender, ffmpeg and
   ComfyUI-client processes it spawned. On Windows it uses `taskkill /PID <pid> /T`. Nothing is killed by name.

For hand-written presets `scripts/touchdown/` runs unchanged. The one hook in `build.py` registers a generated
manifest's `[place]` table (custom points only).

### Per-job names, and why the terrain is still shared

Every stage's cache key in `build.py` hashes the stage settings, the code, and the **whole `[shot]` table**,
including `subject`, `person` and `name`, plus the subject photo's hash. Two consequences:

- A shared `[shot] name` would not hand one person's stills to another, because the person is in every key. But
  it would put everyone's jobs, build logs and reports in one folder, and it wouldn't save anything: a new person
  changes every key anyway, terrain included.
- So each job gets its own name (`app-<preset>-<job id>`) and its own folder, under the preset's usual output
  folder (`scripts/outputs/touchdown/` by default). That parent folder holds the build's shared download caches
  (`imagery/`, `dem_cache/`, `overture/`, `reference_photos/`, ...), and they keep working across jobs.

The terrain stage is the slow one that doesn't depend on the person. It reads the place, the scene text and the
`[terrain]` table, all copied unchanged from the preset. It never reads `subject`, `person` or `name`. So before
each build, the worker asks `touchdown.build` itself (`app/worker/terrain_key.py`, run in the pipeline's Python)
for two keys per manifest:

- the real terrain key, which is the folder name the build will look for;
- a "neutral" key, with those three person fields removed.

If the preset's own build (for example `outputs/touchdown/flux3-monument-valley/terrain/<key>/`) or an earlier
job for the same place has a finished terrain folder with the **same neutral key** (same place, same settings,
same code), the worker copies it into the new job under the job's real key. The build then reports terrain as
cached, and stills, opener, swoop, end, face and the rest still run fresh for the new person. If the terrain code
or settings change, the neutral keys stop matching and the job builds its own terrain. The copy is a real copy,
never a hard link, so a later stage can't write into another job's files. To turn this off, set
`TOUCHDOWN_SHARE_TERRAIN=0`.

## Hosted demo (optional): NAS + desktop worker + Cloudflare Tunnel

```
browser ──HTTPS──> Cloudflare ──tunnel──> NAS: touchdown server (Docker)  <──HTTPS poll── desktop: worker + ComfyUI + Blender
```

The desktop opens no inbound port. The worker polls `POST /api/worker/claim` and pushes progress, previews and
the result back.

**On the NAS** (any amd64 or arm64 machine with Docker):

```bash
cp app/.env.example app/.env        # set TOUCHDOWN_WORKER_TOKEN (long random) and TOUCHDOWN_PASSCODE
docker compose -f app/docker-compose.yml up -d --build                    # LAN only, port 8000
docker compose -f app/docker-compose.yml --profile tunnel up -d --build   # plus the Cloudflare Tunnel
```

The image is `python:3.12-slim` plus pure-Python or wheel-only dependencies, so it builds on both architectures.
It contains only `app/server`, `app/web`, the three preset manifests (the server reads each one's default
person text), the place catalogue and `geoblock_core/lidar.py` (the custom-point coverage check, which needs
network access to the public USGS bucket). Data (SQLite queue, uploads, previews, results, gallery) lives in `app/data/`, mounted at `/data`.

**Cloudflare Tunnel:** in Cloudflare Zero Trust, go to Networks > Tunnels and create a tunnel. Copy its token
into `TUNNEL_TOKEN`, and add a public hostname whose service is `http://touchdown:8000` (the compose service
name). For more than a passcode, put **Cloudflare Access** in front of that hostname (for example an email
one-time PIN for a list of judges). Exempt `/api/worker/*` with a service token, or a bypass policy, so the
worker can still reach it. The worker authenticates with its own bearer token in either case.

**On the desktop:**

```bash
uv sync --extra touchdown --extra comfy --extra app
TOUCHDOWN_SERVER=https://touchdown.example.com TOUCHDOWN_WORKER_TOKEN=<same token> uv run touchdown-worker
```

It takes the same `TOUCHDOWN_PYTHON`, `TOUCHDOWN_BUILD_CMD`, `TOUCHDOWN_COMFY_URL` and other settings as local
mode. `--once` runs a single job and exits.

The server alone is `touchdown-app --server-only`; the Docker image runs that.

## Environment variables

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `TOUCHDOWN_DATA` | `./touchdown-data` (`/data` in Docker) | server | queue database, uploads, previews, results, gallery |
| `TOUCHDOWN_HOST` / `TOUCHDOWN_PORT` | `127.0.0.1` / `8000` | server | where to listen |
| `TOUCHDOWN_PASSCODE` | empty (off) | server | users must enter it before using the app; the compose file requires it |
| `TOUCHDOWN_WORKER_TOKEN` | random per run in local mode | both | shared bearer token for `/api/worker/*`; required for `--server-only` |
| `TOUCHDOWN_MAX_UPLOAD_MB` | `12` | server | photo size limit |
| `TOUCHDOWN_MAX_QUEUE` | `10` | server | queued + running jobs before new ones get HTTP 429 |
| `TOUCHDOWN_WORKER_TIMEOUT` | `300` | server | seconds without a worker heartbeat before a running job is failed |
| `TOUCHDOWN_GALLERY_SEED` | `docs/review/fakeout/touchdown/flux3-v14` | server | folders whose mp4s are copied into the gallery on start |
| `TOUCHDOWN_FORWARDED_IPS` | `127.0.0.1` (`*` in compose) | server | proxies trusted for `X-Forwarded-*` |
| `TOUCHDOWN_SERVER` | `http://127.0.0.1:8000` | worker | server to pull jobs from |
| `TOUCHDOWN_REPO` | this checkout | both | repository root (presets, `scripts/`) |
| `TOUCHDOWN_PYTHON` | the app's Python | worker | Python that runs the build (`{python}`) |
| `TOUCHDOWN_BUILD_CMD` | `{python} -m touchdown.build {manifest} --descent {descent} --url {url}` | worker | build command template; also `{app}`, `{repo}` |
| `TOUCHDOWN_BUILD_CWD` | `<repo>/scripts` | worker | working directory for the build |
| `TOUCHDOWN_COMFY_URL` | `http://127.0.0.1:8193` | worker | `{url}`, the build's `--url` (the API proxy's address) |
| `TOUCHDOWN_FLUX3_URL` | the manifest's | worker | the stills' own ComfyUI address |
| `TOUCHDOWN_FLUX3_FREE_URL` | the manifest's | worker | where the stills' server is asked to unload its models |
| `TOUCHDOWN_STILLS` | `flux3` | worker | `klein` draws the stills with FLUX.2 [klein] instead of the gated FLUX 3 Image Klein |
| `TOUCHDOWN_DESCENT` | `45` | worker | `{descent}` |
| `TOUCHDOWN_OUTPUT` | the manifest's `output`, relative to the build cwd | worker | where job folders go |
| `TOUCHDOWN_SHARE_TERRAIN` | `1` | worker | reuse a finished terrain stage for the same place (see above) |
| `TOUCHDOWN_NAIP_WORDS` | `1` | worker | generated places: describe the ground underfoot from NAIP (network) |
| `TOUCHDOWN_GEOCODER_URL` | `https://nominatim.openstreetmap.org` | server | place-name search for custom points |

## Security notes

- A public link drives a real GPU for 20 minutes per job. Use `TOUCHDOWN_PASSCODE`, and ideally Cloudflare
  Access, for anything public. The queue cap (`TOUCHDOWN_MAX_QUEUE`) is a backstop, not rate limiting per user.
- Job ids are random (about 70 bits) and act as the capability to view a job, its previews and its video. Anyone with
  the link can see the result. User results never appear in the gallery unless you publish them.
- Uploads are decoded and re-encoded (JPEG or PNG only, pixel-count cap). No EXIF or GPS data is stored or
  passed to the pipeline. Previews from the worker are re-encoded too.
- The worker token is compared in constant time. Keep it out of the repo (`app/.env` is git-ignored). The worker
  only runs presets from `shots/` and the configured command; nothing from a request is put on the command line,
  and the user's text only reaches the manifest through `describe_person`, which strips braces and quotes.
- The passcode cookie is an HMAC tied to the server process (it resets on restart), `HttpOnly`, `SameSite=Lax`,
  and `Secure` behind HTTPS.
- Uploaded photos and results stay on disk until you delete them (`<data>/uploads`, `<data>/jobs`, and the job
  folders under the pipeline's output). There is no automatic cleanup yet.

## Known gaps

- Generated manifests (every place except the hand-written three, and every custom point) have not had a real
  build. Their words are templates, much plainer than the hand-written ones, and their landing comes from the
  build's automatic search around the catalogue point.
- No captioner: the person text is typed, or falls back to the preset's description.
- Cancelling stops the local build process tree, but a prompt already queued in ComfyUI finishes there unless
  you interrupt it in ComfyUI.
- The place thumbnails are drawn illustrations, not renders. The map uses no tile server: it draws the bundled US
  state outlines (us-atlas), so it works offline (openstreetmap.org refuses tiles to apps like this, 403). Leaflet 1.9.4 is vendored in
  `app/web/vendor/`.
- The Docker image and compose file have not been built on the actual NAS; its architecture is unknown. The
  image is kept multi-arch-friendly for that reason.
- Tests (`tests/test_app.py`) use a fake build. A real GPU build through the app has not been run yet.

## Tests

```bash
PYTHONPATH=scripts:src:.:app .venv/bin/python -m pytest -q tests/test_app.py tests/test_studio.py tests/test_studio_generation.py tests/test_canonical_integration.py tests/test_enhancement.py
node tests/test_enhancement_ui.cjs
```

### Optional Refine Details enhancement

Completed original jobs offer **Refine Details · 4K** when a capable worker is online.
This creates a separate `refine-details-v1` child job, with its own progress,
cancellation and download. The original video is never replaced. A second click
reopens the existing queued/running/completed enhancement. Failed or cancelled
attempts can be retried. The enhanced page links back to the original.

Enable on the GPU worker with `TOUCHDOWN_ENHANCEMENT=1` and
`TOUCHDOWN_ENHANCE_MODEL_ROOT=/path/to/Live-Models`; retain the existing
`TOUCHDOWN_COMFY_SERVER_CMD`, `TOUCHDOWN_COMFY_SERVER_CWD` and SDK URL settings.
Startup checks the required distilled LTX 2.5, Gemma 4, video VAE and Refine Details
LoRA weights. The worker advertises availability every 15 seconds. An enhancement
worker with `TOUCHDOWN_PHOTOGRAPHIC=0` and `TOUCHDOWN_CANONICAL=0` claims only enhancement jobs. All jobs
share the queue and existing GPU lock with normal generation.

`scripts/touchdown/enhance.py` packages the accepted Bamburgh trial: fixed seed,
eight Euler steps, 1024×576 overlapping tiles, CPU text encoder/canvas, tiled VAE,
no Comfy cache. Inputs are split into at most 57 frames with a one-frame overlap;
the temporal window matches each padded chunk to avoid the tiled-encode streaming
failure found during the trial. Assembly drops shared/padded frames and preserves
source audio where present. Output fits a 3840-pixel long edge on the required
32-pixel grid (3840×2208 for the accepted landscape format).

The UI estimate is **45–55 minutes per 8-second film on the tested RTX 4090**,
based on the [measured trial](../docs/review/bamburgh-refine-details-timing-2026-10-27.md).
This is an estimate, not a guaranteed completion time. Original/output SHA-256,
frame count and frame rate are checked before completion. The user should inspect
faces, motion and chunk boundaries; refinement can introduce texture changes.
Each run retains submitted graphs, chunk timings and a source-bound outcome report.


## Create integration: photographic-grounded-v1

`POST /api/studio/setups/{id}/render` creates canonical jobs. The worker must
advertise `canonical=true` in its heartbeat/claim requests; a legacy photographic
heartbeat cannot enable Create. `TOUCHDOWN_CANONICAL_BUILD_CMD` defaults to
`{python} -m touchdown.canonical_build --input {input} --output {output}`.
This runner prepares fresh photo/portrait inputs, generates the frozen split
flight, derives the regional layout with local Qwen, makes texture and three
native Fill outpaints, projects/finishes the orbit, regenerates the opening and
assembles it at the original frame clock. See the
[canonical specification](../docs/touchdown-canonical-workflow.md) for settings.

Automatic decisions and their raw Qwen reports are saved through authenticated,
artifact-bound checkpoint endpoints. Uploading changed evidence, skipping a
checkpoint, reusing a different decision, claiming completion without all nine
passes, or returning a different video from the final review holds/rejects the
job. Cancellation stops the owned process group. Each Fill pass releases its
runtime before Qwen loads; inference settings match the recorded graph.

The page displays the real stage names and latest check evidence. Review holds
preserve completed stages and findings; they do not fabricate a passing film or
silently fall back to the earlier generator. Human playback of the completed
film remains useful, but routine intermediate user approval is not required.
The historical tests used assistant inspection; replacing that inspection with
Qwen is an additional, explicitly recorded automation layer.

The protected example dropdown labels each method: Bamburgh canonical control,
Eilean grounded-opening variant, Chambord frozen photographic repeat, Peña
photographic control. They are not four proofs of identical generation methods.

Validation includes the API/worker chain with synthetic inference fixtures,
canonical hash/preflight verification and actual local-Qwen checks on saved
control artifacts. This does not by itself demonstrate a fresh arbitrary-place
end-to-end success. The accepted output, manifest and archived inference remain
unchanged. `app-integration-v1.json` records the current integration separately
from acceptance-time metadata in the original `recipe.json`.

### Fresh installs

The canonical worker no longer reads anything under `outputs/`. The frozen flight code and the Prompt Relay
node are vendored in `workflows/touchdown/canonical/photographic-grounded-v1/` (`frozen-code/`,
`relay-runtime/`) and every job's copy is still checked file by file against `recipe.json`. ComfyUI model
locations come from `TOUCHDOWN_MODEL_PATHS` (an `extra_model_paths` file); without it the development
machine's recorded pin is used if present, otherwise ComfyUI's own model folders. Gemma4 remains the LTX 2.5 text encoder. FLUX.1 Fill is needed only for the legacy
`photographic-grounded-v1` opening; default generic-orbit startup skips its model check. Legacy builds and
`--check-control` still verify Fill against `workflows/touchdown/flux-fill-models.json`, caching the hashes
in `models/FLUX-Fill/.verified.json`.
`python -m touchdown.canonical_build --check` checks a fresh install; `--check-control` additionally hashes the
accepted Bamburgh run under `outputs/`, which exists only on the development machine.


### Overhead-guidance-v2 correction (28 October 2026)

Create normalizes its regional layout through `canonical_texture.py` before
freezing terrain inputs. Typed material/water observations replace generated
foreground/background composition instructions; only the flat plan is a texture
image reference. The candidate-only `texture_gate.py` check requires a vertical
map-like view, no sky boundary, no dominant monument and no building side faces.
This named guidance variant fixes the Giza failure mode; it does not change the
frozen flight, portrait, Fill sampler or opening timing. Both the raw proposal
and normalized plan are retained. The original failed job remains unchanged.

The separate Giza recovery still failed opening-motion review. Final checking
now runs `opening_motion_gate.py` before Qwen: confidently stationary surface
features at frames 4 and 8 hold the film even if cloud opacity changes. This
is a rejection check, not proof of physical flight. The experimental stronger
opening pin is not enabled for Create; cross-location motion remains unresolved.
