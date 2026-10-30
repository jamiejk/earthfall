# The pipeline's ComfyUI graphs

For the Earthfall recipe integrated into Create, start with [photographic-grounded-v1](touchdown-canonical-workflow.md). The graph catalog below covers other build/library paths, including Layout To Render; those paths are not the canonical photographic recipe. Its actual submitted graphs are bound in the [canonical manifest](../workflows/touchdown/canonical/photographic-grounded-v1/recipe.json).

What the Touchdown build sends to ComfyUI, exported as API-format workflow JSON in
[`examples/comfyui-workflows/`](../examples/comfyui-workflows/). Every file there is written by
`scripts/export_workflows.py`, which calls the code the build calls, so an exported graph cannot drift from
what a run submits.

```bash
uv run python scripts/export_workflows.py                        # writes examples/comfyui-workflows/
uv run python scripts/export_workflows.py --out /tmp/graphs      # somewhere else
```

The exported files have the shape of ComfyUI's "Save (API Format)": a dict of node id to
`{"class_type": ..., "inputs": ..., "_meta": {"title": ...}}`. That is what the pipeline submits to
`POST /prompt` (through the Comfy SDK, `comfy_sdk`), and the ComfyUI frontend opens it as well: drop a file
on the canvas and it converts the API format to the canvas format, the same way
[`examples/grand_canyon_blocking_api.json`](../examples/grand_canyon_blocking_api.json) is used in the
README. Every node carries a title naming the graph it belongs to and what the node does, so a file opened
on its own says what it is.

`index.json` in the same folder lists every file with the stage it belongs to, the function that built it,
the ComfyUI version and the custom nodes it needs, the model files it asks for and the input files it
expects.

## The files

| File | Stage | Nodes | What it is |
|---|---|---|---|
| `geoblock-blocking-earth-zoom.json` | terrain | 7 | Place → landform scene → camera preset → Blender blocking video |
| `geoblock-ltx25-earth-zoom.json` | whole shot | 272 | The same, plus the photoreal keyframes and LTX 2.5 Layout To Render, in one graph |
| `touchdown-stills-flux3.json` | stills | 36 | The four stills (end, orbit, cloud, opener end) painted by FLUX 3 Image Klein at 2K |
| `touchdown-stills-klein.json` | stills | 58 | The same four stills painted by FLUX.2 [klein] 9B distilled |
| `touchdown-flux3-still.json` | stills | 13 | One FLUX 3 still, as `scripts/touchdown/flux3.py` submits it over plain HTTP |
| `touchdown-ltx25-clip.json` | opener, cloud, swoop, end | 36 | LTX 2.5 keyframe image-to-video with the stills pinned as guides |
| `touchdown-terrain-v2v.json` | swoop | 37 | The measured guide video in, the LTX 2.5 Layout To Render approach out |
| `touchdown-face-plate.json` | face | 18 | The face stage: one klein edit into a detailed face reference |

The two `geoblock-*` files are the GeoBlock library's single-graph path (`geoblock-run`), not what the
Touchdown build submits: the build splits the work into stages so each one can be cached, checked and
picked, and it renders the blocking video in Blender itself. They are here because the pipeline is that
graph, spread over ten stages, and because one file that goes all the way from survey data to a rendered
clip is the clearest picture of what the model is asked to do.

## The graphs, one by one

### Touchdown stills — `touchdown-stills-flux3.json`, `touchdown-stills-klein.json`

The stills stage paints the four pictures the rest of the shot is pinned to: the end still the swoop lands
on, the orbit still the shot starts from, the whiteout still the camera dives through, and the separate
opener end pin. All four go in one submit, so the image model is loaded once.

* Built by `touchdown.comfy_job.stills`: the same call both times, with the `flux3` list naming which
  stills come from FLUX 3 (all four in the first file) and which from FLUX.2 [klein] (all four in the
  second).
* Ids: `41`, `42`, `43` are the reference images (the person's photo, the guide's last frame, the place
  photograph); `100`+ is the end still, `120`+ the orbit still, `140`+ the cloud still, `160`+ the opener end;
  the save node of each is `107`, `127`, `147`, `167`. In the klein graph `1`, `2`, `3` are the FLUX model,
  text encoder and VAE; the FLUX 3 graph loads its weights as `f3u`, `f3c`, `f3v`.
* Models: `flux3-image-klein-bf16.safetensors` and `qwen3vl_4b_bf16.safetensors` (FLUX 3), or
  `flux-2-klein-9b-fp8.safetensors` and `qwen_3_8b_fp8mixed.safetensors` (FLUX.2), and
  `flux2-vae.safetensors` for both.
* ComfyUI: v0.41.3+ for the FLUX 3 stills (the model needs it), v0.32+ for the FLUX.2 ones. No custom
  nodes: the FLUX 2 and FLUX 3 nodes are core.
* To run it: put the three reference images in ComfyUI's `input/` folder under the names in the graph
  (`guide_frame_0081.png`, `subject.jpg`, `scene.jpg`), or change the `LoadImage` nodes to your own files.

### The FLUX 3 still on its own — `touchdown-flux3-still.json`

One FLUX 3 Image Klein still at 1344×768 (16 steps, no guidance) with two reference images, as
`scripts/touchdown/flux3.py` submits it: that module talks to a plain ComfyUI HTTP API, without the SDK
proxy, and the build uses it for the stills on a second server (see `[stills] flux3_url` in a manifest).
Ids `1`-`4` are the model, encoder, VAE and the flux sampling shift; `r0`/`r1` are the two reference images;
`10`/`11` are the prompt and the empty negative; `12` the reference latent; `13` the empty latent, `14` the
sample, `15` the decode and `16` the save.

### The LTX 2.5 clip — `touchdown-ltx25-clip.json`

Plain LTX 2.5 keyframe image-to-video: the stills are pinned as guides and the model invents the move
between them. The opener, the cloud passage, the swoop and the living end are this graph with different
pins (`scripts/touchdown/build.py` calls `comfy_job.ltx` for all four).

* Ids: `44`, `45`, `46` are the pinned images; `300` is the transformer, `302` the video VAE, `304` the text
  encoder, `305`/`306` the prompt and the negative, `307` the conditioning, `308` the empty latent;
  `312`-`314` are the guides on stage 1, `342`-`344` the same guides again on the upscaled stage 2; `390`-`395`
  are the sampler and the guide crop; `330`/`331` the 2x latent upscale; `375`-`377` the half-size stage 1
  saved for diagnosis; `370`, `371`, `90` the decode, the video and the save.
* Models: `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`,
  `ltx-2.5-video-vae-bf16.safetensors`,
  `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` (loaded on the CPU: it is 15.4 GB and the
  transformer needs the card) and `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` for the upscale.
* ComfyUI: v0.32+ with [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo). See
  [LTX 2.5 Layout To Render and FLUX.2 [klein] keyframes](ltx25-flux-models.md) for the weights, the
  licences and the memory notes.

### The measured-terrain swoop — `touchdown-terrain-v2v.json`

The stage where the measured route stops being a guide and becomes the shot: the Blender guide video goes
in as the Layout To Render IC-LoRA guide at frame 0, the guide's first frame as a one-frame reference at
RoPE position -1, the end still is pinned as the last frame, and LTX 2.5 renders the approach over the real
terrain.

* Ids: `50` is the guide video, `60` its components, `40`-`43` the first and last stills and their
  preprocessed copies; `300`-`308` the models and the conditioning; `310`/`311` the two IC-LoRA guides and
  `312` the pinned end still; `390`-`395` stage 1; `330`/`331` the 2x latent upscale; `340`-`342` the same
  guides again at full size on stage 2; `360`-`365` the refine; `370`, `371`, `90` the decode and the save.
* Models: the LTX 2.5 set above plus
  `ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors`, loaded by `301` (`LTXICLoRALoaderModelOnly`).
* ComfyUI: v0.32+ with ComfyUI-LTXVideo. The two guide node classes (`LTXAddVideoICLoRAGuide`,
  `LTXICLoRALoaderModelOnly`) are the ones that pack has to provide.

### The face plate — `touchdown-face-plate.json`

The optional face stage: the blurry face in the late approach is edited into a detailed, square face
reference (`768×768`), which is then tracked onto the living end. One klein edit of one reference image, so
it is the smallest graph here.

* Ids: `42` is the reference crop, `100` the prompt, `102`-`106` the latent, the scheduler, the guider, the
  sample and the decode, `107` the save.
* Models: `flux-2-klein-9b-fp8.safetensors`, `qwen_3_8b_fp8mixed.safetensors`, `flux2-vae.safetensors`.
* ComfyUI: v0.32+; no custom nodes.

### The blocking graph — `geoblock-blocking-earth-zoom.json`

The measured part of the shot, and the only part of it that is not a picture model: the place becomes a
landform scene built from survey data, the camera preset plans a route over it, and Blender renders the
blocking video that the generative stages follow.

* Ids: `2` the place, `3` the landform scene, `4` the camera preset, `5` the Blender render, `20` the saved
  control video, `21` the route map, `22` the opening frame.
* Models: none. It needs the GeoBlock nodes (`comfy_geoblock_prototype/` in this repository, linked into
  ComfyUI's `custom_nodes/`) and Blender 4.2+ on `PATH`.

### The whole library graph — `geoblock-ltx25-earth-zoom.json`

The same blocking, plus the photoreal keyframes painted from it (FLUX.2 [klein] 9B, one per guide point of
the route) and LTX 2.5 Layout To Render following the blocking, in one submit: 272 nodes.

* Ids: `1`-`5` the blocking, `60` the blocking video as components, `61`-`63` the klein model, encoder and
  VAE, `66` and `74` the first and last keyframe painters, `80`-`99` their saves, `170`-`173` the subject and
  its sheet, `200`+ the mid keyframes (eight nodes each), `300`-`308` the LTX 2.5 models and conditioning,
  `310`-`319` the IC-LoRA guides and the pinned keyframes, `330`/`331` the upscale, `340`-`342` the guides
  again on stage 2, `360`-`371` the refine and the decode, `380` the colour smoothing.
* The last keyframe is pinned here (`pin_last_keyframe=True`, the library's `--pin-ending`). The Earth zoom
  otherwise leaves the last seconds to the layout, and the last keyframe is painted and saved but never
  used.
* ComfyUI: v0.32+ with ComfyUI-LTXVideo, and the GeoBlock nodes for the blocking half.

## The node ids

The ids are fixed so the pipeline can address outputs by name, and they are documented at the top of
[`src/geoblock_core/workflow.py`](../src/geoblock_core/workflow.py) (lines 18-29):

| Ids | What |
|---|---|
| `1`-`5` | The place, the landform scene, the camera preset, the Blender render |
| `20`-`23` | The saved outputs: control video, route map, opening frame, depth video |
| `31`-`38` | The place photos (Wikimedia Commons) |
| `41`-`59` | The H3 stage (Ref2VA) and its reference images |
| `60` | The blocking video, as components the image models can read |
| `61`-`65` | The keyframe model, encoder and VAE |
| `66`, `74` | The keyframe painter for the first and the last frame |
| `80`-`99` | The keyframe saves: the last frame, then the mid keyframes |
| `100`-`119` | LTX 2.3 with 3DREAL (the older render-to-real stage) |
| `120`-`130` | Its 2x latent upscale and refine |
| `140`-`158` | The mid keyframes as guides (LTX 2.3): `140`+ on the half-size latent, `150`+ again on the upscaled one |
| `170`-`173` | The subject, its photo and the subject sheet |
| `198`-`199` | The blocking's first frame, pinned as frame 0 |
| `200`+ | The mid keyframes, eight nodes each (the prompt, the painter, the save) |
| `300`-`308` | The LTX 2.5 models, the conditioning and the empty latent |
| `310`-`319` | The Layout To Render guides and the pinned keyframes on stage 1 |
| `330`-`331` | The 2x latent upscaler |
| `340`-`342` | The same guides again, at full size, on stage 2 |
| `360`-`371` | The refine, the decode and the video |
| `375`-`377` | Stage 1 decoded on its own, for diagnosis (`--save-stage1`) |
| `380`-`381` | The colour smoothing and the cut through the cloud deck |

The Touchdown graphs use the same idea with their own ranges: `1`-`3` are the FLUX weights, `40`-`46` the
loaded images, `100`/`120`/`140`/`160` the four stills, `50`/`60` the guide video, and `300`-`377` the LTX
2.5 stage, which is the same code in both pipelines.

## How to see the whole thing

The build is [`scripts/touchdown/build.py`](../scripts/touchdown/build.py). Its stages are the constant
`STAGES` (line 60):

```
terrain  stills  opener  dive  cloud  swoop  body  end  face  sound
```

- **terrain**: measured relief and NAIP imagery, and the Blender guide video with the one shared camera
  path. No model. The guide video is the input to the swoop.
- **stills**: the four FLUX stills (`touchdown-stills-*.json`).
- **opener**: LTX 2.5 from the orbit still into the cloud (`touchdown-ltx25-clip.json`).
- **dive**: no model. A dive over the real NAIP aerial photos, ending pitching forward.
- **cloud**: the plunge into the cloud, retimed to keep its speed.
- **swoop**: the approach to the end still. With the measured route, LTX 2.5 Layout To Render over the guide
  (`touchdown-terrain-v2v.json`); without it, another LTX 2.5 clip.
- **body**: joins only. Colour and brightness ramps, speed match, flow retime, seam softening, shake.
- **end**: the living end, another LTX 2.5 clip.
- **face**: the optional face patch (`touchdown-face-plate.json`).
- **sound**: procedural wind, or LTX 2.3 Foley V2A windows.

Each stage's output is cached under `<output>/<shot>/<stage>/<key>/`, where the key hashes the stage's
settings, the keys of the stages it uses and the stage's own code: changing anything upstream re-runs
everything downstream of it and nothing else. A generated stage lists candidate seeds, and the build stops
so a person can pick one unless `--unattended` is given.

The app that drives it is [`app/`](../app/): a plain HTML front end and a FastAPI server, plus a worker that
writes the manifest and runs `python -m touchdown.build <manifest> --url <ComfyUI>` on the GPU machine. The
server generates no images itself; the app is the queue and the manifest.

## Inputs the graphs expect

The exported graphs name their input files, but the files themselves are not in the repository: the build
uploads the real ones as SDK assets, and the graph refers to them by the name the server knows them under.
The placeholders used in the export are:

| Name | What it stands for |
|---|---|
| `guide_frame_0081.png` | The guide's last frame, the reference the end still is painted from |
| `guide_frame_0001.png` | The guide's first frame, the appearance reference of the swoop |
| `guide.mp4` | The Blender guide video of the measured camera path |
| `subject.jpg` | The photo of the person the shot lands on |
| `scene.jpg` | A photograph of the place, used as a material reference |
| `orbit.png`, `cloud.png`, `opener_end.png`, `end.png` | The stills the other stages pin |
| `approach_frame.png` | The late-approach crop the face plate is edited from |

To run a graph, put files with those names in ComfyUI's `input/` folder, or point the `LoadImage` and
`LoadVideo` nodes at your own.

## Tests

[`tests/test_export_workflows.py`](../tests/test_export_workflows.py) checks that the export runs without a
server, a GPU or a network, that every node has a class type, that every link points at a node that
exists, that no node is left orphaned, and that the files in `examples/comfyui-workflows/` are what the
export writes today.
