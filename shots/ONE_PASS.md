# Unattended Touchdown build

Give the build a shot manifest with the person, place, scene, and **one fixed seed** in each of `[opener]`, `[cloud]`, `[swoop]`, and `[end]`. Do not include `pick` or a manually chosen `[end] start` frame. Use `engine = "procedural"` for sound. [`yosemite-one-pass.toml`](yosemite-one-pass.toml) is an example; [`TEMPLATE.toml`](TEMPLATE.toml) documents the fields but lists three seeds for the older candidate workflow.

With ComfyUI and its SDK proxy running, start the whole build with one command from the repository root:

```sh
PYTHONPATH=scripts:src <python-with-opencv> -m touchdown.build shots/yosemite-one-pass.toml \
  --unattended --url <comfy-sdk-proxy-url> --comfy-python <python-with-comfy-sdk>
```

On Jamie's desktop, the OpenCV Python is `/home/jamie/comfyui-venv/bin/python`, the SDK Python is the repository `.venv/bin/python`, and the SDK proxy is `http://127.0.0.1:8191`. Port 8190 is plain ComfyUI and cannot accept SDK jobs.

The command runs stills, opener, dive, cloud, swoop, assembly, living end, and procedural wind without stopping to select candidates. Outputs, `report.md`, and `audit.json` go under `outputs/touchdown/<shot.name>/`. The audit verifies the video format and frame count, checks for an audio stream, records join changes, and flags large frame jumps in the living end. The cache includes the selected upstream seed; changing it regenerates dependent candidates. A cloud render that never reaches whiteout fails the build instead of silently using its last frame. Rerunning an unchanged manifest reuses completed stages.

An optional `[sound.pace]` block retimes the finished picture before the wind bed is generated. `dive_start` is a fraction of the computed dive, `swoop_end` a fraction of the swoop, and `peak` the playback rate at their handoff. The rate eases from 1× to `peak` and back to 1×; it selects existing frames without interpolation or another model run. The Yosemite example uses `peak = 2.4`, shortening its output from 11.29 to 9.96 seconds. It reduces the time spent on flat-looking near-ground imagery but does not fix invented road geometry.

An unattended render is not a guarantee of visual success. The present numeric checks cannot reliably detect a road that bends into an invented junction, terrain that smears, or a figure that changes identity. The audit does flag a sustained frontal-to-profile face turn in the living end when the bundled YuNet detector can establish a frontal starting pose. Judge first-try reliability on a predeclared set of unseen people and places, one seed per stage, counting every failed shot.

The Yosemite run on 11 October is a **visual failure despite zero audit warnings**. At the road handoff, the original swoop prompt makes the camera follow Wawona Road around a bend and then morphs the road into the invented overlook. A controlled rerun with the same seed, dive, and end still, changing only the prompt to demand a straight heading, reduced the turn but made the road become an implausible rock corridor before the person appeared. The late-dive speed ramp hides some flatness but fixes neither transition. Do not treat this example as evidence that one candidate per component is visually reliable.

That failure prompted the measured-terrain route below: a fixed camera path and landing geometry, followed by a FLUX still built from the guide's last view.

## Measured-terrain route

[`yosemite-terrain-one-pass.toml`](yosemite-terrain-one-pass.toml) enables `[terrain]`. The same `touchdown.build --unattended` command now runs this route from a place catalogue entry and a user description. It fetches the catalogued USGS 1 m lidar project and a wider 3DEP DEM, reuses the NAIP imagery pyramid, renders a 64-frame Blender descent and 81-frame guide in one camera scene, and checks that their boundary is identical. FLUX receives the guide's final view as image 1 and an optional `shot.subject` photo as image 2. LTX 2.5 Layout To Render then follows the guide with the FLUX still pinned at its end. The living-end generation and procedural wind follow automatically.

The Yosemite terrain manifest also enables optional `[face]` refinement. A local YuNet detector crops the figure's actual face from the final shot, and five facial landmarks track those pixels onto the blurry approach. A soft blend fades out shortly after the living end begins, so the animated face takes over. The earlier FLUX crop edit remains an optional `source = "flux"` experiment, but it changed the person's identity on this shot; `source = "final_frame"` preserves it. The stage needs no hand-painted mask or manual face coordinates; if no centered face is found, it leaves the clip alone. The digital zoom used in an earlier cut is disabled (`[end] zoom = 1.0`). The bundled YuNet model and its license are in `assets/models/`.

This route currently requires a conterminous-US place with a 1 m DEM project in the catalogue, Blender, the NAIP/USGS source services, and the installed LTX 2.5 Layout To Render IC-LoRA. Supply `terrain.project` and `terrain.utm_zone` if the catalogue has no project. `terrain.target_x` and `terrain.target_y` are optional metre offsets from the catalogue landing. Without them, a deterministic preflight samples safe standing patches within 150 m and approach headings, scoring relief, foreground image changes and the distant DEM drop. `heading = -1` means search headings when the catalogue has no view direction; explicit headings must be 0–359.999 degrees. An explicit target also requires an explicit heading. The computed dive and guide now translate together with the selected landing, so an offset target cannot introduce a sideways camera sweep. The guide records each camera pose in `terrain_guide.json`.

Before LTX, a narrow endpoint gate catches the case where a neutral paved foreground in the measured guide turns into coloured terrain in the FLUX still. On the Mather trial it flags the first road-to-cliff endpoint and accepts the later road-preserving endpoints. This does not establish that every road edge, island outline or landmark is in the right place. The raster preflight cannot recognize cars, roofs or road topology reliably, and the endpoint gate does not certify the scene as geographically accurate. An unattended build stops with a reason when either high-confidence check fails.

Set `embedding_rank = true` in `[terrain]` to render up to `view_previews` cheap final-view thumbnails and four-pose approach contact sheets, then rank the geometry-safe candidates with local [EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers) image–text embeddings. `view_query` describes the desired final setting and `route_query` the appearance of the approach; if omitted, the build derives them from the shot description. The combined score weights the final view 65% and the route 35%. `search_radius_m` can widen the landing search to 280 m. For natural-ground shots, `avoid_pavement = true` checks the last 220 m for sustained pavement and the full 500 m approach in a 180 m-wide swath for large connected smooth surfaces before the ranker runs. It rejects large-surface occupancy above 5% in any quarter of the route, or imagery coverage below 85%. Final previews with more than 70% neutral foreground are also excluded before FLUX, using the same narrow paved-surface heuristic as the endpoint gate. This is a conservative large-pavement heuristic, not a road segmentation model; it may miss small paths or mistake pale, smooth rock for pavement. The model runs in a separate process and exits before FLUX/LTX. The shortlist, final thumbnails, contact sheets and scores are saved as `view_plan.json`, `scene/previews/` and `view_ranking.json` under the terrain stage. Geometry remains a hard constraint: embeddings rank views but cannot supply coordinates or certify that a road continues. The first Mather trial of this option moved the landing from the paved turnabout to measured open ground near the canyon, with no human candidate selection.

EmbeddingGemma 2 is an optional dependency. Set `TOUCHDOWN_EMBED_DEPS` to a directory of isolated Python packages; the build prepends it to the rank subprocess's `PYTHONPATH`, so ComfyUI's environment is unchanged. Set `TOUCHDOWN_EMBED_PYTHON` if this subprocess needs a different Python. Download the weights once with `hf download google/embeddinggemma-2`; subsequent builds set the Hub to offline mode. As of 12 October 2026, the released Transformers on this workstation (5.15.1) does not recognize `embedding_gemma2`; this run used sentence-transformers 6.1.0 and source snapshots of Transformers `2fa1fdd5b27ae54bdcee0516848c7963393a5f5c` and huggingface_hub `96b6f6831f968b7fcb48023df7f0eb3e26405cf8` in an isolated dependency directory. Install a Transformers release with native `embedding_gemma2` support when available; use `bfloat16` or `float32`, not `float16`.

The endpoint check now also measures registered distant landmarks and large newly coloured foreground regions. It rejects both the old Mather cliff endpoint and the later invented island, while accepting the new natural-ground endpoint and the Yosemite endpoint. It still cannot prove that every shrub, path or foreground contour is geographically exact; the low-resolution guide often lacks enough foreground features to register them.

The route does not run a visual judge between stages. It can still produce a bad shot: FLUX may restyle cliffs or invent a wall, LTX may deform cars or ignore the person pin, and the current audit measures only media continuity. Treat each full first-run clip as one result when measuring reliability across unseen people and places.

The first EmbeddingGemma-ranked [Mather Point trial](../docs/review/touchdown-embedding-view-transfer-2026-10-12.md) chose a measured open-ground endpoint at 36.061406, -112.108418, heading 15°. The 10.92-second clip completed with one precommitted seed per component and no stage picks. Its endpoint landmark check passed, and the portrait looks more naturally planted than the paved-turnabout version. It is still a visual failure: the flight crosses a road loop, the guided swoop reshapes a pale rock and road into different terrain, and the living-end figure turns nearly to profile before facing front again. The new face-turn audit flags the last problem; it cannot yet detect the moving-ground substitution. The first attempt also hit a GPU out-of-memory error when another local model service claimed roughly 15 GB mid-render; the completed stages were reused after that service was temporarily stopped. Reserve the GPU for an uninterrupted one-pass trial.

The first integrated [Yosemite terrain run](../docs/review/touchdown-terrain-one-pass-2026-10-12.md) completed unattended in 11.17 seconds of final video with no objective audit warnings. Its guided approach passes a new dissolve/held-frame and end-pin gate. Parked cars, pale road markings and the invented barrier still need a better ground-level treatment; this one configured example does not establish general first-try success.

The [Grand Canyon Mather Point transfer trial](grand-canyon-mather-one-pass.toml) produced a 10.92-second export with sound and no objective audit warnings. It exposed two general pipeline faults: the original 18 m landing search found no acceptable ground at the catalogued coordinate, and the cloud retimer omitted the opaque exit frame, causing a visible jump into the computed dive. The scorer now checks a larger area and surrounding relief; the retimer includes the opaque frame and clears that cloud over the first ten measured-dive frames. The first attempts therefore count as transfer failures, even though the corrected build completes without stage picks. The first Mather endpoint also turned the turnabout island into a cliff-edge overlook. Its revised FLUX endpoint and LTX prompts keep the subject on a sandy island surrounded by pavement through the landing and animate a small head and body movement at the end. LTX still regularizes the island's outline and lets the camera drift slightly in the living end; the numeric audit does not flag either issue.

## To do: data sources

### Commons photo-op preflight

[`mather-photo-op-preflight.toml`](mather-photo-op-preflight.toml) enables `terrain.photo_op = true` alongside `embedding_rank`. Run it with `--unattended --until terrain` first. The build automatically searches Commons by place name and coordinates, downloads an attributed reference pack into `outputs/touchdown/reference_photos/<place>/`, and ranks the images with a plain-language `photo_query`. This mode permits photos of pedestrian areas, people, paths and railings that the older landscape-only downloader excluded. It requires no Commons API key.

Every image retains its Commons page, author, licence, hash and reported coordinates. Only explicit `Location`/`Camera location` templates become geometric camera anchors; object/page coordinates without that evidence remain visual references. Conflicting title and metadata headings are flagged and excluded from bearing calculations. A camera anchor defines a nearby search area, not the exact position of the figure. The best-ranked camera references within 0.04 of the top image similarity constrain the standing-patch search to `photo_anchor_radius_m` (default 10 m, allowed 5–40 m). These similarity thresholds are ranking heuristics, not calibrated confidence.

All landing searches now check the local standing patch and directional approach without a blanket 10/20 m radial-drop rejection or score penalty. The earlier radial rule rejected genuine cliff overlooks and has been removed globally. Photo-op mode uses a smaller local standing patch, while the camera approach checks still apply. For paved pedestrian viewpoints and city plazas, disable `avoid_pavement`; colour neutrality alone cannot distinguish a legitimate platform from a vehicle turnabout. The Mather pilot moved the intended figure to within 8.3 m of the reported camera in a photograph showing the actual pedestrian overlook.

The reference rank, camera anchors and selected support are saved in `photo_op_ranking.json`, `photo_anchors.json` and `view_ranking.json` under the terrain stage. Optional `reference_manifest` reuses a saved photo-op pack and is hashed into the stage cache. The photos currently inform waypoint selection; they are not yet fed to FLUX as matched scene references. Reported photo coordinates have unverified accuracy, and the aerial DEM still omits railings, stairs and viewing-platform structure. This pilot establishes photo-backed discovery and a bounded landing search, not a solved video transition. See the [pilot review](../docs/review/touchdown-commons-photo-op-2026-10-13.md).

Commons is the implemented provider. NPS/NPGallery, Geograph, Mapillary and KartaView remain optional source extensions; no paid photo API is required by this workflow.

- Use [Sparkgeo's geospatial MCP catalog](https://github.com/sparkgeo/geo-mcp-servers) as a discovery index, especially its NASA Earthdata and STAC/Planetary Computer entries. For each promising source, verify that imagery or elevation actually covers a test landing and record ground resolution, acquisition date, DTM versus DSM versus oblique view, download method, and reuse licence. The catalog and its activity checks do not establish these properties.
- Compare any candidate against the NAIP and USGS 1 m/3DEP inputs already used by Touchdown. Prioritize a genuinely better near-ground geometry or texture source, including open oblique or photogrammetric data where available; another wrapper around the same aerial pixels is not a quality improvement.
- Keep the renderer's data access reproducible through direct source URLs, provenance and a local cache. Add an MCP service to the build only if a measured pilot demonstrates a capability the direct provider code lacks.
- Validate usable pixels across the actual landing window before fetching the imagery pyramid or starting model jobs. The Bryce Inspiration Point catalogue listed `UT_StatewideKane_2020_A20` because its 10 km tile exists, but the 400 m landing window was 0% valid in the first unattended run. Tile existence alone is not coverage; the build should preflight DEM and imagery footprints and report a specific alternative when possible.

The [13 October corridor trial](../docs/review/touchdown-corridor-preflight-2026-10-13.md) exercises the full-swath and paved-endpoint preflight. It avoids the previously selected parking corridor but still loses a curved road behind the person during the LTX approach. Its zero audit warnings do not establish visual continuity. The front-facing living-end prompt removed the earlier excessive turn.

### Shared pedestrian surface prototype

`shots/mather-shared-platform.toml` enables `terrain.platform_layout = true` and
`stills.photo_reference = true`. Nearby OpenStreetMap pedestrian ways become
surface meshes in the same Blender scene for both approach and endpoint. The
ranked Commons reference provides material evidence to FLUX; it does not define
a second camera. Exact rendered visibility masks check pavement retention and
edge support in the endpoint before video generation.

This is experimental: default path widths are estimates, steps follow the DEM,
and railings are not reconstructed. The surface check is an endpoint heuristic,
not a certificate of geometry preservation throughout the video. Raw map data
and projected layouts retain OSM attribution; stills facts retain Commons credit.

### Terrain detail and inferred walking surface

`shots/mather-platform-detail.toml` holds the selected Mather target and heading
fixed for a controlled comparison. It reuses the baseline photo ranking file;
run the shared-platform trial first. Rendering retains the 1 m near-source DEM
while the search still uses the 2 m array. Smooth near-terrain normals reduce
lighting facets without fabricating elevation detail.

With `terrain.platform_fit = true`, mapped pedestrian strips near the target use
one inferred walking plane capped
at 5% grade. Height changes fade between 6 and 16 m and cannot exceed 0.6 m.
Faces requiring larger changes are omitted, with counts recorded in
`walking_surface_audit.json`. This is not a reconstructed platform outline.
Small underlying terrain cuts stay inside the strip footprint. No cliff-wide
flattening, invented railings or photo-calibrated dimensions are introduced.

`terrain_resolution_audit.json` records render spacing and texture limitations.
`surface_sequence.json` samples approach frames 0/16/32/48/64/80 against their
matching surface masks. It is diagnostic rather than an unattended pass/fail
gate: appearance and edge thresholds have not been calibrated over moving views.

### Backward conditioning experiment

This is an isolated runner, not a production build stage. It uses the fixed
Mather camera and existing endpoint, eight fixed-seed FLUX edits at frames
77/74/71/68/65/62/59/56, then one fixed-seed LTX pass over an 81-frame guide.
There are no candidate picks or runtime visual judges. The extra frame edits
increase generation cost; this is not a single FLUX inference.

`touchdown.backward_depth` runs inside Blender and caches axial terrain depth
for frames 56..80. It uses the near/far DEM and recorded camera poses. Those
depths do not measure invented railings or the inserted figure, and do not
include the inferred platform fit. The cache records input hashes.

`touchdown.backward_ramp` accepts `--scene`, `--endpoint`, `--depth`, `--output`,
`--comfy-python`, `--url` and `--mode plain|depth`. Run it in the OpenCV Python
with `PYTHONPATH=scripts:src`. The depth mode reprojects the later edited frame
before passing it as the second FLUX reference. Occluded/out-of-frame pixels
are filled from the current render. The plain mode uses the unwarped later
frame as reference. Intermediate guide frames use geometric/flow warps;
finished images are not crossfaded.

DualCFGGuider (regular, cfg_cond2_negative=1) mixes predictions from geometry-only
and geometry-plus-later conditioning. cfg_conds rises from 0.2 at frame 56 to
0.8125 at frame 77; frame 80 is the unchanged endpoint. Image-to-image starts
from encoded rendered geometry. Default denoise rises from 0.6 to 0.81875.
Use `--denoise 1` to hold denoise fixed and test only the conditioning ramp.
Graphs, per-frame parameters and quality diagnostics are retained with outputs.
Pavement/flow checks do not certify scene realism or identity.

`--appearance-only` removes the render reference from the photographic
conditioning branch. Used with `--mode depth --denoise 1`, this tests two
separate regimes: rendered-layout conditioning versus the camera-reprojected
later photograph. The geometry-only branch remains unchanged.

Example depth preparation (inside Blender):

```sh
blender -b -t 4 -noaudio --python scripts/touchdown/backward_depth.py -- \
  --dem <terrain-stage>/dem --poses <terrain-stage>/scene/terrain_guide.json \
  --output outputs/touchdown/backward-ramp/depth
```

Example fixed-seed generation (from the repository root):

```sh
PYTHONPATH=scripts:src /home/jamie/comfyui-venv/bin/python -m touchdown.backward_ramp \
  --scene <terrain-stage>/scene --endpoint <stills-stage>/end.png \
  --depth outputs/touchdown/backward-ramp/depth \
  --output outputs/touchdown/backward-ramp/separate-regimes \
  --comfy-python /home/jamie/Projects_Live/ComfyUI_Node/.venv/bin/python \
  --mode depth --denoise 1 --appearance-only
```

The same `--appearance-only --denoise 1` test can use `--mode plain` to
propagate the unwarped photograph, avoiding terrain-depth seams but relying on
the conditioning ramp and LTX for viewpoint adaptation. Give each configuration
a separate output folder.

### Bounded continuity alternatives

`python -m touchdown.continuity_trials --route retreat|portrait|clay` runs one
fixed-seed configuration without candidate selection. Supply `--endpoint`,
`--output`, `--comfy-python`, and optionally `--url`. Clay also requires
`--terrain-stage` and `--imagery`. Use the OpenCV Python and `PYTHONPATH=scripts:src`.

Retreat generates 73 frames away from the endpoint and reverses them. Portrait
constructs a one-metre 2.5D dolly and sends it through LTX. Its colour/GrabCut
extractor is specific to the central mustard jacket in this experiment. Clay
adds an untextured terrain guide and stationary humanoid proxy, makes photographic
first/end references and runs an 81-frame LTX pass. The first appearance reference
must describe the starting viewpoint; supplying grey clay retains the render look,
and supplying an unrelated eye-level endpoint can hold that composition.

These are experiments, not production defaults. See
`docs/review/touchdown-continuity-alternatives-2026-10-13.md` for defects and the
measured guide/endpoint foot-placement mismatch.

### Reversed-retreat full-shot join prototype

The normal terrain route remains unchanged. For a bounded integration trial,
`terrain_blender.py --retreat-entry <json>` renders a standalone computed descent
to a specified standoff. `timing = "accelerate"` increases fractional approach
speed; `look_at_waypoint = true` holds the selected target and heading.

`python -m touchdown.retreat_join --prefix <cached-body> --prefix-frames 53
--descent <computed.mp4> --landing <reversed.mp4> --living <49-frame-end.mp4>
--output <folder> --mist` assembles the Mather prototype and adds procedural wind.
The landing stays at its native timing; nine living-prefix frames are skipped.
The mist fully covers the scene cut and does not establish shared geometry.
See `docs/review/touchdown-retreat-integration-2026-10-13.md` for exact inputs,
known limitations and the direct-cut control. This is not a generic scene workflow.
