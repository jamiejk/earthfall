# Earthfall canonical workflow: photographic-grounded-v1

**Accepted working recipe, 27 October 2026.** Jamie accepted the complete Bamburgh Castle result: “This is good for now, let's write this up as the canonical workflow.” Use this recipe as the starting point for subsequent Touchdown flight work. Preserve the accepted result and compare any proposed changes against it.

The intended visible sequence is a forward rush toward Earth, disappearance of the curved horizon above the frame, passage through substantial clouds, a connected approach toward the landmark, and arrival at the person with their native moving portrait ending. The orbital reference is built **from the generated destination landscape**, so its broad geography agrees with the view below the clouds.

Create is now integrated with these canonical generation stages through `touchdown.canonical_build`. The historical control was carried out by the assistant, including layout authoring and intermediate inspection; it did not require Jamie to approve each stage. The app replaces those assistant actions with recorded local Qwen layout/review steps, so normal generation runs without user approval prompts. This automation is an additional integration layer, not evidence that arbitrary-location reliability has been established. The older strict phase/30,000-foot experiment is outside this version: there is no additional required 30,000-foot pin or newly imposed eight-second trim.

## Generic orbit opening — 28 October 2026

Create's default recipe is now `photographic-generic-orbit-v1`, stages references → flight → opening → review.
The opening no longer depicts the destination from orbit, so the layout, texture, three outpaints, projection and
orbit finish (and their six checkpoints) are skipped. `scripts/touchdown/generic_opening.py`:

| Step | Setting |
|---|---|
| Join | Among flight frames with < 1% blue sky in the top quarter, ≥ 97% cloud overall and in the top band, and J…J+8 all ≥ 90% cloud, the brightest and least textured (densest white) within 12 frames of the first, keeping the film 8.1–9.1 s; none holds the job |
| Flight zoom at J | Median ECC affine scale over J…J+4 (high-passed 640×368 greys), log rate clamped to 0.04–0.2 |
| Opening mode | `camera` (default): a globe-only proxy, no cloud added, at Layout To Render 0.3. `layers` (the cumulus-shell proxy below, 0.5) and `keyframes` (the earlier canonical pins and Prompt Relay, no proxy) are selectable with `TOUCHDOWN_OPENING_MODE` |
| Camera (`camera`) | Zoom 2.5–3.5% per frame to play 16, accelerating smoothly to a peak by play 34 and easing to the flight's rate at J; 12× in all |
| Stills | `assets/touchdown/opening/generic-orbit.png` and `cumulus.png` (`layers` only), FLUX 3 Klein, text only, seed 2081561606 (prompts and graphs beside them); `limb-fit.json` fitted once |
| Proxy (`layers`) | 49 frames; orbit still on a sphere, approached at 0.016 log/frame for play 0–28 (~1.2 s, as the example films), rising smoothly to the flight's rate by play 34; two cumulus shells and a cumulus deck from crops of `cumulus.png` close over play ~27–33; the deck eases into flight frame J over the last 14 moving frames |
| Proxy checks | `camera`: no flat white frame, ground visible ≥ 4% through play 20. `layers`: no flat white frame; last-16 mean grey difference below max(10.5, 1.25 × the pinned flight frames' own largest step); ground or limb visible (≥ 10%) through play 24; otherwise held |
| Generation | Frozen `split_flight._generate_layout`: proxy as the Layout To Render video at 0.3 (`camera`) or 0.5 (`layers`), flight J+8…J at frame 0 (strength 1), generic orbit as look reference and end pin; scene seed + 32 |
| Assembly | Generated 48…9 reversed, then flight J…end; no crossfade or retiming |
| Review | Opening motion gate, then Qwen final review |

Tested on Giza (v10, film 214 frames) with a 16-frame orbit: smooth dive, join difference ~11; the orbit was then lengthened to ~28 frames.
On Taj Mahal (28 Oct) the `layers` proxy showed three cloud phases (orbit clouds, a fade into milky streaky layers,
white-out); `camera` with the densest-white join gave one bank growing into the flight's white cloud: orbit ~22 frames,
zoom 2.4% per frame rising to 11%, 2.6× by frame 28, cut difference 9.9 against the flight's own 5–10. Known limits:
LTX smooths the cumulus into waxy shapes in the middle of the dive, and a mild brightness step remains at the cut.

## Overhead guidance correction — 28 October 2026

The first Giza Create run exposed two failures: a missing import stopped the
first Fill pass, and an oblique pyramid image incorrectly passed the overhead
texture check. The import is fixed and covered by actual-wrapper transport tests.

Current Create uses the named **overhead-guidance-v2** variant for regional
normalization, texture and its camera check. `canonical_texture.py` extracts
only typed surface/water observations, builds fixed material descriptions and
passes the flat plan as the texture's sole image reference. It excludes the
oblique photograph and model-written foreground/background composition from
texture generation. `texture_gate.py` grades only the candidate texture using
explicit camera/content observations and a deterministic acceptance rule.
The normalized layout and original raw layout/model response are both retained.

This changes the texture guidance and review layer relative to the accepted
control; it is not described as a byte-identical canonical replay. Flight,
portrait, Fill sampling, globe mapping and opening timing remain unchanged.
See the [Giza recovery record](review/giza-overhead-recovery-2026-10-28.md).

The Giza recovery's opening was subsequently rejected: clouds obscure a
stationary Earth surface. Its nominal opening settings match Bamburgh, but
that has not established reliable camera motion across locations. The final
review now includes `opening_motion_gate.py`, a feature-tracking check that
rejects a confidently stationary early surface before Qwen can approve it.
This check never grants a motion pass; expansion alone cannot establish
physical parallax or exclude morphing. An opening-only strength-1.0 experiment
also failed visual review and is not a Create default. See the
[opening regression investigation](review/giza-opening-regression-2026-10-28.md).

## Integrated Create path — 27 October 2026

The production route is **saved setup → `photographic-grounded-v1` queue job →
canonical-capable worker → `touchdown.canonical_build`**. It does not dispatch a
new Create job to the earlier `touchdown.studio_build` complete-build function.
The latter remains available for historical jobs and supplies shared upstream
helpers. Old photographic workers cannot claim canonical jobs or advertise
Create availability.

| App stage | Implementation and behavior |
| --- | --- |
| Fresh inputs | `canonical_stage.py prepare` uses the saved photographs, portrait, ending and framing; creates a Qwen-grounded arrival, checks ground contact, creates regional stills and a native moving portrait. `canonical_scene.py` retains the photographic scene prompts without the strict phase/30,000-foot override. |
| Frozen split flight | `canonical_stage.py flight` loads an isolated copy of the accepted archived code and exact `flight-preset.json`; verifies the cloned inference files against the accepted control, then generates parent, repaired approach anchor and descent. The Qwen approach gate remains active. It does not require a pre-existing film of the new destination. |
| Approximate regional plan | `canonical_review.py` assesses the canonical cloud-exit candidate, frame 64, and authors simple land/water polygons, destination UV and continuation descriptions. This is a creative drawing inspired by the visible landscape, not surveyed map reconstruction. Invalid or unreadable geometry holds the take. |
| Texture / outpaints / Earth | Existing recorded `scene_grounded_workflow.py` stages create texture, project the expanded guide and finish the orbit. `canonical_fill.py` runs the same native Fill graph/prompt/settings once per expansion, releasing the GPU runtime before each Qwen check. All three passes require recorded passing decisions. |
| Grounded opening / join | The isolated stage driver uses the exact canonical opening timing, nine reversed cloud frames, intermediate strength 0.5 and 40-frame replacement. Remaining source frames keep their own order and timing; no new fixed overall duration or 30,000-foot requirement is imposed. |
| Review / delivery | Nine source-bound checkpoints cover flight, layout, texture, each of three outpaints, projection, orbit and final film. Decisions run automatically. Failed or uncertain findings hold the job and preserve evidence. Completion requires every decision and a final review bound to the uploaded video hash. |

Enable the GPU worker with `TOUCHDOWN_CANONICAL=1` and the existing local runtime
configuration. `python -m touchdown.canonical_build --check` verifies the 302
control hashes, archived preset, model verification records and required local
dependencies before the worker advertises support. Local-only archive/model
paths remain dependencies; the checkout alone is not a portable installation.

The job page shows the actual canonical stages and latest vision evidence.
An accepted completed film retains the optional **Refine Details · 4K** action;
that is a separate child job and never replaces the original.

The integration is covered by API/worker tests with explicitly synthetic model
outputs and local-Qwen replay checks on saved control artifacts. These checks
must be distinguished from a newly generated end-to-end film through Create.
A fresh arbitrary-place success is not inferred from transport tests or replay.
Qwen checks use sampled images and cannot certify camera motion, identity or
geographic accuracy; inspect the delivered film before sharing.

The immutable `recipe.json` records acceptance-time state, including
`app_default_changed: false`. That historical field is not the current app
configuration. Current integration is recorded separately in
[`app-integration-v1.json`](../workflows/touchdown/canonical/photographic-grounded-v1/app-integration-v1.json).
The operator replay commands below remain available for reproducing the original
assistant-reviewed control; their decision files are not routine user tasks in Create.

## Accepted control and source of truth

| Record | Location |
| --- | --- |
| Accepted complete film | `outputs/touchdown/studio/studio-bamburgh-complete-grounded-v1-20261027/complete-flight.mp4` |
| Accepted run directory (called `BASE` below) | `outputs/touchdown/studio/studio-bamburgh-complete-grounded-v1-20261027` |
| Machine-readable canonical recipe and evidence hashes | [`recipe.json`](../workflows/touchdown/canonical/photographic-grounded-v1/recipe.json) |
| Stage runner | [`scene_grounded_workflow.py`](../scripts/touchdown/scene_grounded_workflow.py) |
| Flight preparation/worker | [`photographic_transfer.py`](../scripts/touchdown/photographic_transfer.py), using archived `photographic_recipe.py` inference |
| Detailed case report | [Bamburgh review](review/bamburgh-complete-grounded-workflow-2026-10-27.md) |
| Shared flight preset / opening prompts | [`flight-preset.json`](../workflows/touchdown/canonical/photographic-grounded-v1/flight-preset.json), [`opening-timing.json`](../workflows/touchdown/canonical/photographic-grounded-v1/opening-timing.json) |
| Example authored geography | [`scene-layout.bamburgh.json`](../workflows/touchdown/canonical/photographic-grounded-v1/scene-layout.bamburgh.json) |

The case contains nine fresh generation jobs, recorded in `BASE/job-ledger.json`: parent, repaired anchor, descent, regional texture, three outpaints, orbital finish and opening. Photographic scene inputs and the native moving portrait were reused. Generating new source photographs/portraits is an upstream preparation step, not something this nine-job test validated.

The complete accepted film is **195 frames at 24 fps, 8.125 seconds, 2560×1472**. Its SHA-256 is in the canonical manifest. A source flight's historical seam registration and hold trimming can produce a different frame count in another case. Preserve that case's actual frame clock during opening replacement; do not silently pad, trim, interpolate or retime it to match Bamburgh.

## Fixed stage order and settings

### 1. Fresh photographic split flight

Start from a prepared case containing its scene configuration, integrated arrival/high/orbit stills, location and subject references, and native moving portrait. Record source paths, hashes, selection reason, scene seed and reuse scope before generation. The accepted case uses seed **2081561474**.

Use the archived `photographic-flight-v1` code/preset, not whatever the live app happens to select. Generate the retreat backward/upward and reverse for delivery. Parent: **241 frames**, including **48 orbital handles**; original opening target **72** frames before historical seam registration. Regional guide: delivery target **64**, generation frame **128**, strength **0.5**. FLUX composition repair: approach anchor at delivery target **120**, strength **0.5**. Separate descent: **121 frames**, with the first **25 native portrait frames reversed** as motion context. Preserve the native ending and archived assembly/crop/hold-trimming behavior.

Flight inference is **1280×736, 24 fps**, LTX 2.5 distilled with the archived Prompt Relay token-span conditioning. Parent seed is scene seed **+32**, descent **+33**, repaired anchor **+502**. No Layout/IC-LoRA, colored/clay viewport video, moving high-photo crop, photographic pin ladder, seam carry, seam bridge or latent 2× refinement is enabled. The preset contains dormant settings for some of these features; their presence does not enable them.

Review the actual fresh flight before proceeding. It must have a usable cloud passage, readable landmark approach and connected landing. A technically completed job or advisory vision score alone is insufficient.

### 2. Regional reference grounded in that flight

Choose a clear **fresh-flight cloud-exit frame**, then author a simple straight-down land/water plan of its broad geometry. Record frame index, polygons, colors, target UV and geographic descriptions in `scene-layout.json`. Bamburgh uses frame **64 / 2.667 seconds**, target **(0.49, 0.60)** and a connected promontory, with sea left/foreground and mainland right.

In the recorded control this was an assistant-authored scene-specific step. Create now derives this per-case plan with Qwen. Follow the landscape in the generated film, including its approximation, rather than claiming surveyed geographic registration. For another landmark, change the layout, region, target and descriptions to that case; do not reuse Bamburgh's coastline or the earlier Eilean islands. Once `terrain-prepare` freezes the inputs, alterations require a new experiment root.

FLUX creates a **1920×1088** nadir texture using the flat plan as reference 1 and the cloud-exit photograph as reference 2 for natural appearance only. Seed **scene +1**, **16 steps, Euler/simple, CFG 1, denoise 1**. Review for preserved broad geometry, straight-down view, and absence of an oblique scene/giant landmark.

### 3. Three native FLUX Fill outpaints

Expand the overhead texture **2× in each axis**, three times: **8× per axis overall**. Each job shrinks its input to **960×544**, pads **480 pixels left/right and 272 top/bottom**, and outputs **1920×1088**. Feather **24**, `noise_mask=false`; native center pixels are not pasted back or claimed exact.

Use native `flux1-fill-dev.safetensors`, `clip_l.safetensors`, `t5xxl_fp16.safetensors`, `ae.safetensors`; Comfy casts the official diffusion weights to **fp8_e4m3fn** for this GPU. Every pass uses seed **1399629552**, **20 steps, Euler/normal, CFG 1, FLUX guidance 30, denoise 1**. Use the runner's scene-specific overhead-continuation prompt, not the old example prompt embedded in the generic Fill template.

Each expansion requires a recorded visual decision before the next submission. The control used assistant inspection; Create records automatic Qwen decisions after releasing the generation runtime. Check broad land/water continuity, target ancestry and invented features. The accepted run developed text-like labels in these intermediate guides; its orbital finish removed them. Record defects rather than calling the raw outpaints clean photographs. If the final orbital image retains such artifacts, hold it rather than passing it into LTX.

### 4. Curved Earth guide and orbital finish

Project the third expansion using the accepted bounded mapping: unit sphere, camera height **0.08 radii**, pitch **45°**, horizontal FOV **80°**, heading **+Y**. Expanded target is `0.5 + (original_target - 0.5)/8`.

```text
u = target_x + 0.48 * tanh(longitude / 0.075)
v = target_y - 0.48 * tanh((latitude - centre_latitude) / 0.075)
```

This is a bounded **nonuniform artistic projection**, not a uniform ground scale or real orbital/geographic registration. Check the target diagnostic, texture coverage, curved horizon and narrow space band. Do not substitute the earlier failed uniform-angle projection.

FLUX finishes the guide as an orbital photograph: guide as the sole image reference, **1920×1088**, scene seed **+2**, same **16-step Euler/simple, CFG 1** still recipe. Preserve coastline, horizon, camera direction and relative land/water sizes; add atmosphere and restrained weather. Inspect for visible text, CGI appearance, geography drift, giant landmarks and cloud concealment before accepting the image.

### 5. Grounded opening, generated in reverse

Prepare inputs from this case, not a previous location:

| Input / setting | Fixed value |
| --- | --- |
| Opening generation | **49 frames, 1280×736, 24 fps**, scene seed **+32** |
| Moving cloud context | Fresh flight frames **40–48**, reversed to **48…40**; nine frames loaded at generated frame **0**, strength **1.0** |
| Intermediate Earth guide | New orbit image, centered **1.8×** closer crop, at generated frame **40**, strength **0.5** |
| Orbital endpoint | New orbit image, center aspect crop to 1280:736, frame **-1 / 48**, strength **1.0** |
| Prompt conditioning | Exact archived `opening-timing.json`, active Prompt Relay, guide positions **[0, 4.5, 40, 48]** |
| Sampler | **Euler ancestral, CFG 1** |
| Manual sigmas | **1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0** |

Generation travels backward/upward: moving cloud retreat → rapid cloud exit → continued planet retreat. Reversing it gives the forward rush. Retain the exact prompts and phase spans from the JSON; do not summarize them into a new prompt. **Keep the intermediate strength at 0.5.** The earlier 1.0 experiment introduced a jump/hold.

LTX models: `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`, `ltx-2.5-video-vae-bf16.safetensors`, `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` (text encoder on CPU). FLUX stills use the selected case's recorded model configuration; Bamburgh uses `flux3-image-klein-int8-convrot.safetensors`, `qwen3vl_4b_int8_convrot.safetensors`, `flux2-vae.safetensors`. Exact graph/weight/backend records take precedence over family names.

### 6. Assembly and review

Use generated frames **48…9** as delivery frames **0–39**, then append fresh source frames **40…end** at their existing positions. Scale the new opening to the source delivery size. Export H.264, CRF **16**, `yuv420p`, faststart. No crossfade, extra interpolation, retiming or inserted freeze is added. Later source frames are re-encoded, so encoded/pixel identity is not promised.

Inspect dense first-second samples, every frame around the **39→40** cloud join, the post-cloud view, castle approach and subject ending. Confirm an advancing viewpoint with the horizon leaving the frame, not only clouds drifting or opacity transitions. Check actual count/rate/duration, source-frame order, graph guide strengths, Prompt Relay, absence of Layout, hashes and job ledger. Save the whole-film board and known limitations. A seamless edit does not by itself prove physical camera travel.

## Recorded local execution

Run from the repository root with its `.venv`. The installed local ComfyUI backend, SDK proxy and vision checker, existing model files, archived Peña code tree and captured upstream relay are dependencies. Runtime settings are loaded by the workers from the existing private `/tmp/touchdown-generation-runtime.env` (JSON-encoded `TOUCHDOWN_*` values); do not print or commit it. The owned runtime uses the shared GPU lock, deterministic mode and explicit model paths, refuses an occupied generation port and stops only its own server.

The runner currently reads shared opening timing and Fill model-verification records from the saved Eilean experiment paths declared at its top. The transfer helper also reads archived Chambord/Peña and model-path records. Preserve these local dependencies and the accepted case; the tracked JSON snapshots here document them but do not automatically rewire those code paths. Model-check reuse is stat-bound to previously byte-verified weights, not a claim of fresh full-byte hashing. Moving to another machine/backend requires explicit revalidation. Generated assets and weights are ignored by git: the repository alone is not a portable backup.

For a same-input replay, choose a **fresh** directory. These commands prepare a new run; do not point them at `BASE`:

```bash
export BASE=outputs/touchdown/studio/studio-bamburgh-complete-grounded-v1-20261027
export RUN_ROOT=outputs/touchdown/studio/studio-bamburgh-grounded-v1-REPLAY_NEW
mkdir -p "$RUN_ROOT"
cp "$BASE/selection.json" "$RUN_ROOT/selection.json"
PYTHONPATH=scripts:src .venv/bin/python - <<'PY'
import os
from pathlib import Path
from touchdown import photographic_transfer as transfer
root = Path(os.environ['RUN_ROOT']).resolve()
transfer.SELECTION = root / 'selection.json'
transfer.prepare(root / 'flight')
PY
PYTHONPATH= .venv/bin/python scripts/touchdown/photographic_transfer.py worker --root "$RUN_ROOT/flight"
```

For a different destination, create `selection.json` with `selected.source` pointing to its prepared case and record its actual selection method/seed/reuse scope. The legacy transfer wrapper has “random” boilerplate; it does not perform the selection, and the explicit selection record governs.

Review the fresh flight, then write this run's `scene-layout.json`. The saved Bamburgh example is a schema/example, not authority to skip reviewing a new generated landscape. If a same-input replay's flight agrees, copy that example and record the comparison. Then execute each stage in order:

```bash
PYTHONPATH=scripts:src .venv/bin/python scripts/touchdown/scene_grounded_workflow.py terrain-prepare --root "$RUN_ROOT"
PYTHONPATH=scripts:src .venv/bin/python scripts/touchdown/scene_grounded_workflow.py texture --root "$RUN_ROOT"
# Inspect terrain/texture.png and record the review before continuing.
PYTHONPATH=scripts:src .venv/bin/python scripts/touchdown/scene_grounded_workflow.py expand --root "$RUN_ROOT"
```

The expansion worker stays running at each review checkpoint. From a second terminal/controller, inspect `terrain/expand-N/raw-outpaint.png`, then write `terrain/decisions/expand-N.json` with `{"continue": true, "reason": "actual review findings", "evidence": ["reviewed artifact path"]}` only when that intermediate is usable. Use `false` to hold; do not pre-create approvals. Repeat for **all three**, including the third so the runtime exits normally. Review timeout is **900 seconds** per checkpoint; Fill generation timeout is **1800 seconds** per job. A hold/error is not a reason to rerun blindly into the same directories: retain evidence and create a documented recovery/variant.

```bash
PYTHONPATH=scripts:src .venv/bin/python scripts/touchdown/scene_grounded_workflow.py project --root "$RUN_ROOT"
# Inspect terrain/globe-guide.png and globe-target-diagnostic.png.
PYTHONPATH=scripts:src .venv/bin/python scripts/touchdown/scene_grounded_workflow.py orbit --root "$RUN_ROOT"
# Inspect terrain/orbit.png and record acceptance before preparing LTX.
PYTHONPATH=scripts:src .venv/bin/python scripts/touchdown/scene_grounded_workflow.py opening-prepare --root "$RUN_ROOT"
PYTHONPATH= .venv/bin/python scripts/touchdown/scene_grounded_workflow.py opening-worker --root "$RUN_ROOT"
PYTHONPATH=scripts:src .venv/bin/python scripts/touchdown/scene_grounded_workflow.py assemble --root "$RUN_ROOT"
```

Use separate worker processes with `PYTHONPATH=` as shown: they install the frozen inference package before importing it. Do not call them from a Python process that already imported the live inference modules. Automated review artifacts for the accepted case can be rebuilt with `.venv/bin/python "$BASE/review/build_review.py"`; that case-specific script asserts its **195-frame** result and is not a generic validator for another scene.

## Acceptance and version changes

Bamburgh is the accepted visual control. Its horizon leaves the upper frame, its sampled cloud join is continuous and its landing/portrait are usable. There remains a weaker-motion stretch around **0.3–0.8 seconds**. Regional geometry is approximate; the below-cloud headland reads nearly as an island, and generated terrain/architecture redraw. The same recipe's complete reliability across locations and seeds is unproven. Historical Chambord repetition established same-input repeatability for the base flight on the captured backend, not for this entire grounded chain.

Keep prompts, seeds/offsets, model identities, archived code, timing, guide strengths, mapping and assembly fixed for a canonical replay. Scene configuration, photographs, native portrait, region, reviewed exit frame and authored layout are explicit per-case inputs. Record those changes. A timing, prompt, seed search, projection, model or assembly change is a named variant with its own fresh root and comparison against the control; it does not silently become v1. Retain failures and logs. Promote a revised canonical version only after reviewing its complete film.

## Verify the saved control without generation

The canonical manifest binds 302 input, code, graph and evidence files by SHA-256. The copied preset/timing/layout JSON files are ordinary repository artifacts; the large accepted outputs remain in the local ignored run tree. This read-only check performs no model calls:

```bash
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('workflows/touchdown/canonical/photographic-grounded-v1/recipe.json').read_text())
for name, expected in manifest['files'].items():
    with Path(name).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise SystemExit('Changed canonical evidence: ' + name)
print('Verified', len(manifest['files']), 'canonical files')
PY
```
