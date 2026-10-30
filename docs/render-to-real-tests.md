# Render-to-real tests: H3 depth control vs LTX 3DREAL

*5 October 2026, local RTX 4090 (24 GB), 95 GB RAM. Mather Point rim reveal, heading 5°, seed 20261005. Follows [h3-geometry-vs-look.md](h3-geometry-vs-look.md).*

## Setup that works locally

- ComfyUI `/run/media/jamie/Data/Image Creation/Comfy` (v0.38.2) via `scripts/start_comfy_h3.sh --disable-smart-memory --reserve-vram 3.5 --vram-headroom 6`, plus `comfy-api-proxy` on 127.0.0.1:8189.
- Do not turn dynamic VRAM off: with `--disable-dynamic-vram` the H3 weights went to pinned RAM and systemd-oomd killed the session (15:09).
- ComfyUI caches node outputs between prompts. After changing `blender_render.py`, restart ComfyUI or the old blocking video is reused.

## H3 with Fun ControlNet depth

| Run | Result |
| --- | --- |
| 10 s (243 frames), depth 0.8 | Out of GPU memory on step 1 even with 6 GB headroom: the control block asks for another 4.2 GB. |
| 5 s, depth 0.8 → 60%, 3 photos | Runs (~40 s/step, ~13 min). Still follows the photos: opens at the edge, draws the overlook and people. |
| 5 s, depth 1.0 → 100%, 3 photos | No overlook or people; still opens at the edge; rock goes pale. |
| 5 s, depth 1.0 → 100%, 1 photo | Follows the blocking closely, but copies the clay's grey sky and white ridge tops. |
| + `--palette grand-canyon` | Real rock bands and sky. A pale haze band at the horizon became an ocean until the haze was cut. Foreground still reads as CG. |

## LTX 2.3 + 3DREAL Strong v2 (IC-LoRA)

`scripts/queue_ltx_3dreal.py` builds the graph from stock nodes (`LTXVAddGuide` + `GetICLoRAParameters`; fal's workflow needs six custom node packs). Weights: `ltx-2.3-22b-dev-fp8`, `ltx-2.3-22b-distilled-lora-384` at 1.0, `gemma_3_12B_it_fp8_scaled`, `3DREAL-strong-v2`. 960×544, 8 steps, CFG 1: ~4 min to load, ~32 s to sample 121 frames.

| Run | Result |
| --- | --- |
| Clay blocking, no keyframe | Geometry exact, barely repainted. |
| fal settings (CFG 3, 15 steps, distilled 0.5) | Little different. |
| Palette blocking, no keyframe | Exact blocking, adds pines on the plateau, still CG. |
| + H3's frame 0 as keyframe | Most photoreal but follows the keyframe's canyon view instead of the plateau. |
| + Qwen keyframe of our frame 0 | Follows the blocking start to end; canyon half still pastel. |
| + Qwen start **and** end keyframes | **Best: photoreal throughout and on the blocking's geometry.** |
| Same, full 10 s (241 frames) | Works end to end; ~7.5 min including model load. |
| + `--upscale` (2× latent upscale, 3-step refine) | 1920×1088; strata ledges and shrubs resolve instead of smearing. ~2 min when models are already loaded. |

**Rule:** the blocking carries geometry, the keyframes carry realism, and each keyframe must be a repaint of *our* frame, not a photo.

**Crash:** any LTX guide or keyframe below strength 1.0 fails in cuDNN attention ("No valid execution plans built") and can segfault ComfyUI on the next job. Keep guides at 1.0.

## Keyframes with Qwen-Image-Edit 2511

`scripts/queue_qwen_keyframe.py`: `qwen_image_edit_2511_fp8mixed` + `Qwen-Image-Edit-2511-Lightning-4steps` LoRA, 4 steps. The pre-merged `..._fp8_e4m3fn_scaled_lightning` file outputs noise in this ComfyUI (its `weight_scale` keys are ignored).

- Prompt from the fal blog: "Image 1 is the only source of truth for geometry … repaint surfaces only; do not reframe".
- One reference photo helps (plateau look). Two photos pull Qwen to the photos' viewpoint.
- The end keyframe is best from the prompt alone.
- `--quality` (no Lightning, 24 steps, CFG 4, 1792×1008) is sharper but redraws the shape (a new butte, a moved horizon), so Lightning keyframes stay the default for render-to-real.

## Weather

`--weather` on `geoblock-run`, the Render Blocking Video node and both queue scripts: clear, partly cloudy, overcast, hazy, golden hour, thunderstorm (`src/geoblock_core/weather.py`). Each preset sets the Blender sky gradient, sun strength and angle, sky light and haze, and adds a matching sentence to the H3, LTX and Qwen prompts. The models copy the blocking's light, so the render and the prompt need to agree. Partly cloudy has no clouds in the blocking; the prompt carries them.

## Waypoint photos

`place_photos.route_waypoint` / `fetch_waypoint_photos` search Commons at the camera's position and heading for a given frame. Pure geosearch returns many off-subject images (a shed, a lizard, an agave), so picks still need a visual filter. A local Qwen-VL scorer is the next step. Commons search also sheds load ("too busy"); the client now retries.

## Trees on the rim plateau

The blocking plateau is an empty flat field, so trees exist only in the Qwen keyframe. A terse prompt ("scattered pinyon and juniper") gives round, evenly sized bushes on dirt circles, and LTX has no geometry for them, so they swell, slide and pop as the camera moves. Prompting for "natural pinyon-juniper woodland: irregular, windswept trees of many sizes … grasses and sagebrush between them" (the thunderstorm take) looks far more real. Tree proxies scattered in Blender would give them correct parallax.
