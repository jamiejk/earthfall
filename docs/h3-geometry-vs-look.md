# Balancing geometry and look in MiniMax H3

*5 October 2026. Research for the Grand Canyon shot. The [first-run review](grand-canyon-h3-first-run.md) has the problem: without photos H3 follows the blocking but invents the place, and with photos it looks right but leaves the blocking.*

## How H3 takes each input

Read from `comfy_extras/nodes_minimax_h3.py` in our ComfyUI (commit `daeb5e5`) and the Comfy-Org [`video_minimax_h3_r2v`](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json) template.

| Input | What the code does with it | How hard it binds |
| --- | --- | --- |
| `MiniMaxH3ReferenceToVideo` ref images (≤9) | Resized (`match`: down to the output's pixel area; `max`: 2048 px short edge), VAE-encoded, and carried as extra reference tokens on every step. Qwen3-VL also sees them in the prompt as `<Picture i>`. | Soft and global. Each one is just another reference with no frame position. |
| R2V ref videos (≤3, 2–15 s) | Same, as `<Video k>`. The VAE latent is a reference block, and Qwen sees the clip at **2 fps** with timestamps. | Soft. It describes a motion and a scene and is not tied to output frames. This is how our blocking video goes in today, on equal terms with the photos. |
| `MiniMaxH3AddGuide` | Encodes an image or a 17k+5-frame clip and pins it at `frame_idx` as a keyframe that is re-injected every step and never denoised. | Hard, but on pixels. The output reproduces that frame, so a clay frame stays clay. |
| `MiniMaxH3FunControlNetApply` | Encodes a per-frame control video (depth, canny, pose, HED, MLSD) and adds a control branch's output into 5 of the 50 DiT blocks (0, 10, 20, 30, 40), between `start_percent` and `end_percent`, scaled by `strength`. | Hard and frame-aligned on **structure** only, so appearance stays free. This is the lever we lack. |
| `MiniMaxH3SigmaShift` | Video/audio flow shift (defaults 12 and 3). | Indirect. Not a balance control. |
| Prompt | The template note says R2V is "very sensitive to prompt wording" and to name which tag drives what. | Medium. |

**Why the photos won.** Our graph passes the clay opening frame, three photos and the blocking video as five equal-weight references. Nothing ties output frame *n* to blocking frame *n*, so H3 blends them toward the strongest prior. That prior is the photos' framing, from the overlook looking out.

## Levers, cheapest first

1. **Say who does what in the prompt.** For example: "`<Video 1>` is the exact camera path and terrain shape; keep every ridge where it is. `<Picture 2>`–`<Picture 4>` only show rock colour, vegetation and light; do not copy their viewpoint, railings or people." This costs nothing to try.
2. **Remove the clay opening frame from the references.** As `<Picture 1>` it pulls the look toward clay and adds nothing positional. To pin the start, use `AddGuide` at frame 0 with a *photographic* first frame instead (see 5).
3. **Use fewer photos that face the same way as the shot.** One or two views looking roughly along the route heading, without railings or people. `--auto-photos 2 --heading` already prefers titled view directions.
4. **Fun ControlNet with a Blender depth pass (main fix).** Render the same route as a depth (Z) video, which Blender can do in the same pass, and apply `minimax_h3_fun_controlnet_union_2.0` to the Ref2VA model. Community settings: strength **0.6–1.0**, `end_percent` **≈0.6** so late steps can add texture, and depth for "scene geometry and camera placement". The patch was trained on fl2va, but the [ComfyUI-H3-FunControl](https://github.com/wyzborrero/ComfyUI-H3-FunControl) notes and the [RunComfy depth/pose workflow](https://www.runcomfy.com/comfyui-workflows/minimax-h3-fun-control-in-comfyui-depth-and-pose-video) both run it on ref2va with reference images, which is our case. The photos can then carry only the look.
5. **Photographic first frame via AddGuide.** Make one photoreal frame of the opening view (a 5-frame H3 run with depth control, or an image model), then anchor it at `frame_idx 0`. This fixes the opening. Optionally anchor a last frame too.
6. **Steps.** The template uses 20 steps. We use 12 with the `beta` scheduler, which the template note recommends over `simple` for reference-heavy prompts. Twelve may be slightly under-converged, so worth one A/B test once the faster setup below is in.

## Speed

- **Official 4-step Ref2VA turbo LoRA**: `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` (1.96 GB, Comfy-Org). The template wires it with `LoraLoaderModelOnly` at strength 1 and 4 steps. Sampling time scales with steps, so 12 → 4 should cut our ~24 min graph to under 10 min (estimate). It is v0.1, so check its quality against the 12-step take.
- Kijai's `MiniMax-H3-Ref2VA-Acc-8Step` LoRA is a middle option at 8 steps.
- `ref_image_size: match` (what we use) is the fast setting; `max` costs several times more.
- The Veda sparse-attention node is already supported by the repo's queue scripts. ComfyUI-H3-FunControl reports Sol-Attn at about 1.5× with the control net, but only with `morton: false`.
- The Civitai link (`civitai.com/models/2980016`) is blocked from the UK under the Online Safety Act, so I could not read it.

## Weights

All under the Comfy-Org repo (also mirrored by Kijai):

| File | Size | Use |
| --- | --- | --- |
| `model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors` | 4.5 GB | depth/canny/pose/HED/MLSD control, int8 for the 4090 |
| `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 2.0 GB | 4-step Ref2VA |

## Proposed test order

All tests use the same route and seed, each compared against the two existing takes:

1. Turbo 4-step and today's photo graph. This checks speed and quality, and every later test gets cheaper.
2. Plus the explicit prompt, with the clay opening frame removed.
3. Plus a depth control video at strength 0.8 and `end_percent` 0.6.
4. Plus a photographic frame-0 guide.
