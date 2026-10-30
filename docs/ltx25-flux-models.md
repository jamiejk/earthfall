# LTX 2.5 Layout To Render and FLUX.2 [klein] keyframes

Two optional swaps in the `--ltx` render-to-real stage, both off by default:

- `--keyframes flux-klein` paints the photoreal keyframes with FLUX.2 [klein] 9B distilled (4 steps, the frame
  and, on eye-level Earth zoom frames, the `--subject` photo as reference latents) instead of Qwen-Image-Edit 2511.
- `--ltx-version 2.5` renders with LTX 2.5 22B distilled and Lightricks' Layout To Render IC-LoRA instead of
  LTX 2.3 + 3DREAL. The graph follows Lightricks' `LTX-2.5_ICLoRA_Layout_To_Render_Two_Stage_Distilled.json`
  (video only, no audio): the blocking video is the IC-LoRA guide at frame 0, the first keyframe is a one-frame
  IC-LoRA reference at RoPE position -1, and stage 2 re-adds both at full size on the upscaled latent (clips over 5 s / 121 frames leave the
  layout guide out of stage 2: it doubles the tokens and ran a 10 s Earth zoom out of memory on 24 GB). The mid
  and last keyframes are pinned with LTXVAddGuide on both stages.

Both need ComfyUI 0.32+ and an up-to-date ComfyUI-LTXVideo. Both partner models (LTX, BFL) sit in gated
Hugging Face repos: accept the terms once, then download with a token that can read gated repos.

| Folder | File | Source |
|---|---|---|
| diffusion_models | flux-2-klein-9b-fp8.safetensors | black-forest-labs/FLUX.2-klein-9b-fp8 |
| text_encoders | qwen_3_8b_fp8mixed.safetensors | Comfy-Org/flux2-klein-9B (split_files/text_encoders) |
| vae | flux2-vae.safetensors | Comfy-Org/flux2-dev (split_files/vae) |
| diffusion_models | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors | Lightricks/LTX-2.5 |
| text_encoders | gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors | Lightricks/LTX-2.5 |
| vae | ltx-2.5-video-vae-bf16.safetensors | Lightricks/LTX-2.5 |
| latent_upscale_models | ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors | Lightricks/LTX-2.5 |
| loras | ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors | Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render |

Licences: LTX-2 Community License (free under $10M annual revenue); FLUX.2 [klein] 9B is under the FLUX
Non-Commercial License (BFL's card allows commercial use of outputs); klein 4B is Apache 2.0.

Memory on a 24 GB card: the int8 LTX 2.5 stack peaks around 22.7 GB (third-party report), so nothing else may be
resident. The 14 GB text encoder has to unload before sampling; decode uses 512 px tiles because 768 reportedly
stalls 10 s clips.

`_add_ltx25` in `src/geoblock_core/workflow.py` was checked against Lightricks' workflow JSON from the LoRA's
repo (6 Oct 2026). Same sigmas (8 steps from 1.0, then 3 from 0.909375), CFG 1 and euler_ancestral on both
stages; the guide tokens are cropped after each stage. The ComfyUI-LTXVideo custom nodes are required
(`LTXICLoRALoaderModelOnly`, `LTXAddVideoICLoRAGuide`).
