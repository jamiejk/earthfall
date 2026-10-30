# Final upscaler

An optional last pass on the finished LTX video (1920x1088, 24 fps). Two targets:

- `4k`: double to 3840x2176.
- `1080p`: keep the size, but halve the frames first and let the model rebuild the texture at 2x. LTX's own
  2x latent upscale leaves distant canyon texture and tree crowns soft, so this is a detail pass rather than an
  upscale.

```
geoblock-run ... --ltx --ltx-version 2.5 --keyframes flux-klein --final-upscale seedvr2 --final-target 4k
geoblock-run --upscale-video outputs/runs/<job>/shot_ltx25_00001_.mp4 --final-upscale seedvr2 --final-target 1080p
```

In a shot graph the upscaler takes the decoded LTX frames directly (no MP4 round trip). The LTX video is still
saved as before; the upscaled one is node `490` (`*_final_<upscaler>_<target>`). In the ComfyUI UI, put
**GeoBlock 8 · Final Upscale Prep** between the LTX decode and the upscaler: it unloads the video model (SeedVR2
and FlashVSR load their weights outside ComfyUI's model manager, so LTX would otherwise still be in VRAM), halves
the frames when needed, and outputs the height to wire into SeedVR2's `resolution`.

## Candidates

| `--final-upscale` | Model | Licence | How it works | Expected cost on a 4090 |
|---|---|---|---|---|
| `seedvr2` | SeedVR2 3B fp8 (ByteDance) via numz/ComfyUI-SeedVR2_VideoUpscaler 2.5.24 | Apache 2.0 (weights and node) | One-step diffusion restoration, temporal batches of 21 (4k) / 61 (1080p) frames, 4-frame overlap, LAB colour match | Unknown here; reports say minutes per second of 4K |
| `seedvr2-7b` | SeedVR2 7B fp8 (last block fp16) | Apache 2.0 | Same, bigger model | Slower; may need block swap at 4k |
| `flashvsr` | FlashVSR v1.1 (Wan 2.1 1.3B base) via lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast | Weights Apache 2.0; node GPL-3.0 (same as this repo) | One-step streaming diffusion, 4x only, so 4k from half-size frames | Fast on paper; third-party ComfyUI ports are known to lose quality |
| `ltx-detail` | LTX 2.5 + Lightricks' Refine Details IC-LoRA, tiled-fusion sampler | LTX-2 Community License (partner model) | Lightricks' `LTX-2.5_V2V_TiledFusion_Upscale.json`: 8 distilled steps from noise with the clip as the IC-LoRA guide, HD tiles fused every step | Slowest: Lightricks quote ~22 min for 121 frames to 4K on an RTX PRO 6000; at 4k a 10 s clip is 25 tiles x 4 temporal windows |
| `esrgan` | Real-ESRGAN x2plus | BSD-3 | Frame by frame, no temporal model | Baseline; expected to shimmer on foliage |

Rejected without testing:

- MiniMax-H3-X2-Detail-VAE: decodes MiniMax H3 latents only (and its detail mode needs the RGB source image), so
  it cannot take LTX output.
- LTX 2.5 Pixel Spatial Upscaler IC-LoRA: a 2x re-render from a half-size reference, meant to replace stage 2.
  As a pass after a 1080p render it would have to run at 4K in one canvas, which does not fit 24 GB.
- Topaz, fal, WaveSpeed and other hosted upscalers: paid services.
- Upscale-A-Video (non-commercial licence), 4x-UltraSharp (CC BY-NC-SA).

SeedVR2's own model card warns it "tends to over-sharpen lightly degraded inputs like 720p AIGC videos" and is
"not robust to ... large motion sequences". Both apply to our fly-downs, so the test decides.

## Shoot-out

`scripts/setup_upscalers.sh <ComfyUI dir> <Live-Models dir>` installs the nodes (pinned commits) and the weights
(into Live-Models, symlinked into ComfyUI). `scripts/upscaler_shootout.py` runs each upscaler on 121-frame
stretches of finished renders, waits for an empty queue before each job, records wall time, peak GPU memory and
lowest free RAM, and scores every output against its source:

- **PSNR vs source**: the output shrunk back to 1920x1088 against the source frame. Lower means the upscaler
  changed or invented content.
- **Colour shift**: mean per-channel difference (0-255) after shrinking.
- **Detail vs bicubic**: high-frequency energy relative to a bicubic upscale (1.0 = nothing added).
- **Flow-warped flicker vs bicubic**: frame-to-frame change of the high-frequency band after warping the next
  frame back along the source's optical flow, relative to bicubic. 1.0 is as steady as the source; above it is
  added shimmer. A sanity run on a 960 px preview gave 1.00 for bicubic, 1.10 for a strong unsharp mask and 2.15
  for bicubic plus per-frame noise. Real-ESRGAN x2plus itself, run on CPU on 25 preview frames (15 s a frame),
  scored 1.77: the frame-by-frame shimmer on the juniper and sage that the metric is meant to catch.

results.md ends with a provisional pick per target. The pass marks were fixed before any GPU run: flow-warped
flicker at most 1.15, colour shift at most 3, PSNR at least 30 dB and at most 45 min for a 10 s clip, with the
worst clip counting. Among the upscalers that pass, the one adding the most detail wins. The shimmer videos still
get watched before anything becomes the default.

It also writes crop sheets and a side-by-side crop video per clip and target into `docs/review/upscaler/`.
Results are in [review/upscaler/results.md](review/upscaler/results.md).
