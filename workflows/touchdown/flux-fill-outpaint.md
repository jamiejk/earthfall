# Native ComfyUI FLUX.1 Fill outpainting

Drag `flux-fill-outpaint.json` into ComfyUI. Load the intended photograph in **Load Image**. The graph resizes a working copy to 960×544, pads 480 pixels left/right and 272 top/bottom, and generates a 1920×1088 result. The source file is unchanged. This is a 2× expansion in each dimension at working resolution, not native 4K generation.

Three outputs are saved: **padded input**, **padding mask** (white = generated region), and **raw outpaint**. There is no image composite or manual restoration of the source centre. Inspect the raw outpaint for seams, duplicated buildings/people, altered proportions and inconsistent lighting before using it as a video keyframe. A correctly connected graph is not visual acceptance.

The official-example recipe produces a coherent expanded still but redraws some source pixels. Its separate `-preserve-source` graph exposes a failed diagnostic: simply restoring the native original reveals a rectangular colour boundary. Use neither result as proof of exact nesting. The `-hard-protect` variant tests source protection during sampling (`noise_mask=true`, zero feather), keeps the raw output visible, and adds a 3840×2176 output containing the exact native source in the centre. **That variant also failed visual review**, with an obvious rectangle and blurred surroundings. These two variants are retained as failed diagnostics; use the base graph only for ordinary raw outpainting, not as a validated exact-nesting solution.

## Models

The exact upstream revisions, sizes and SHA-256 hashes are in `flux-fill-models.json`. Files belong in the corresponding ComfyUI model folders, or add this project's `models/FLUX-Fill/{diffusion_models,text_encoders,vae}` folders to ComfyUI's extra model paths.

- `diffusion_models/flux1-fill-dev.safetensors`: official BFL Fill model, not generic Flux, Flux Klein or Kontext.
- `text_encoders/clip_l.safetensors` and `t5xxl_fp16.safetensors`.
- `vae/ae.safetensors`: FLUX.1 VAE, not the FLUX.2 VAE.

The isolated runner configures these extra paths automatically. To download the pinned files using an existing Hugging Face login with access to BFL Fill:

```bash
PYTHONPATH=scripts:src /home/jamie/comfyui-venv/bin/python -m touchdown.fill_outpaint --install-models
```

For an interactive ComfyUI session using these same files, add `--extra-model-paths-config /run/media/jamie/Data/Projects_Live/ComfyUI_Node/workflows/touchdown/flux-fill-model-paths.yaml` to its startup command, then import the UI JSON. This optional interactive setup is not applied to the main application by the isolated runner.

The total is about 34.2 GB. The graph loads the official BF16 transformer with `fp8_e4m3fn` casting for the 24 GB GPU; this changes numerical precision and may affect quality. The files themselves remain the official BF16 weights. The T5 encoder stays FP16.

## Reproduce the controlled test

From the project root:

```bash
PYTHONPATH=scripts:src .venv/bin/python -m touchdown.fill_outpaint \
  --source outputs/touchdown/studio/studio-chillon-minimal-clay-v2-20261026/photo-096.png \
  --root outputs/touchdown/studio/chillon-native-fill-example \
  --run
```

Use a fresh output directory for each explicitly authorized candidate. Omitting `--run` prepares the input, UI graph, API graph and identity record without using the GPU. Runtime settings come from the project's existing private runtime file; credentials are not copied into the workflow or outputs. The runner refuses an occupied generation port and releases only the server process it starts.

The fixed recipe is seed 1399629552, guidance 30, 20 steps, CFG 1, Euler/normal, denoise 1, 24-pixel feather, and `InpaintModelConditioning.noise_mask=false`, matching the official example's conditioning behavior. Both image and mask feed the dedicated Fill conditioning. This is an isolated still-image trial; it changes no flight timings, viewport, LTX recipe or application gallery.

Add `--hard-protect` to the runner command to execute the hard-protection variant in a fresh root. It uses the same recipe except the two explicit source-protection settings above, and verifies the final centre against the original pixel-for-pixel. Exact centre equality does not automatically pass seam inspection.

## References

- [Official ComfyUI Flux examples, including Fill outpainting](https://comfyanonymous.github.io/ComfyUI_examples/flux/). The original PNG and embedded graphs are retained under `outputs/review/flux-fill-outpaint-20261027`.
- [Official FLUX.1 Fill model](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev).
- [User-supplied article](https://flux3-ai.net/blog/flux-outpainting/) and [BFL's hosted outpainting documentation](https://docs.bfl.ai/flux_tools/flux_outpainting). The article covers a separate hosted endpoint, not this downloadable Fill model or a ComfyUI workflow. No BFL hosted request is made here.
