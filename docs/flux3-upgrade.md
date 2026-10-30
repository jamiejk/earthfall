# FLUX 3 Image Klein upgrade

Checked 16 October 2026. **Prepared, not installed or render-tested:** the local Hugging Face credentials were denied access to the official ComfyUI package. Accept access on the [model page](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy), then authenticate the same account with `hf auth login` if necessary.

The local ComfyUI installation at `/run/media/jamie/Data/Image Creation/Comfy` is v0.38.2 (`daeb5e5`), with FLUX.2 Klein 9B FP8 and an RTX 4090 (24 GB). The new model's official workflows target ComfyUI v0.41. The checked v0.41.2 release is `b53f77a74e21091cbdebfdf7b85fc80326932646`; its requirements also update the frontend, workflow templates, Comfy Kitchen and Aimdo. Updating weights alone is insufficient. The existing installation has not been changed.

## Download

From this repository:

```bash
bash scripts/download_flux3_klein.sh
```

The script pins BFL's ComfyUI package to `5c9b714851dc072b04a23ef6d2d36421f5ccb3a0`, downloads the distilled INT8 transformer, INT8 Qwen3-VL-4B encoder, INT8 prompt upsampler, VAE, three official workflows and license files, then verifies checksums. The weights total about 20.8 GB. It keeps this package in the ignored `models/FLUX-3-Klein/` directory, separate from existing models. The VAE is included for a self-contained package even though a FLUX.2 VAE is already installed.

## Activate after access is granted

With ComfyUI stopped, record its current commit and Python package versions for rollback, update the checkout to the pinned v0.41.2 commit, and install that revision's `requirements.txt` into its own `/home/jamie/comfyui-venv` environment. Preserve `extra_model_paths.yaml` and existing custom nodes. Confirm the GeoBlock and H3 nodes still load before resuming video work.

Add this entry to ComfyUI's `extra_model_paths.yaml` (adjust the path on another machine):

```yaml
geoblock_flux3:
  base_path: /run/media/jamie/Data/Projects_Live/ComfyUI_Node/models/FLUX-3-Klein
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae
```

Restart ComfyUI and load the downloaded `workflows/image_flux3_klein_distilled_t2i.json`. Select the downloaded INT8 model and encoder if the template defaults differ. Keep `enable_prompt_upsampler` enabled and use the distilled defaults: 16 steps, no guidance. Run a small text-to-image smoke test, then test `image_flux3_klein_distilled_image_edit.json` with a GeoBlock opening frame. The editing templates do not contain a prompt-upsample branch: supply a full scene caption expanded from the edit request and reference images by a vision-capable LLM, as BFL documents.

## Expected fit for GeoBlock

[BFL's model card](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) describes generation and editing with up to ten references, text rendering and bounding-box placement. Its INT8 package is intended for consumer GPUs, including 8 GB cards; actual peak memory and latency on this 4090 remain unmeasured. NVFP4 requires Blackwell, so it is not the selected format for this Ada GPU.

Reference images anchor content but do not lock geometry, and boxes are not hard masks. Evaluate a clay opening frame plus a place photograph against the existing FLUX.2/Qwen outputs: check horizon, viewpoint, building footprint, terrain silhouette, material detail and requested weather. Record the expanded caption, seed, dimensions, time and peak memory. Better-looking images alone do not establish suitability for measured-scene keyframes or improve the H3 video model itself.

Sources: [BFL model and workflow instructions](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy), [reference implementation](https://github.com/black-forest-labs/flux-3-image-klein), [ComfyUI v0.41.2 requirements](https://github.com/Comfy-Org/ComfyUI/blob/v0.41.2/requirements.txt).
