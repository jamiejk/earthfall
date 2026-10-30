# Programmatic flyover pipeline: status after the Woodside test

The Woodside Library test is reproducible, but it is still a prepared fixture rather than an arbitrary-address ComfyUI generator. The recommended input contract is: location or address, coverage radius, terrain provider, camera route/waypoints, duration, geographic reference source, visual style and optional user-supplied style images. Each stage should write a manifest with source, licence, datum, bounds, hashes and output settings.

| Stage | Current automation | Remaining work |
| --- | --- | --- |
| Resolve place | User-triggered landmark search returns numbered Photon/OSM results; the user selects a specific feature and coverage radius. The old fixed-name resolver remains only in fixture graphs. | Connect the selected place to a real geometry-provider registry and validate its full footprint and camera corridor. Use a dedicated/self-hosted geocoder when traffic outgrows the public demo service. |
| Find terrain and 3D height | The Woodside ground DEM and lidar point cloud are combined with checked OSM footprints; Worm's Head has a measured local-relief manifest. Every blocking route checks its target-height evidence. | General provider registry and region-aware structure-height coverage checks for arbitrary places. |
| Find imagery | The Woodside script fetches NAIP crops and records acquisition/licence data. Commons mining is scripted for Worm's Head. | Provider registry outside US NAIP; machine-check rights and coverage, then review ambiguous licences or stale imagery. |
| Build shot | Blender rejects Woodside renders without measured target/building heights. The current blocking guide uses separate building volumes over measured terrain, without an aerial roof drape. | Parameterise camera waypoints, altitude, target and framing. |
| Assemble H3 request | The local script stages checked images/video, writes an API graph, prompt, seed and manifest, and can submit to a GPU ComfyUI URL. | Generalise from the pinned fixture; make image-role selection and frame count a full Comfy adapter. |
| Evaluate output | Dimensions, frame count and file presence can be checked automatically. | Human approval remains needed for geographic faithfulness, legible style, temporal consistency and model-looking artifacts; add automated contact sheets and candidate comparison. |

For the Woodside style test, `assets/style_references/woodside_clearline/` contains two original imagegen illustrations. They are style references, not surveyed geography. The current request puts the measured blocking guide's opening frame and a 220 m address-centred NAIP crop in image slots 1–2, the two clear-line illustrations in slots 3–4, and the measured 3D blocking video in video slot 1. No Tintin artwork or characters are included.

Prepare the staged API request without generating video:

```bash
python scripts/queue_woodside_h3.py --take closemeasuredtight --style clear-line
```

This writes `outputs/woodside_library_3dep/h3_clearline_closemeasured_tight_workflow_api.json` and its reference manifest. To run it later, use the same command with `--submit --url http://127.0.0.1:<GPU-Comfy-port>`. The script never submits unless `--submit` is given. The Comfy preparation canvas `examples/woodside_library_comfy.json` also shows both style images via `GeoBlockClearLineStyle`.

Historical Woodside H3 runs established that prompts alone do not isolate a video's camera path from its surface appearance. Flat controls also encouraged map-like patches and once induced a pin. Those experiments remain recorded in [the test log](woodside-library-test.md), but are no longer eligible as blocking references. The current model has measured building volumes; further style work can add separate proxy trees and roads or a temporally consistent illustration stage.

## Optional Veda H3 attention

[Veda-on-ComfyUI](https://github.com/veda-sparse/Veda-on-ComfyUI) provides a `MODEL`-to-`MODEL` sparse-attention node for native H3 T2VA, FL2VA and R2VA. Add `--veda` to either `scripts/queue_h3_preview.py` or `scripts/queue_woodside_h3.py` to insert it between the H3 model loader and the guider/scheduler. This is opt-in and writes separate `_veda` workflow and video names, so the dense baseline remains available for same-seed comparison. The generated video uses the node's trained 90% sparsity; reference images and video retain full attention (`reference_sparsity=0%`) because they carry the geographic identity and camera path. Submission checks that the node is loaded on the target ComfyUI server.

The node requires ComfyUI 0.38.0 or newer and the 275 MB predictor in `models/veda/`. Its released predictor was trained around 8-step Turbo T2VA at selected 768-line sizes; GeoBlock's 12-step, 864 × 480 Ref2VA graph is outside that training configuration. The project reports speed on an RTX 5070, but has not published a 4090 measurement for this ComfyUI kernel. Treat speed and geographic/style fidelity as unverified until a same-seed dense/Veda pair is rendered and compared frame by frame. The 4090 GPU backend self-tests passed locally (3 tests), which checks kernel operation, not shot quality or end-to-end speed.
