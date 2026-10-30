"""Native FLUX.1 Fill trials with separate raw and optional source-restored outputs.

The paired UI/API graphs derive from ComfyUI's official Flux Fill example.
Run with PYTHONPATH=scripts:src .venv/bin/python -m touchdown.fill_outpaint
  --source <photo.png> --root <fresh-output-directory> [--run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import urllib.parse
import urllib.request
import uuid

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / 'workflows/touchdown/flux-fill-outpaint.json'
API = WORKFLOW.with_suffix('.api.json')
MODELS = WORKFLOW.parent / 'flux-fill-models.json'
SEED = 1399629552


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate(g, hard_protect=False):
    def inp(n): return g[str(n)]['inputs']
    if inp(31) != dict(unet_name='flux1-fill-dev.safetensors', weight_dtype='fp8_e4m3fn'):
        raise ValueError('Requires the dedicated Fill model, with declared fp8 loading')
    if inp(32)['vae_name'] != 'ae.safetensors' or inp(34)['type'] != 'flux':
        raise ValueError('Requires FLUX.1 VAE and dual text encoders')
    if inp(38) != dict(noise_mask=hard_protect, positive=['26', 0], negative=['7', 0],
                      vae=['32', 0], pixels=['44', 0], mask=['44', 1]):
        raise ValueError('Fill image/mask conditioning must follow the official example')
    if inp(44) != dict(left=480, top=272, right=480, bottom=272, feathering=0 if hard_protect else 24, image=['45', 0]):
        raise ValueError('Expansion geometry changed')
    if (inp(45)['width'], inp(45)['height'], inp(45)['image']) != (960, 544, ['17', 0]):
        raise ValueError('Working source must be 960x544')
    if inp(3) != dict(seed=SEED, steps=20, cfg=1.0, sampler_name='euler', scheduler='normal',
                      denoise=1.0, model=['39', 0], positive=['38', 0], negative=['38', 1],
                      latent_image=['38', 2]):
        raise ValueError('Frozen official 20-step recipe changed')
    if inp(26)['guidance'] != 30 or inp(9)['images'] != ['8', 0]:
        raise ValueError('Requires guidance 30 and raw VAE decode output')
    allowed = {'KSampler', 'CLIPTextEncode', 'VAEDecode', 'SaveImage', 'LoadImage',
               'FluxGuidance', 'UNETLoader', 'VAELoader', 'DualCLIPLoader',
               'InpaintModelConditioning', 'DifferentialDiffusion', 'ImagePadForOutpaint',
               'ImageScale', 'MaskToImage'}
    if hard_protect:
        allowed.add('ImageCompositeMasked')
        if inp(50) != dict(destination=['49', 0], source=['17', 0], x=960, y=544, resize_source=False):
            raise ValueError('Exact centre must come from the native original')
        if inp(49) != dict(image=['8', 0], upscale_method='lanczos', width=3840, height=2176, crop='disabled'):
            raise ValueError('Native output canvas geometry changed')
        if inp(51)['images'] != ['50', 0]:
            raise ValueError('Missing separate native-preserved output')
    if any(n['class_type'] not in allowed for n in g.values()):
        raise ValueError('Unexpected nodes: this trial excludes compositing and video generation')


def prepare(source, root, hard_protect=False):
    from PIL import Image
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('Use a fresh output directory; submitted candidates cannot be rerolled')
    if Image.open(source).size != (1920, 1088):
        raise ValueError('This controlled trial requires the native 1920x1088 reference')
    workflow = WORKFLOW.with_name('flux-fill-outpaint-hard-protect.json') if hard_protect else WORKFLOW
    api = workflow.with_suffix('.api.json')
    g = json.loads(api.read_text()); validate(g, hard_protect)
    photo = root / 'chillon-outpaint-source.png'
    photo.write_bytes(source.read_bytes())
    write(root / 'graph.api.json', g)
    (root / 'workflow.json').write_bytes(workflow.read_bytes())
    write(root / 'identity.json', dict(source=str(source), source_sha256=sha(source),
        source_copy_sha256=sha(photo), script_sha256=sha(Path(__file__)),
        workflow_sha256=sha(workflow), api_sha256=sha(api), seed=SEED,
        source_size=[1920, 1088], working_source_size=[960, 544], output_size=[1920, 1088],
        source_rectangle=[480, 272, 960, 544], feathering=0 if hard_protect else 24,
        hard_source_protection=hard_protect, native_centre_restored=False,
        model_loading='Official BF16 weights cast to fp8_e4m3fn by ComfyUI for the 24GB GPU',
        original_example='https://comfyanonymous.github.io/ComfyUI_examples/flux/',
        video_submissions=0, full_flight_clock_changed=False))
    write(root / 'status.json', dict(status='prepared', visual_acceptance=False))
    return g, photo


def run(root, graph, photo, hard_protect=False):
    from touchdown import flux3, studio_build as studio
    from touchdown.flight_runtime import capture_runtime
    models = json.loads(MODELS.read_text())
    for m in models:
        p = REPO / m['path']
        if p.stat().st_size != m['size'] or sha(p) != m['sha256']:
            raise ValueError(f"Model checksum mismatch: {p.name}")
        m['path'] = str(p)
    write(root / 'model-fingerprints.json', models)
    config = root / 'model-paths.json'
    write(config, {'flux_fill_trial': dict(is_default=True, **{
        role: str((REPO / 'models/FLUX-Fill' / role).resolve())
        for role in ['diffusion_models', 'text_encoders', 'vae']})})
    command = os.environ['TOUCHDOWN_COMFY_SERVER_CMD']
    if '--extra-model-paths-config' in command:
        raise ValueError('Uncontrolled model path override')
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD'] = command + ' --deterministic --extra-model-paths-config ' + shlex.quote(str(config))
    capture_runtime(root)
    with studio.runtime('comfy', root):
        info = flux3._call(studio.COMFY_URL, '/object_info')
        missing = {n['class_type'] for n in graph.values()} - set(info)
        if missing:
            raise ValueError(f'Missing native Comfy nodes: {missing}')
        write(root / 'node-preflight.json', dict(nodes=sorted({n['class_type'] for n in graph.values()})))
        graph['17']['inputs']['image'] = flux3._upload(studio.COMFY_URL, photo)
        validate(graph, hard_protect); write(root / 'submitted-graph.json', graph)
        response = flux3._call(studio.COMFY_URL, '/prompt', dict(prompt=graph,
            client_id=uuid.uuid4().hex, extra_data={'extra_pnginfo': {'workflow': json.loads((root / 'workflow.json').read_text())}}))
        write(root / 'submission.json', response)
        pid = response['prompt_id']; started = time.monotonic()
        write(root / 'status.json', dict(status='running', prompt_id=pid, visual_acceptance=False))
        while True:
            history = flux3._call(studio.COMFY_URL, f'/history/{pid}').get(pid, {})
            status = history.get('status', {})
            if status.get('status_str') == 'error':
                write(root / 'history.json', history)
                raise RuntimeError('Fill generation failed; inspect saved history and runtime log')
            if status.get('completed'):
                break
            if time.monotonic() - started > 1800:
                raise TimeoutError('Fill trial exceeded 30 minutes')
            time.sleep(2)
        write(root / 'history.json', history)
        outputs = [('9', 'raw-outpaint'), ('47', 'padding-mask'), ('48', 'padded-input')]
        if hard_protect:
            outputs.append(('51', 'native-centre-preserved'))
        for node, label in outputs:
            output = history['outputs'][node]['images'][0]
            query = urllib.parse.urlencode(output)
            with urllib.request.urlopen(studio.COMFY_URL + '/view?' + query) as r:
                (root / f'{label}.png').write_bytes(r.read())
        from PIL import Image
        import numpy as np
        raw = Image.open(root / 'raw-outpaint.png').convert('RGB')
        padded = Image.open(root / 'padded-input.png').convert('RGB')
        mask = np.asarray(Image.open(root / 'padding-mask.png').convert('L'))
        if raw.size != (1920, 1088) or padded.size != raw.size or mask.shape != (1088, 1920):
            raise ValueError('Unexpected output or mask geometry')
        protected = mask == 0
        error = np.abs(np.asarray(raw).astype(float) - np.asarray(padded).astype(float))
        write(root / 'pixel-audit.json', dict(protected_pixel_fraction=float(protected.mean()),
            protected_mean_absolute_rgb_error=float(error[protected].mean()),
            raw_output_size=list(raw.size), centre_pasted_back=False,
            interpretation='Diagnostic only; low pixel error does not establish a coherent expansion'))
        if hard_protect:
            final = Image.open(root / 'native-centre-preserved.png').convert('RGB')
            original = np.asarray(Image.open(photo).convert('RGB'))
            exact = final.size == (3840, 2176) and np.array_equal(
                original, np.asarray(final)[544:1632, 960:2880])
            write(root / 'native-preservation-audit.json', dict(native_original_centre_exact=exact,
                output_size=list(final.size), border_is_upscaled_working_generation=True,
                visual_acceptance=False, output_sha256=sha(root / 'native-centre-preserved.png')))
            if not exact:
                raise ValueError('Final centre is not pixel-exact')
        write(root / 'status.json', dict(status='awaiting_visual_review',
            raw_output_sha256=sha(root / 'raw-outpaint.png'), elapsed_seconds=time.monotonic()-started,
            visual_acceptance=False, native_centre_restored=False, video_submissions=0))


def install_models():
    """Fetch pinned official files; requires huggingface_hub and existing model access."""
    from huggingface_hub import hf_hub_download
    import shutil
    for m in json.loads(MODELS.read_text()):
        destination = REPO / m['path']
        if destination.exists():
            if destination.stat().st_size != m['size'] or sha(destination) != m['sha256']:
                raise ValueError(f'Refusing to overwrite changed model: {destination.name}')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        fetched = Path(hf_hub_download(repo_id=m['repo'], filename=m['filename'],
            revision=m['revision'], local_dir=destination.parent))
        if fetched != destination:
            shutil.move(fetched, destination)
        if sha(destination) != m['sha256']:
            raise ValueError(f'Checksum mismatch: {destination.name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--root', type=Path)
    p.add_argument('--install-models', action='store_true')
    p.add_argument('--run', action='store_true')
    p.add_argument('--hard-protect', action='store_true', help='Protect the source during sampling and save a second native-centre output')
    a = p.parse_args()
    if a.install_models:
        install_models()
        return
    if a.source is None or a.root is None:
        p.error('--source and --root are required unless installing models')
    if a.run:
        for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
            key, value = line.split('=', 1)
            if key.startswith('TOUCHDOWN_'):
                os.environ[key] = json.loads(value)
    root = a.root.resolve()
    graph, photo = prepare(a.source.resolve(), root, a.hard_protect)
    if a.run:
        try:
            run(root, graph, photo, a.hard_protect)
        except Exception as e:
            write(root / 'status.json', dict(status='failed', error=str(e), visual_acceptance=False))
            raise


if __name__ == '__main__':
    main()
