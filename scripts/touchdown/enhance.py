"""Optional source-bound Refine Details export, using the measured bounded recipe."""
from pathlib import Path
import argparse, hashlib, json, os, shlex, shutil, subprocess, time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def chunks(count):
    if count < 1:
        raise ValueError('Empty video')
    result = []
    first = 0
    while count - first > 57:
        result.append((first, first + 49, 49))
        first += 48
    real = count - first
    result.append((first, count, max(9, ((real - 1 + 7) // 8) * 8 + 1)))
    return result


def graph(asset, length, width, height, fps, prefix):
    from geoblock_core.upscale import upscale_file_graph
    g = upscale_file_graph(asset, method='ltx-detail', target='4k', prefix=prefix, seed=2081561474, fps=fps)
    for key in ('400', '449', '450'):
        g.pop(key)
    g['443']['inputs']['device'] = 'cpu'
    g['444']['inputs']['text'] = 'Sharp photographic detail, crisp natural texture, fine surface detail, clean edges, natural film grain, high resolution footage. Preserve the source framing, motion, lighting and colours.'
    g['445']['inputs']['text'] = 'blurry, soft, plastic, smeared detail, oversharpened halos, warped faces'
    g['446']['inputs']['frame_rate'] = fps
    g['451']['inputs'].update(width=width, height=height, length=length)
    g['452']['inputs'].update(image=['402', 0], width=width, height=height)
    # Streaming must fit ONE temporal window when tiled encoding is enabled.
    g['456']['inputs'].update(tile_frames=length, use_streaming=True, use_tiled_encode=True, tile_size=512, tile_overlap=64, latent_downscale_factor=1)
    g['459']['inputs'].update(tile_width=1024, tile_height=576, overlap_frac=.5, tile_frames=length, canvas_device='cpu')
    g['460']['inputs'].update(tile_size=512, overlap=64, temporal_size=64, temporal_overlap=16)
    return g


def check():
    from geoblock_core.workflow import LTX25_MODELS
    from geoblock_core.upscale import LTX_DETAIL_LORA
    root = Path(os.environ['TOUCHDOWN_ENHANCE_MODEL_ROOT'])
    for role, name in [('diffusion_models', LTX25_MODELS['transformer']), ('text_encoders', LTX25_MODELS['text_encoder']), ('vae', LTX25_MODELS['video_vae']), ('loras', LTX_DETAIL_LORA)]:
        if not (root / role / name).is_file():
            raise ValueError(f'Missing enhancement model: {role}/{name}')
    for program in ('ffmpeg', 'ffprobe'):
        if not shutil.which(program):
            raise ValueError(f'Missing {program}')
    if not os.environ.get('TOUCHDOWN_COMFY_SERVER_CMD') or not Path(os.environ.get('TOUCHDOWN_COMFY_SERVER_CWD', '')).is_dir():
        raise ValueError('Configure the owned ComfyUI runtime first')
    return root


def probe(path):
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate', '-of', 'json', str(path)]))['streams'][0]
    return dict(frames=int(data['nb_read_frames']), fps=data['avg_frame_rate'], width=int(data['width']), height=int(data['height']))


def run(source, root):
    model_root = check()
    from fractions import Fraction
    from touchdown import studio_build as studio
    from geoblock_core.run import _client, _submit
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'submission-start.json').exists():
        raise ValueError('Existing enhancement submission; use a new job to retry')
    def write(name, value):
        path = root / name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
        temporary.replace(path)
    stages = {'source': 'done', 'refine': 'running', 'assemble': 'pending', 'verify': 'pending'}
    def progress(message):
        write('progress.json', dict(stages=stages, message=message))
    started = time.monotonic()
    original_hash = sha(source)
    info = probe(source)
    fps = float(Fraction(info['fps']))
    scale = 3840 / max(info['width'], info['height'])
    width, height = [max(32, round(info[k] * scale / 32) * 32) for k in ('width', 'height')]
    plan = chunks(info['frames'])
    write('recipe.json', dict(recipe='refine-details-v1', source_sha256=original_hash, source=info, chunks=plan, width=width, height=height, seed=2081561474))
    paths = root / 'model-paths.json'
    write(paths.name, {'enhancement': {'is_default': True, **{role: str(model_root / role) for role in ('diffusion_models', 'text_encoders', 'vae', 'loras')}}})
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD'] += ' --cache-none --deterministic --extra-model-paths-config ' + shlex.quote(str(paths))
    results = []
    progress('Preparing bounded refinement chunks')
    with studio.runtime('comfy', root):
        import urllib.request
        with urllib.request.urlopen(studio.COMFY_URL + '/object_info', timeout=60) as response:
            node_info = json.load(response)
        with _client(studio.VIDEO_URL) as client:
            write('submission-start.json', {'wall_time': time.time()})
            for i, (first, last, length) in enumerate(plan):
                part = root / f'chunk-{i+1}'
                part.mkdir(exist_ok=True)
                progress(f'Refining section {i+1} of {len(plan)}. About 45–55 minutes for an 8-second film.')
                vf = f'trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS'
                if length > last-first:
                    vf += f',tpad=stop_mode=clone:stop={length-(last-first)}'
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-vf', vf, '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '0', '-pix_fmt', 'yuv420p', str(part/'input.mp4')], check=True)
                g = graph(client.assets.from_file(str(part/'input.mp4')), length, width, height, fps, f'earthfall/refine_{root.name}_{i+1}')
                for node in g.values():
                    if node['class_type'] not in node_info:
                        raise ValueError('Missing ComfyUI node: '+node['class_type'])
                write(f'chunk-{i+1}/graph.json', g)
                tick = time.monotonic()
                saved = _submit(client, g, ('490',), part/'generation')
                result = dict(video=str(Path(saved['490']).resolve()), seconds=time.monotonic()-tick)
                results.append(result)
                write(f'chunk-{i+1}/result.json', result)
    stages.update(refine='done', assemble='running')
    progress('Joining refined sections with the original timing')
    # Trim each independently: decoding all 4K sections together exceeds host RAM.
    trimmed = []
    for i, ((first,last,_), result) in enumerate(zip(plan, results)):
        path = root / f'join-{i}.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-i',result['video'],'-vf',f'trim=start_frame={int(i>0)}:end_frame={last-first},setpts=PTS-STARTPTS','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p',str(path)],check=True)
        trimmed.append(path)
    listing = root/'concat.txt'
    listing.write_text(''.join(f"file 'join-{i}.mp4'\n" for i in range(len(trimmed))))
    final = root/'refined-4k.mp4'
    command = ['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-i',str(source),'-map','0:v:0','-map','1:a?','-c','copy','-movflags','+faststart',str(final)]
    subprocess.run(command,check=True)
    stages.update(assemble='done', verify='running')
    progress('Checking frame count, timing and original file')
    output = probe(final)
    if (output['width'], output['height']) != (width, height) or output['frames'] != info['frames'] or Fraction(output['fps']) != Fraction(info['fps']) or sha(source) != original_hash:
        raise ValueError('Source preservation or frame-clock check failed')
    write('outcome.json', dict(video=final.name, video_sha256=sha(final), source_sha256=original_hash, source_frames=info['frames'], output_frames=output['frames'], source_fps=info['fps'], output_fps=output['fps'], clock_preserved=True, width=width, height=height, seconds=time.monotonic()-started, chunks=results, manual_review='Check the face, motion and chunk boundaries before sharing.'))
    stages['verify']='done'
    progress('Enhanced film ready. Original unchanged.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.check:
        check(); print('Refine Details models and runtime configuration ready.'); return
    if not args.input or not args.output:
        parser.error('--input and --output are required')
    run(args.input.resolve(),args.output.resolve())

if __name__=='__main__':main()
