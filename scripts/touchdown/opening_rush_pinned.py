"""One extra photographic scale pin for the isolated opening-rush experiment."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

DRIVER = Path(__file__).with_name('opening_rush.py')
spec = importlib.util.spec_from_file_location('opening_driver', DRIVER)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def prepare(root):
    import cv2
    base.prepare(root)
    image = cv2.imread(str(root / 'orbit-start.png'))
    h, w = image.shape[:2]
    scale, center = 1.8, (.5, .65)
    cw, ch = round(w / scale), round(h / scale)
    left, top = round((w-cw)*center[0]), round((h-ch)*center[1])
    cropped = image[top:top+ch, left:left+cw]
    target = root / 'closer-earth.png'
    cv2.imwrite(str(target), cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4))
    plan = base.read(root / 'experiment.json')
    plan.update(extra_photographic_guide=dict(path=str(target), generation_frame=40,
        delivery_frame=8, strength=1.0, scale=scale, crop_rectangle=[left, top, cw, ch],
        provenance='A single crop of the original photographic opening; no new scene or rendered viewport',
        purpose='Require surface expansion and horizon departure, not just moving cloud opacity'),
        comparison='Prompt-only opening test failed; one intermediate photographic scale constraint added')
    plan['hashes'].update({str(target): base.sha(target), str(Path(__file__).resolve()): base.sha(Path(__file__))})
    base.write(root / 'experiment.json', plan)


def worker(root):
    # Select the archived package before base.worker imports it. Only add one guide
    # to its graph; leave the graph builder and model code unchanged.
    for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
        key, value = line.split('=', 1)
        if key.startswith('TOUCHDOWN_'):
            os.environ[key] = json.loads(value)
    code = root / 'frozen-code'
    sys.path[:0] = [str(code / 'scripts'), str(code / 'src')]
    from touchdown import split_flight as flight
    generate = flight._generate

    def with_scale_pin(f, client, graph, frames, pins, strengths, timing, guides, seed, prefix, folder, options):
        assert frames == 49 and guides == [0, 4.5, 48]
        closer = f._load(graph, client, '47', root / 'closer-earth.png')
        return generate(f, client, graph, frames, [pins[0], (40, closer), pins[-1]],
            {40: 1.0}, timing, [0, 4.5, 40, 48], seed, prefix, folder, options)

    flight._generate = with_scale_pin
    base.worker(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'worker', 'assemble'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == 'assemble':
        base.assemble(root)
    else:
        globals()[args.stage](root)
