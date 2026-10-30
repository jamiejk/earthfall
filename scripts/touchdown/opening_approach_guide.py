"""Opening-only experiment: a moving approach guide at the orbit end, instead of two stills.

The canonical opening pins the painted orbit still at frame 48 and a 1.8x crop of it at
frame 40 (strength 0.5), with the flight's own cloud frames at 0-8. Between two stills
LTX holds the orbit view and dissolves cloud over it (Giza, 28 Oct). A still is only a
soft prior; a video guide carries the camera. So the orbit end gets an 8-frame pull-back
made from the orbit still itself (the mechanism the flight already uses for its high
view): in retreat order the window grows from 1/zoom to the whole still, so played
forwards the surface expands and the limb slides out of the top of the frame. The 1.8x
crop moves to frame 33, keeping the chain of scale going toward the cloud.

Copies an existing opening's images, timing, frozen inference and seed; never changes the
canonical Create defaults. Preserve the source film.
"""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys

FRAMES = 49
APPROACH_FRAMES = 8          # a mid-clip video guide needs 8k frames starting at frame 8j+1
APPROACH_FRAME = FRAMES - APPROACH_FRAMES   # 41: frames 41..48 end on the orbit still
CLOSER_FRAME = 33


def driver():
    spec = importlib.util.spec_from_file_location(
        'recorded_opening_driver', Path(__file__).with_name('scene_grounded_workflow.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen(code):
    sys.path[:0] = [str(code / 'scripts'), str(code / 'src')]
    from touchdown import split_flight as flight
    assert Path(flight.__file__).is_relative_to(code)
    return flight


def prepare(source, root, zoom, closer_strength):
    g = driver()
    g.opening_verify(source / 'opening')
    root.mkdir(exist_ok=False)
    folder = root / 'opening'
    folder.mkdir()
    for name in ('frozen-code', 'pinned-model-paths.json', 'timing.json',
                 'grounded-orbit-full.png', 'orbit-start.png', 'closer-earth.png',
                 'cloud-retreat-prefix.mp4'):
        src, dst = source / 'opening' / name, folder / name
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            shutil.copy2(src, dst)
    plan = g.read(source / 'opening/experiment.json')
    assert plan['strengths'] == {'40': .5} and plan['generated_frames'] == FRAMES
    for name, expected in dict(plan['hashes']).items():
        path = Path(name)
        if path.is_relative_to(source / 'opening'):
            copied = folder / path.relative_to(source / 'opening')
            assert g.sha(copied) == expected
            plan['hashes'][str(copied)] = expected
    # The pull-back is cut from orbit-start.png (the 1280x736 aspect crop LTX is pinned to), so its
    # last frame is that still exactly and the first is a 1/zoom window on it.
    flight = frozen(folder / 'frozen-code')
    approach = folder / 'orbit-approach.mp4'
    flight.high_motion(folder / 'orbit-start.png', approach, APPROACH_FRAMES, zoom, 1280, 736)
    plan.update(strengths={str(CLOSER_FRAME): closer_strength},
                approach=dict(path=str(approach), frame=APPROACH_FRAME, frames=APPROACH_FRAMES, zoom=zoom,
                              source='orbit-start.png', purpose='A moving guide at the orbit end so the camera '
                              'is already travelling toward Earth; the limb leaves the top of the frame'),
                variant='opening-approach-guide-v1',
                comparison_source=str(source),
                changes=[f'Orbit end: 8-frame pull-back guide (zoom {zoom}) at frames {APPROACH_FRAME}-48 '
                         'replaces the single orbit still at 48',
                         f'1.8x crop moved from frame 40 to {CLOSER_FRAME}, strength {closer_strength}'],
                app_promoted=False)
    plan['hashes'][str(approach)] = g.sha(approach)
    plan['hashes'][str(Path(__file__).resolve())] = g.sha(Path(__file__).resolve())
    g.write(folder / 'experiment.json', plan)


def worker(root):
    g = driver()
    folder = root / 'opening'
    flight = frozen(folder / 'frozen-code')
    generate = flight._generate
    plan = g.read(folder / 'experiment.json')
    strength = plan['strengths'][str(CLOSER_FRAME)]

    def with_approach(f, client, graph, frames, pins, strengths, timing, guides,
                      seed, prefix, part, options):
        assert frames == FRAMES and strengths == {40: .5} and guides == [0, 4.5, 40, 48]
        prefix_pin, closer, _orbit = pins
        graph.update({'52': {'class_type': 'LoadVideo', 'inputs': {'file': client.assets.from_file(plan['approach']['path'])}},
                      '53': {'class_type': 'GetVideoComponents', 'inputs': {'video': ['52', 0]}}})
        pins = [prefix_pin, (CLOSER_FRAME, closer[1]), (APPROACH_FRAME, ['53', 0])]
        guides = [0, 4.5, CLOSER_FRAME] + flight.guide_times(APPROACH_FRAME, APPROACH_FRAMES)
        return generate(f, client, graph, frames, pins, {CLOSER_FRAME: strength}, timing,
                        guides, seed, prefix, part, options)

    flight._generate = with_approach
    g.opening_worker(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'worker', 'assemble'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--zoom', type=float, default=1.3)
    p.add_argument('--closer-strength', type=float, default=.5)
    a = p.parse_args()
    if a.stage == 'prepare':
        if a.source is None:
            p.error('--source is required for preparation')
        prepare(a.source.resolve(), a.root.resolve(), a.zoom, a.closer_strength)
    elif a.stage == 'worker':
        worker(a.root.resolve())
    else:
        driver().assemble(a.root.resolve())
