"""Native Fill expansion of a sky-free oblique surface seed for a rough globe guide."""
import argparse
import json
from pathlib import Path

from touchdown import orbital_reference as experiment


def run(root, index):
    experiment.environment()
    experiment.verify(root)
    from touchdown import fill_outpaint as fill
    import cv2
    if index == 1:
        image = cv2.imread(str(root / 'overhead.png'))
        # Preserve aspect ratio while excluding the failed nadir view's upper sky band.
        crop = image[128:1088, 113:1807]
        source = root / 'surface-seed.png'
        cv2.imwrite(str(source), cv2.resize(crop, (1920, 1088), interpolation=cv2.INTER_LANCZOS4))
        experiment.write(root / 'surface-policy.json', dict(source=str(root / 'overhead.png'),
            source_sha256=experiment.sha(root / 'overhead.png'), crop=[113, 128, 1807, 1088],
            surface_seed_sha256=experiment.sha(source), true_overhead=False,
            scope='Appearance prototype only; oblique terrain perspective remains approximate',
            expansion_runner_sha256=experiment.sha(Path(__file__))))
    else:
        source = root / f'expand-{index-1}/raw-outpaint.png'
    folder = root / f'expand-{index}'
    graph, photo = fill.prepare(source, folder)
    graph['23']['inputs']['text'] = (
        'Extend the existing aerial landscape photograph naturally beyond every edge, matching '
        'its existing camera direction, illumination, colours and connected terrain. The wider '
        'Scottish Highland region contains connected irregular blue sea lochs, branching inlets, '
        'green-brown mountain ridges and valleys. Continue existing water connections and shorelines '
        'smoothly into a much wider regional landscape. Keep all existing features small relative '
        'to the new wider region. Continuous land and water fill the entire image. No sky or '
        'horizon, no giant castle, no repeated island landmark, no person, no map labels, no '
        'borders, no inset photograph or collage. One coherent natural photographic surface.'
    )
    fill.validate(graph)
    experiment.write(folder / 'graph.api.json', graph)
    ui = json.loads((folder / 'workflow.json').read_text())
    next(n for n in ui['nodes'] if n['id'] == 23)['widgets_values'][0] = graph['23']['inputs']['text']
    experiment.write(folder / 'workflow.json', ui)
    experiment.write(root / 'status.json', dict(status=f'surface_outpaint_{index}_running', video_submissions=0))
    fill.run(folder, graph, photo)
    experiment.verify(root)
    experiment.write(root / 'status.json', dict(status=f'surface_outpaint_{index}_awaiting_review', video_submissions=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.index)
