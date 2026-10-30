"""Observe the complete orbital/cloud/regional/ground sequence, independently of its prompt.

These local-model observations are fallible. A passing report is a review candidate,
not certification of a physical altitude or of every intervening frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import cv2
from pydantic import BaseModel, ConfigDict, Field

from touchdown.authored_scene import digest
from touchdown.flight_regression import read, write

PRESET = Path(__file__).with_name('presets') / 'earth-camera-v1.json'
ORDER = ('orbit', 'cloud_cover', 'high_regional', 'detailed_ground')


def observed_chat(url, images, prompt, schema, output, metadata):
    """Resume only observations made with the same pixels, prompt, schema, code and model."""
    from touchdown.studio_brief import chat
    identity = dict(images=[dict(label=label, sha256=digest(path)) for path, label in images],
                    prompt=prompt, schema=schema.model_json_schema(), model=metadata,
                    code_sha256=digest(Path(__file__)))
    if output.exists() and read(output).get('observation_identity') == identity:
        return read(output)['answer']
    answer = chat(url, images, prompt, schema, output)
    write(output, {**read(output), 'observation_identity': identity})
    return answer


class FrameObservation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    view: Literal['orbit', 'cloud_cover', 'high_regional', 'detailed_ground', 'unclear']
    curved_planet_limb_visible: bool
    near_clouds_fill_frame: bool
    broad_connected_region_visible: bool
    nearby_landmark_or_person_dominates: bool
    detail_scale: Literal['continental', 'regional', 'buildings', 'person', 'obscured', 'unclear']
    superimposed_different_scenes: bool
    evidence: str = Field(min_length=5, max_length=1000)


class TransitionObservation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    transition: Literal['camera_travel', 'dissolve', 'wipe', 'cut', 'unclear']
    cloud_depth_or_changing_perspective_visible: bool
    different_scene_appears_by_opacity_change: bool
    evidence: str = Field(min_length=5, max_length=1200)


def classify(observation):
    """A requested label cannot override incompatible observed scale or an overlay."""
    o = observation
    if o.get('superimposed_different_scenes'):
        return 'unclear'
    view = o.get('view')
    if view == 'orbit' and o.get('curved_planet_limb_visible'):
        return view
    if (view == 'cloud_cover' and o.get('near_clouds_fill_frame')
            and not o.get('curved_planet_limb_visible') and not o.get('broad_connected_region_visible')
            and o.get('detail_scale') == 'obscured'):
        return view
    if (view == 'high_regional' and o.get('broad_connected_region_visible')
            and not o.get('nearby_landmark_or_person_dominates')
            and not o.get('curved_planet_limb_visible')
            and not o.get('near_clouds_fill_frame')
            and o.get('detail_scale') == 'regional'):
        return view
    if view == 'detailed_ground' and o.get('detail_scale') in ('buildings', 'person'):
        return view
    return 'unclear'


def sequence_verdict(rows, policy):
    """Require consecutive evidence of all four stages, in order, including a readable regional interval."""
    if not rows or any(b['frame'] <= a['frame'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Frame evidence must be nonempty, distinct and chronological')
    stages = [classify(row['observation']) for row in rows]
    runs = []
    for i, stage in enumerate(stages):
        if stage == 'unclear':
            continue
        if runs and runs[-1]['stage'] == stage and runs[-1]['last_index'] == i - 1:
            runs[-1].update(last_index=i, end=rows[i]['time'])
        else:
            runs.append(dict(stage=stage, first_index=i, last_index=i, start=rows[i]['time'], end=rows[i]['time']))
    minimum = dict(orbit=policy['minimum_orbit_seconds'], cloud_cover=policy['minimum_cloud_seconds'],
                   high_regional=policy['minimum_regional_seconds'], detailed_ground=0)
    qualified = [r for r in runs if r['end'] - r['start'] + 1e-6 >= minimum[r['stage']]]
    chosen, failures, cursor, anchored = [], [], -1, True
    for stage in ORDER:
        if not anchored:
            failures.append(f'{stage} is not established after a complete preceding sequence')
            continue
        run = next((r for r in qualified if r['stage'] == stage and r['first_index'] > cursor), None)
        if run is None:
            failures.append(f'Missing or too brief {stage} after preceding stage')
            anchored = False
        else:
            chosen.append(run)
            cursor = run['last_index']
    confident = [s for s in stages if s != 'unclear']
    if confident and confident[0] != 'orbit':
        failures.append('First established view is not orbital')
    ranks = [ORDER.index(s) for s in confident]
    if any(b < a for a, b in zip(ranks, ranks[1:])):
        failures.append('Observed stages move backwards in the required sequence')
    if any(row['observation'].get('superimposed_different_scenes') for row in rows):
        failures.append('Superimposed scenes observed')
    phase_checks = []
    if policy.get('delivery_contract'):
        clock = policy['delivery_contract']
        if rows[-1]['time'] < (clock['bounds'][-1]-1)/clock['fps']:
            failures.append('Video truncates the fixed delivery schedule')
        expected = ('orbit','cloud_cover','high_regional','detailed_ground','detailed_ground')
        for name,lo,hi in zip(expected,clock['bounds'][:-1],clock['bounds'][1:],strict=True):
            # Inspect inside each fixed interval, allowing one LTX latent at its boundaries.
            inside = [i for i,r in enumerate(rows) if (lo+8)/clock['fps'] <= r['time'] < (hi-8)/clock['fps']]
            matching = [i for i in inside if stages[i] == name]
            passed = len(matching) >= 2
            phase_checks.append(dict(stage=name,start_frame=lo,end_frame=hi,matching_samples=len(matching),passed=passed))
            if not passed:
                failures.append(f'{name} lacks evidence inside its fixed frame interval [{lo}, {hi})')
    return dict(status='pass' if not failures else 'fail', failures=failures,
                classified_frames=stages, runs=runs, selected_runs=chosen, phase_checks=phase_checks,
                altitude_scope='Regional appearance only; exact 30,000-ft altitude cannot be certified from generated images')


def transition_verdict(observation):
    return (observation['transition'] == 'camera_travel'
            and observation['cloud_depth_or_changing_perspective_visible']
            and not observation['different_scene_appears_by_opacity_change'])


def review(video, folder, policy=None, url=None):
    from touchdown import studio_build as studio
    from touchdown.studio_brief import MODEL, MODEL_DIGEST
    policy = policy or read(PRESET)['review']
    url = url or studio.VISION_URL
    video, folder = video.resolve(), folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    video_hash = digest(video)
    identity = dict(video=str(video), video_sha256=video_hash, policy=policy, code_sha256=digest(Path(__file__)),
                    model_name=MODEL, model_digest=MODEL_DIGEST)
    final = folder / 'report.json'
    if final.exists() and read(final).get('identity') == identity:
        return read(final)
    cap = cv2.VideoCapture(str(video))
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0 or count < 2:
        cap.release()
        raise ValueError('Cannot decode flight')
    step = max(1, round(policy['sample_seconds'] * fps))
    wanted = set(range(0, count, step)) | {count - 1}
    frames, index = [], 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if index in wanted:
                path = folder / 'frames' / f'{index:05d}.jpg'
                path.parent.mkdir(exist_ok=True)
                height = round(image.shape[0] * min(1, 1280 / image.shape[1]))
                width = round(image.shape[1] * height / image.shape[0])
                cv2.imwrite(str(path), cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
                frames.append(dict(frame=index, time=index / fps, image=str(path), sha256=digest(path)))
            index += 1
    finally:
        cap.release()
    if index != count:
        raise ValueError('Short flight decode')
    rows = []
    prompt = ('Describe only the actual image. Classify the camera view and independently report its visual evidence. '
              'Orbit requires a curved planetary limb. Cloud cover requires nearby opaque cloud filling the view. '
              'High regional means many kilometres of connected terrain/water patterns with local landmarks microscopic, '
              'no dominant nearby building or person, and no readable facade details. A castle with a distant horizon '
              'is detailed_ground, not high_regional. Distinguish physical cloud occlusion from two transparent scenes '
              'superimposed. Use unclear when evidence is insufficient. Do not infer a requested stage or altitude.')
    with studio.runtime('vision', folder) as metadata:
        identity['model'] = metadata
        for frame in frames:
            output = folder / 'observations' / f"{frame['frame']:05d}.json"
            observation = observed_chat(url, [(Path(frame['image']), 'OBSERVED FRAME')], prompt, FrameObservation, output, metadata)
            rows.append(dict(**frame, observation=observation))
            write(folder / 'progress.json', dict(observed=len(rows), total=len(frames)))
        verdict = sequence_verdict(rows, policy)
        transitions = []
        if verdict['status'] == 'pass':
            for left, right in zip(verdict['selected_runs'], verdict['selected_runs'][1:]):
                lo, hi = max(0, left['last_index'] - 1), min(len(rows), right['first_index'] + 2)
                selected = rows[lo:hi]
                # Bound context while retaining both sides of a long unresolved interval.
                if len(selected) > 8:
                    selected = [selected[round(i * (len(selected)-1) / 7)] for i in range(8)]
                images = [(Path(r['image']), f"TIME {r['time']:.3f}s") for r in selected]
                observation = observed_chat(url, images,
                    'These are chronological frames from one video, not a proposed storyboard. Does the transition '
                    'show continuous camera travel with changing perspective or clouds passing near the lens, or '
                    'does one picture become transparent and another appear beneath it? Look for duplicate horizons '
                    'and landscape features remaining static while opacity changes. A whiteout alone does not prove '
                    'travel. If these samples cannot establish the mechanism, use unclear.',
                    TransitionObservation, folder / 'transitions' / f"{left['stage']}-{right['stage']}.json", metadata)
                transitions.append(dict(from_stage=left['stage'], to_stage=right['stage'],
                                        observation=observation, acceptable=transition_verdict(observation)))
            if policy['require_transition_travel'] and not all(t['acceptable'] for t in transitions):
                verdict['status'] = 'fail'
                verdict['failures'].append('At least one scale transition lacks evidence of camera travel')
    if digest(video) != video_hash:
        raise ValueError('Flight changed during review')
    # Model digest is saved separately so cache identity also remains checkable offline.
    result = dict(identity={k:v for k,v in identity.items() if k != 'model'}, model=identity['model'],
                  status=verdict['status'], verdict=verdict, frames=rows, transitions=transitions,
                  advisory=True, human_playback_review='required', fps=fps, count=count)
    write(final, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.video, args.output)
    print(json.dumps(dict(status=result['status'], verdict=result['verdict']), indent=2))
    return 0 if result['status'] == 'pass' else 2


if __name__ == '__main__':
    raise SystemExit(main())
