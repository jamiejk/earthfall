"""Fixed delivery clock for the existing LTX split/Prompt Relay recovery path."""
from __future__ import annotations
import json
import copy
from pathlib import Path

FPS = 24
BOUNDS = (0, 29, 58, 108, 156, 192)
NAMES = ('orbit', 'clouds', 'high_aerial', 'approach', 'settle')
CONTEXT_STRATEGIES = ('token-spans-v1', 'isolated-phases-v1')


def configure(cfg, strategy='token-spans-v1'):
    """One product clock and frozen baseline settings for app and experiments.

    Context strategy is a named implementation experiment, never a timing override.
    Existing archived configurations are not mutated.
    """
    if strategy not in CONTEXT_STRATEGIES:
        raise ValueError('Unknown temporal context strategy')
    cfg = copy.deepcopy(cfg)
    contract = json.loads((Path(__file__).with_name('presets')/'split-phase-v1.json').read_text())
    contract.update(version='split-phase-v2', implementation={'relay_context': strategy})
    cfg.setdefault('video', {}).update(flight_mode='split', split=copy.deepcopy(contract['split']), phase_contract=contract)
    return cfg


def context_strategy(o):
    return o.get('phase_contract', {}).get('implementation', {}).get('relay_context', 'token-spans-v1')


def validate_options(o):
    c = o['phase_contract']
    if c['version'] not in ('split-phase-v1', 'split-phase-v2') or c['fps'] != FPS or tuple(c['bounds']) != BOUNDS:
        raise ValueError('Fixed phase clock changed')
    if context_strategy(o) not in CONTEXT_STRATEGIES:
        raise ValueError('Unknown temporal context strategy')
    if c['regional_altitude_ft'] != 30000 or c['regional_altitude_m'] != 9144:
        raise ValueError('Regional view must be 30,000 ft / 9,144 m')
    if c['engine'] != 'ltx-2.5-prompt-relay' or c['hold_trimming'] or c['retiming']:
        raise ValueError('Original conditioned engine and unmodified delivery clock required')
    for key, value in c['split'].items():
        if o.get(key) != value:
            raise ValueError(f'Fixed recovery setting changed: {key}')


def expected_beats(stage, o):
    if stage not in ('parent', 'descent'):
        raise ValueError('Unknown scheduled generation stage')
    limit = 193 + o['parent_handles'] if stage == 'parent' else o['descent_frames']
    cursor, result = 0, []
    for i in reversed(range(len(NAMES))):
        length = BOUNDS[i+1]-BOUNDS[i] + (i == 4) + (o['parent_handles'] if i == 0 and stage == 'parent' else 0)
        length = min(length, limit-cursor)
        if length <= 0:
            break
        result.append((NAMES[i], cursor, cursor+length))
        cursor += length
    if cursor != limit:
        raise ValueError('Incomplete phase clock')
    return result


def validate_timing(timing, o, stage):
    validate_options(o)
    beats = timing['generation_beats']
    actual = [(b['name'], b['start_frame'], b['end_frame']) for b in beats]
    expected = expected_beats(stage, o)
    if actual != expected or timing['fps'] != FPS or timing['source_frames'] != expected[-1][2]:
        raise ValueError('Phase boundaries do not map to the fixed delivery clock')
    if timing['temporal_stride'] != 8 or timing['epsilon'] != .1:
        raise ValueError('Prompt Relay timing parameters changed')
    if timing.get('context_strategy', 'token-spans-v1') != context_strategy(o):
        raise ValueError('Temporal context strategy differs from the declared experiment')
    if timing['single_prompt'] != timing['global_prompt']+' '+' '.join(b['prompt'] for b in beats):
        raise ValueError('Per-phase instructions were replaced by another prompt')
    for b in beats:
        text = b['prompt']
        if b['name'] == 'high_aerial' and ('30,000' not in text or '9,144' not in text or 'two kilometres' in text.lower()):
            raise ValueError('Regional phase has incorrect altitude instructions')
        if b['name'] == 'orbit' and 'never reveal an isolated whole globe' not in text:
            raise ValueError('Close orbital framing restriction missing')
        if b['name'] == 'clouds' and 'WHOLE frame' not in text:
            raise ValueError('Opaque physical cloud passage instruction missing')


def validate_graph(graph, timing, o, stage):
    validate_timing(timing, o, stage)
    relay = graph.get('380', {})
    r = relay.get('inputs', {})
    if relay.get('class_type') != 'TouchdownPromptRelay' or r.get('enabled') is not True:
        raise ValueError('Enabled Prompt Relay is mandatory')
    if json.loads(r['timing_json']) != timing or r.get('model') != ['300',0] or r.get('latent') != ['308',0]:
        raise ValueError('Prompt Relay is bound to different timing/model/latent')
    if graph['307']['inputs']['positive'] != ['380',1]:
        raise ValueError('Phase text is disconnected from conditioning')
    guiders = [n for n in graph.values() if n['class_type'] == 'CFGGuider']
    if not guiders or any(n['inputs']['model'] != ['380',0] for n in guiders):
        raise ValueError('Sampler bypasses temporal conditioning')
    latent = graph['308']['inputs']
    if (latent['length'],latent['width'],latent['height']) != (timing['source_frames'],o['width'],o['height']):
        raise ValueError('Generation dimensions or clock changed')
    videos = [n for n in graph.values() if n['class_type'] == 'CreateVideo']
    if not videos or any(n['inputs']['fps'] != FPS for n in videos):
        raise ValueError('Generation frame rate changed')
    if any(n['class_type'] in ('LTXICLoRALoaderModelOnly','LTXAddVideoICLoRAGuide','WanImageToVideo') for n in graph.values()):
        raise ValueError('Alternative engine/layout conditioning is forbidden')
    pins = {n['inputs']['frame_idx']:n['inputs'] for n in graph.values() if n['class_type']=='LTXVAddGuide'}
    expected = {0, 192-o['high_delivery_frame'], -1} if stage=='parent' else {0,192-o['anchor_delivery_frame'],o['descent_frames']-8}
    if set(pins) != expected:
        raise ValueError('Required reference/guide assignments changed')
    image_nodes = {p:v['image'] for p,v in pins.items()}
    expected_images = ({0:['51',0],192-o['high_delivery_frame']:['45',0],-1:['46',0]} if stage=='parent'
                       else {0:['51',0],192-o['anchor_delivery_frame']:['45',0],o['descent_frames']-8:['53',0]})
    if image_nodes != expected_images:
        raise ValueError('Reference pixels connected at incorrect phases')
    centres = [0]+[8*i-3.5 for i in range(1,(o['prefix_frames']-1)//8+1)]
    expected_centres = (centres+[192-o['high_delivery_frame'],timing['source_frames']-1] if stage=='parent'
                        else centres+[192-o['anchor_delivery_frame'],o['descent_frames']-8+3.5])
    if json.loads(r['guide_frames']) != expected_centres:
        raise ValueError('Appended guide planes are mapped to incorrect phase times')
    high = 192-o['high_delivery_frame'] if stage=='parent' else 192-o['anchor_delivery_frame']
    if pins[high]['strength'] != (o['high_strength'] if stage=='parent' else o['anchor_strength']):
        raise ValueError('Reference strength changed')


def validate_mapping(mapping, prefix_frames):
    """Every surviving generated/native frame keeps its original delivery position."""
    if len(mapping) < BOUNDS[-1]:
        raise ValueError('Delivery truncates the scheduled settle phase')
    native_seen = False
    for d,(source,index) in enumerate(mapping):
        if source in ('parent','descent'):
            if native_seen or index != BOUNDS[-1]-d:
                raise ValueError('Seam selection or frame deletion shifted the delivery clock')
        elif source == 'native':
            native_seen = True
            if index != d-(BOUNDS[-1]-(prefix_frames-1)):
                raise ValueError('Native portrait moved off the delivery clock')
        else:
            raise ValueError('Unknown delivery frame source')
    if not native_seen:
        raise ValueError('Native portrait missing')
