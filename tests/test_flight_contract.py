import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from touchdown import flight_contract as contract, split_flight as flight

PRESET = Path(flight.__file__).with_name('presets')/'split-phase-v1.json'


def configuration():
    c=json.loads(PRESET.read_text())
    return dict(place='Test landmark',video=dict(split=copy.deepcopy(c['split']),phase_contract=c,framing='three-quarter'))


def test_original_forward_clock_and_reverse_phase_prompts():
    cfg=configuration();o=flight.options(cfg)
    assert flight.BOUNDS == contract.BOUNDS == (0,29,58,108,156,192)
    assert flight.FPS == 24
    for stage,build in [('parent',flight.parent_timing),('descent',flight.descent_timing)]:
        timing=build(cfg,o);contract.validate_timing(timing,o,stage)
        high=next(b for b in timing['generation_beats'] if b['name']=='high_aerial')
        assert '30,000 FEET (9,144 METRES)' in high['prompt']
        assert 'microscopic' in high['prompt']
    assert 'no roof' in flight.parent_timing(cfg,o)['single_prompt']
    assert cfg['place'] not in next(b for b in flight.parent_timing(cfg,o)['generation_beats'] if b['name']=='orbit')['prompt']


@pytest.mark.parametrize('change',['speed','duration','engine','trim','high_pin','layout'])
def test_recovery_refuses_contract_changes_before_inference(change):
    cfg=configuration();c=cfg['video']['phase_contract']
    if change=='speed':c['fps']=16
    if change=='duration':c['bounds'][-1]=80
    if change=='engine':c['engine']='wan'
    if change=='trim':c['hold_trimming']=True
    if change=='high_pin':cfg['video']['split']['high_strength']=.8
    if change=='layout':cfg['video']['split']['parent_layout']=True
    with pytest.raises(ValueError):flight.options(cfg)


def scheduled_graph(tmp_path):
    import fakeout_earth_zoom as f
    cfg=configuration();o=flight.options(cfg);timing=flight.parent_timing(cfg,o)
    graph={}
    f._ltx(graph,SimpleNamespace(no_upscale=True),241,[(0,['51',0]),(128,['45',0]),(-1,['46',0])],timing['single_prompt'],123,'test')
    flight.set_strength(graph,128,.5)
    graph['308']['inputs'].update(width=1280,height=736)
    flight.wire_relay(graph,timing,tmp_path/'audit.json',flight.guide_times(0,25)+[128,240],enabled=True)
    return graph,timing,o


def test_real_ltx_graph_uses_relay_and_phase_pins(tmp_path):
    graph,timing,o=scheduled_graph(tmp_path)
    contract.validate_graph(graph,timing,o,'parent')


@pytest.mark.parametrize('change',['relay_off','model_bypass','text_bypass','fps','clock','pin'])
def test_actual_submitted_graph_cannot_bypass_phase_contract(tmp_path,change):
    graph,timing,o=scheduled_graph(tmp_path)
    if change=='relay_off':graph['380']['inputs']['enabled']=False
    if change=='model_bypass':next(n for n in graph.values() if n['class_type']=='CFGGuider')['inputs']['model']=['300',0]
    if change=='text_bypass':graph['307']['inputs']['positive']=['305',0]
    if change=='fps':next(n for n in graph.values() if n['class_type']=='CreateVideo')['inputs']['fps']=16
    if change=='clock':graph['308']['inputs']['length']=81
    if change=='pin':next(n for n in graph.values() if n['class_type']=='LTXVAddGuide' and n['inputs']['frame_idx']==128)['inputs']['image']=['46',0]
    with pytest.raises(ValueError):contract.validate_graph(graph,timing,o,'parent')


def mapping():
    return [('parent',192-d) for d in range(72)]+[('descent',192-d) for d in range(72,168)]+[('native',j) for j in range(37)]


def test_complete_delivery_keeps_every_phase_and_native_frame_on_clock():
    contract.validate_mapping(mapping(),25)


@pytest.mark.parametrize('change',['drop','duplicate','shorten','shift_seam','speed_native'])
def test_assembly_cannot_hide_a_bad_flight_by_moving_the_timeline(change):
    m=mapping()
    if change=='drop':m.pop(32)
    if change=='duplicate':m.insert(32,m[32])
    if change=='shorten':m=m[:180]
    if change=='shift_seam':m[72]=('descent',119)
    if change=='speed_native':m[-1]=('native',38)
    with pytest.raises(ValueError):contract.validate_mapping(m,25)


def test_seam_search_only_accepts_pairs_on_the_same_clock(monkeypatch):
    monkeypatch.setattr(flight,'_seam',lambda a,b:dict(acceptable=True,max_displacement_at_640px=0,mean_absolute_error=0))
    monkeypatch.setattr(flight,'_cost',lambda m:0)
    found=flight.far_seam(lambda d:d,list(range(121)),72,fixed_clock=True)
    assert found['q']==192-found['P']


def test_appended_guides_must_keep_their_original_clock(tmp_path):
    graph,timing,o=scheduled_graph(tmp_path)
    graph['380']['inputs']['guide_frames']='[0, 8, 16, 24, 128, 240]'
    with pytest.raises(ValueError,match='guide planes'):contract.validate_graph(graph,timing,o,'parent')


def test_recovery_keeps_existing_portrait_and_phase_instructions():
    from touchdown.phase_recovery import configuration as recover
    cfg=dict(seed=1,place='Another place',prompts=dict(arrival='unchanged arrival',near='unchanged action',high='old'),video=dict(framing='three-quarter'))
    before=copy.deepcopy(cfg);new=recover(cfg)
    assert cfg==before
    assert new['prompts']['arrival']==cfg['prompts']['arrival']
    assert new['prompts']['near']==cfg['prompts']['near']
    assert '30,000 FEET (9,144 METRES)' in new['prompts']['high']
    assert flight.options(new)['parent_layout'] is False


def test_right_order_at_wrong_times_does_not_pass_schedule_review():
    from touchdown.flight_sequence import sequence_verdict
    policy=json.loads(PRESET.with_name('earth-camera-v1.json').read_text())['review']
    stages=['orbit']*4+['cloud_cover']*4+['high_regional']*7+['detailed_ground']*18
    rows=[]
    for i,stage in enumerate(stages):
        rows.append(dict(frame=i*6,time=i*.25,observation=dict(view=stage,
            curved_planet_limb_visible=stage=='orbit',near_clouds_fill_frame=stage=='cloud_cover',
            broad_connected_region_visible=stage=='high_regional',nearby_landmark_or_person_dominates=stage=='detailed_ground',
            detail_scale={'orbit':'continental','cloud_cover':'obscured','high_regional':'regional','detailed_ground':'person'}[stage],
            superimposed_different_scenes=False)))
    assert sequence_verdict(rows,policy)['status']=='pass'
    policy['delivery_contract']=json.loads(PRESET.read_text())
    # Compress the same ordered stages into the first three seconds, then extend ground.
    for r in rows:r['time']/=2
    rows+= [dict(rows[-1],frame=200,time=8.333)]
    verdict=sequence_verdict(rows,policy)
    assert verdict['status']=='fail'
    assert any('fixed frame interval' in f for f in verdict['failures'])


def test_real_descent_graph_keeps_prefix_anchor_and_parent_seam_on_clock(tmp_path):
    import fakeout_earth_zoom as f
    cfg=configuration();o=flight.options(cfg);timing=flight.descent_timing(cfg,o);graph={}
    f._ltx(graph,SimpleNamespace(no_upscale=True),121,[(0,['51',0]),(72,['45',0]),(113,['53',0])],timing['single_prompt'],124,'test')
    flight.set_strength(graph,72,.5)
    graph['308']['inputs'].update(width=1280,height=736)
    flight.wire_relay(graph,timing,tmp_path/'audit.json',flight.guide_times(0,25)+[72]+flight.guide_times(113,8),enabled=True)
    contract.validate_graph(graph,timing,o,'descent')


def test_exact_scheduled_observations_pass_fixed_window_review():
    from touchdown.flight_sequence import sequence_verdict
    policy=json.loads(PRESET.with_name('earth-camera-v1.json').read_text())['review']
    policy['delivery_contract']=json.loads(PRESET.read_text());rows=[]
    for i in range(33):
        d=i*6;stage='orbit' if d<29 else 'cloud_cover' if d<58 else 'high_regional' if d<108 else 'detailed_ground'
        rows.append(dict(frame=d,time=d/24,observation=dict(view=stage,
            curved_planet_limb_visible=stage=='orbit',near_clouds_fill_frame=stage=='cloud_cover',
            broad_connected_region_visible=stage=='high_regional',nearby_landmark_or_person_dominates=stage=='detailed_ground',
            detail_scale={'orbit':'continental','cloud_cover':'obscured','high_regional':'regional','detailed_ground':'person'}[stage],
            superimposed_different_scenes=False)))
    verdict=sequence_verdict(rows,policy)
    assert verdict['status']=='pass',verdict['failures']
    assert all(c['passed'] for c in verdict['phase_checks'])


def test_failed_retained_regional_phase_blocks_parent_gate_despite_good_ending():
    from touchdown.phase_review import opening_verdict
    def obs(stage):
        return dict(view=stage,curved_planet_limb_visible=stage=='orbit',near_clouds_fill_frame=stage=='cloud_cover',
            broad_connected_region_visible=stage=='high_regional',nearby_landmark_or_person_dominates=stage=='detailed_ground',
            detail_scale={'orbit':'continental','cloud_cover':'obscured','high_regional':'regional','detailed_ground':'buildings'}[stage],superimposed_different_scenes=False)
    phases=[dict(phase=name,expected_view=view,start_frame=lo,end_frame=hi,observations=[obs(view)]*3)
            for name,view,lo,hi in [('orbit','orbit',0,29),('clouds','cloud_cover',29,58),('high','high_regional',58,72)]]
    travel=[dict(boundary=29,observation=dict(transition='camera_travel',cloud_depth_or_changing_perspective_visible=True,different_scene_appears_by_opacity_change=False))]
    assert opening_verdict(phases,travel)['status']=='pass'
    phases[-1]['observations']=[obs('detailed_ground')]*3
    verdict=opening_verdict(phases,travel)
    assert verdict['status']=='fail'
    assert '[58, 72)' in verdict['failures'][0]


def test_actual_runner_stops_before_anchor_or_descent_when_parent_gate_fails(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from touchdown import phase_recovery as recovery,studio_build as studio,flight_sequence as sequence
    cfg=configuration();root=tmp_path/'run';(root/'scene').mkdir(parents=True)
    (root/'scene/high.png').write_bytes(b'regional reference bytes')
    m=dict(identity=dict(config=cfg,source_hashes={}),model_paths={})
    monkeypatch.setattr(recovery,'prepare',lambda *a:m)
    monkeypatch.setattr(recovery,'asset_plan',lambda *a:[dict(name='high')])
    monkeypatch.setattr(recovery,'generate_asset',lambda *a:None)
    monkeypatch.setattr(recovery,'verify_lineage',lambda *a:None)
    monkeypatch.setattr(flight,'enable_prompt_relay',lambda:dict(status='ready'))
    from touchdown import flight_runtime
    monkeypatch.setattr(flight_runtime,'capture_runtime',lambda *a:{})
    monkeypatch.setenv('TOUCHDOWN_COMFY_SERVER_CMD','python main.py')
    monkeypatch.setenv('PYTHONPATH','scripts:src')
    @contextmanager
    def runtime(*a):yield dict(name='test')
    monkeypatch.setattr(studio,'runtime',runtime)
    monkeypatch.setattr(sequence,'observed_chat',lambda *a:dict(view='high_regional',broad_connected_region_visible=True,
        nearby_landmark_or_person_dominates=False,curved_planet_limb_visible=False,near_clouds_fill_frame=False,
        detail_scale='regional',superimposed_different_scenes=False))
    called=[]
    monkeypatch.setattr(flight,'parent',lambda *a:called.append('parent'))
    monkeypatch.setattr(flight,'descent',lambda *a:called.append('descent'))
    monkeypatch.setattr(recovery,'review_opening',lambda *a:dict(status='fail',failures=['Missing regional phase']))
    with pytest.raises(studio.Held,match='Missing regional'):
        recovery.run(tmp_path/'source',root,[])
    assert called==['parent']
    assert json.loads((root/'status.json').read_text())['stage']=='parent_review'


def test_conditioning_pair_changes_only_context_implementation():
    source=configuration()
    a=contract.configure(source,'token-spans-v1')
    b=contract.configure(source,'isolated-phases-v1')
    ta=flight.parent_timing(a,flight.options(a));tb=flight.parent_timing(b,flight.options(b))
    assert ta.pop('context_strategy')=='token-spans-v1'
    assert tb.pop('context_strategy')=='isolated-phases-v1'
    assert ta==tb
    a['video']['phase_contract']['implementation']=b['video']['phase_contract']['implementation']
    assert a==b


def test_app_holds_failed_full_sequence_before_arrival_or_publication(tmp_path,monkeypatch):
    from touchdown import studio_build as studio,phase_review
    cfg=contract.configure(configuration())
    def failed(root,cfg):return dict(status='fail',verdict=dict(failures=['Generated dissolve']))
    monkeypatch.setattr(phase_review,'review_final',failed)
    outcome={}
    with pytest.raises(studio.Held,match='Generated dissolve'):
        studio.review_delivery(cfg,tmp_path,{},tmp_path,tmp_path/'motion',tmp_path/'near',outcome,lambda *a:None)
    assert outcome['flight_sequence_review']['status']=='fail'
    assert not outcome.get('checks_passed')


def test_held_parent_preview_preserves_source_clock_and_source_bytes(tmp_path):
    import cv2
    import numpy as np
    import subprocess
    from touchdown.phase_review import parent_preview
    from touchdown.authored_scene import digest
    folder=tmp_path/'motion/flight/parent';folder.mkdir(parents=True)
    source=folder/'source.mp4';graph=folder/'graph.json';graph.write_text('{}')
    enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','96x64',
        '-r','24','-i','pipe:0','-c:v','libx264','-crf','12',str(source)],stdin=subprocess.PIPE)
    for i in range(241):enc.stdin.write(np.full((64,96,3),i,np.uint8).tobytes())
    enc.stdin.close();assert enc.wait()==0
    original=digest(source)
    (folder/'result.json').write_text(json.dumps(dict(video=str(source),sha256=original,
        workflow=str(graph),workflow_sha256=digest(graph))))
    target=parent_preview(tmp_path)
    cap=cv2.VideoCapture(str(target))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==193 and cap.get(cv2.CAP_PROP_FPS)==24
    for d in [0,64,128,192]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,d);ok,image=cap.read();assert ok
        assert abs(float(image.mean())-(192-d))<4
    cap.release()
    assert digest(source)==original
    report=json.loads((tmp_path/'diagnostics/preview.json').read_text())
    assert report['source_mapping']==list(range(192,-1,-1))
    assert report['scope']=='Parent diagnostic, not delivery'
