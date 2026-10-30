import copy
import hashlib

import pytest

from touchdown import camera_flight as camera
from touchdown import flight_sequence as sequence


def observation(stage):
    return dict(view=stage, curved_planet_limb_visible=stage == 'orbit',
                near_clouds_fill_frame=stage == 'cloud_cover',
                broad_connected_region_visible=stage == 'high_regional',
                nearby_landmark_or_person_dominates=stage == 'detailed_ground',
                detail_scale={'orbit':'continental', 'cloud_cover':'obscured',
                              'high_regional':'regional', 'detailed_ground':'person'}[stage],
                superimposed_different_scenes=False, evidence='Observed scene features')


def rows(stages):
    return [dict(frame=i*6, time=i*.25, observation=observation(stage)) for i, stage in enumerate(stages)]


POLICY = sequence.read(sequence.PRESET)['review']


def test_required_flight_order_with_readable_region():
    evidence = rows(['orbit']*3 + ['cloud_cover']*3 + ['high_regional']*3 + ['detailed_ground']*3)
    verdict = sequence.sequence_verdict(evidence, POLICY)
    assert verdict['status'] == 'pass'
    assert [r['stage'] for r in verdict['selected_runs']] == list(sequence.ORDER)


def test_cloud_exit_straight_to_castle_fails_even_with_good_ground_ending():
    evidence = rows(['orbit']*3 + ['cloud_cover']*4 + ['detailed_ground']*12)
    verdict = sequence.sequence_verdict(evidence, POLICY)
    assert verdict['status'] == 'fail'
    assert any('high_regional' in f for f in verdict['failures'])


def test_one_fleeting_high_frame_does_not_establish_regional_view():
    evidence = rows(['orbit']*3 + ['cloud_cover']*3 + ['high_regional'] + ['detailed_ground']*3)
    assert sequence.sequence_verdict(evidence, POLICY)['status'] == 'fail'


def test_wrong_order_and_opacity_overlay_are_rejected():
    evidence = rows(['orbit']*3 + ['high_regional']*3 + ['cloud_cover']*3 + ['detailed_ground']*3)
    assert sequence.sequence_verdict(evidence, POLICY)['status'] == 'fail'
    evidence = rows(['orbit']*3 + ['cloud_cover']*3 + ['high_regional']*3 + ['detailed_ground']*3)
    evidence[4]['observation']['superimposed_different_scenes'] = True
    assert sequence.sequence_verdict(evidence, POLICY)['status'] == 'fail'


def test_high_label_cannot_override_a_dominant_castle_or_facade_detail():
    o = observation('high_regional')
    o['nearby_landmark_or_person_dominates'] = True
    assert sequence.classify(o) == 'unclear'
    o['nearby_landmark_or_person_dominates'] = False
    o['detail_scale'] = 'buildings'
    assert sequence.classify(o) == 'unclear'


def test_opaque_clouds_cannot_count_as_clear_regional_view_and_visible_region_is_not_full_cloud():
    o = observation('high_regional')
    o['near_clouds_fill_frame'] = True
    assert sequence.classify(o) == 'unclear'
    o = observation('cloud_cover')
    o['broad_connected_region_visible'] = True
    assert sequence.classify(o) == 'unclear'


@pytest.mark.parametrize('kind', ['dissolve', 'wipe', 'cut', 'unclear'])
def test_scene_transition_labels_cannot_pass_as_travel(kind):
    assert not sequence.transition_verdict(dict(transition=kind,
        cloud_depth_or_changing_perspective_visible=True, different_scene_appears_by_opacity_change=False))


def test_travel_requires_depth_evidence_and_no_opacity_replacement():
    base = dict(transition='camera_travel', cloud_depth_or_changing_perspective_visible=True,
                different_scene_appears_by_opacity_change=False)
    assert sequence.transition_verdict(base)
    assert not sequence.transition_verdict({**base, 'cloud_depth_or_changing_perspective_visible':False})
    assert not sequence.transition_verdict({**base, 'different_scene_appears_by_opacity_change':True})


def test_camera_prompt_uses_scene_content_and_explicit_30000ft_stage():
    cfg = dict(place='A different landmark', words=dict(region='A different region', support='connected grass'))
    before = copy.deepcopy(cfg)
    prompt = camera.flight_prompt(cfg, sequence.read(sequence.PRESET))
    assert 'A different landmark' in prompt and 'A different region' in prompt
    assert '30,000 FEET' in prompt and '9,144 METRES' in prompt
    assert prompt.index('regional interval') < prompt.index('physically enter the high cloud deck')
    assert cfg == before


def test_weight_fingerprint_records_content_and_stat(tmp_path):
    weight = tmp_path / 'weight.safetensors'
    weight.write_bytes(b'known model content')
    result = camera.fingerprint(weight)
    assert result['sha256'] == hashlib.sha256(weight.read_bytes()).hexdigest()
    assert result['size'] == weight.stat().st_size and result['path'] == str(weight.resolve())


def test_versioned_short_recipe_is_scene_generic_and_keeps_scale_order():
    preset = sequence.read(camera.PRESET.with_name('earth-camera-v2-short.json'))
    prompt = camera.flight_prompt(dict(place='Test place', words=dict(region='Test region', support='shore')), preset)
    assert 'Test place' in prompt and 'Test region' in prompt
    assert prompt.index('30,000 FEET') < prompt.index('thick opaque cloud layer') < prompt.index('End in orbit')
    assert preset['review'] == POLICY


def test_audit_graph_serializes_sdk_assets_without_replacing_submission_objects():
    class Asset:
        def __str__(self):
            return 'arrival.png'
    asset = Asset()
    graph = {'44':{'class_type':'LoadImage', 'inputs':{'image':asset}}}
    saved = camera.graph_snapshot(graph)
    assert saved['44']['inputs']['image'] == 'arrival.png'
    assert graph['44']['inputs']['image'] is asset


def test_recipe_snapshot_preserves_original_code_after_working_copy_changes(tmp_path):
    from touchdown.recipe_snapshot import capture
    source = tmp_path/'module.py'; source.write_text('original code')
    run = tmp_path/'run'; run.mkdir()
    camera.write(run/'experiment.json',dict(identity=dict(code_hashes={str(source):camera.digest(source)})))
    record = capture(run)[0]
    source.write_text('later code')
    assert capture(run) == [record]
    assert (run/record['captured']).read_text() == 'original code'
    (run/record['captured']).write_text('damaged snapshot')
    with pytest.raises(ValueError,match='Captured recipe code changed'):
        capture(run)


def test_changed_recipe_cannot_be_first_captured_as_the_original(tmp_path):
    from touchdown.recipe_snapshot import capture
    source = tmp_path/'module.py'; source.write_text('original code')
    run = tmp_path/'run'; run.mkdir()
    camera.write(run/'experiment.json',dict(identity=dict(code_hashes={str(source):camera.digest(source)})))
    source.write_text('later code')
    with pytest.raises(ValueError,match='Cannot capture changed recipe source'):
        capture(run)


@pytest.mark.parametrize('changed', ['pixels','policy'])
def test_portrait_assembly_rejects_stale_review_before_loading_sources(tmp_path, changed):
    (tmp_path/'flight-forward.mp4').write_bytes(b'new flight pixels')
    manifest = dict(identity=dict(source=str(tmp_path/'source'),preset=dict(review=POLICY)))
    identity = dict(video_sha256=camera.digest(tmp_path/'flight-forward.mp4'),policy=copy.deepcopy(POLICY))
    if changed == 'pixels':
        identity['video_sha256'] = 'old hash'
    else:
        identity['policy']['minimum_regional_seconds'] = 0
    report = dict(status='pass',identity=identity)
    with pytest.raises(ValueError,match='review is stale'):
        camera.assemble(tmp_path,manifest,report)


def test_observation_cache_is_bound_to_pixels_prompt_and_model(tmp_path, monkeypatch):
    from touchdown import studio_brief
    image=tmp_path/'frame.jpg'; image.write_bytes(b'pixels')
    output=tmp_path/'observation.json'; calls=[]
    def fake_chat(url,images,prompt,schema,path):
        answer=observation('orbit'); calls.append(prompt)
        camera.write(path,dict(answer=answer))
        return answer
    monkeypatch.setattr(studio_brief,'chat',fake_chat)
    args=('unused',[(image,'FRAME')],'observe',sequence.FrameObservation,output,dict(digest='model-a'))
    sequence.observed_chat(*args); sequence.observed_chat(*args)
    assert len(calls)==1
    sequence.observed_chat(*args[:2],'new prompt',*args[3:])
    assert len(calls)==2
    sequence.observed_chat(*args[:-1],dict(digest='model-b'))
    assert len(calls)==3
    image.write_bytes(b'changed pixels')
    sequence.observed_chat(*args[:-1],dict(digest='model-b'))
    assert len(calls)==4
