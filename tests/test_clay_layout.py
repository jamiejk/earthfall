import copy
import numpy as np
import pytest

from touchdown.clay_layout import adult_landmarks,HEIGHT,HEAD_HEIGHT,KEYFRAMES,ground_height,camera_pose
from touchdown.clay_layout_trial import build_graph,validate_graph,require_photos


def test_adult_is_normally_proportioned_and_limbs_keep_length_through_action():
    assert HEIGHT==1.78 and 7.5<HEIGHT/HEAD_HEIGHT<8
    for frame in range(96,193):
        p=adult_landmarks(frame)
        for side in ['left','right']:
            assert p[f'{side}_foot'][2]-.04==pytest.approx(ground_height(0,0))
            assert np.linalg.norm(np.subtract(p[f'{side}_elbow'],p[f'{side}_shoulder']))==pytest.approx(.29)
            # The forearm has a small fixed depth displacement, not changing length.
            assert np.linalg.norm(np.subtract(p[f'{side}_wrist'],p[f'{side}_elbow']))==pytest.approx(np.hypot(.25,.025))
            assert p[f'{side}_hip'][2]-p[f'{side}_ankle'][2]==pytest.approx(.82)
        assert camera_pose(frame)['time_seconds']==frame/24
    assert adult_landmarks(192)['right_hand'][0]>adult_landmarks(156)['right_hand'][0]+.35


def graph():return build_graph(1399629552,{'guide':'guide',**{i:f'photo{i}' for i,_ in KEYFRAMES}})


def test_all_five_stage_photos_are_pinned_in_both_passes():
    g=graph();validate_graph(g,1399629552)
    assert [i for i,_ in KEYFRAMES]==[0,12,36,60,96]
    assert sum(n['class_type']=='LTXVAddGuide' for n in g.values())==10


@pytest.mark.parametrize('node,key,value',[
    ('313','frame_idx',36),('345','frame_idx',48),('346','image',['p060c',0]),
    ('340','strength',.5),('308','length',89),('360','noise_seed',42),('370','samples',['395',2]),
])
def test_missing_or_mistimed_stage_guides_cannot_enter_inference(node,key,value):
    g=copy.deepcopy(graph());g[node]['inputs'][key]=value
    with pytest.raises(ValueError):validate_graph(g,1399629552)


def test_rejected_photo_review_blocks_generation_before_hash_checks(tmp_path):
    import json
    from touchdown.authored_scene import digest
    for name in ['guide.mp4','geometry-audit.json','photograph-plan.json']:(tmp_path/name).write_bytes(name.encode())
    (tmp_path/'raw-guide-review.json').write_text(json.dumps(dict(status='pass',guide_sha256=digest(tmp_path/'guide.mp4'),geometry_sha256=digest(tmp_path/'geometry-audit.json'))))
    (tmp_path/'photo-anchor-review.json').write_text(json.dumps(dict(status='fail')))
    with pytest.raises(ValueError):require_photos(tmp_path)
