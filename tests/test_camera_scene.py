import numpy as np
import pytest
from touchdown.camera_scene import camera_pose,ground_height,project,START,END,FRAMES


def test_camera_respects_original_clock_altitude_and_fixed_lens():
    poses=[camera_pose(d) for d in range(START,END+1)]
    assert (START,END,FRAMES)==(96,192,97)
    assert camera_pose(108)['altitude_above_local_ground_m']==pytest.approx(9144)
    assert all(p['focal_length_mm']==35 for p in poses)
    assert all(p['delivery_frame']/24==p['time_seconds'] for p in poses)
    assert all(p['position'][2]-ground_height(*p['position'][:2])>=1.61 for p in poses)
    assert np.all(np.diff([p['position'][1] for p in poses])>0)
    for bad in [95,193]:
        with pytest.raises(ValueError):camera_pose(bad)


def test_approach_finishes_with_smooth_deceleration_and_real_foreground_pass():
    poses=[camera_pose(d) for d in range(108,193)]
    delta=np.linalg.norm(np.diff([p['position'] for p in poses],axis=0),axis=1)
    assert delta[-1]<.01 and delta[0]>1000
    # At the settle boundary motion continues, without a stationary photo hold.
    assert .1<delta[47]<.6 and .1<delta[48]<.6
    point=[-2.5,-12,ground_height(-2.5,-12)+.8]
    projections=[project(point,p) for p in poses]
    visible=[q for q in projections if q is not None and 0<q[0]<1 and 0<q[1]<1]
    assert len(visible)>=4
    assert projections[-1] is None  # the camera has actually passed it
    near=np.array(visible[-4:])[:,:2]
    assert np.max(np.linalg.norm(np.diff(near,axis=0),axis=1))>.01


def test_final_view_keeps_head_and_knees_and_crops_below_them():
    pose=camera_pose(192);z=ground_height(0,0)
    head=project([0,0,z+1.72],pose)
    knee=project([0,0,z+.5],pose)
    foot=project([0,0,z],pose)
    assert 0<head[0]<1 and 0<head[1]<1
    assert 0<knee[1]<.2 and foot[1]<0


def test_changed_or_rejected_photographic_anchors_cannot_enter_treatment(tmp_path):
    import json
    from touchdown.camera_scene import require_photo_review
    from touchdown.authored_scene import digest
    for name in ['guide.mp4','photo-first.png','photo-last.png']:(tmp_path/name).write_bytes(name.encode())
    record=dict(status='pass',guide_sha256=digest(tmp_path/'guide.mp4'),
        first_sha256=digest(tmp_path/'photo-first.png'),last_sha256=digest(tmp_path/'photo-last.png'))
    path=tmp_path/'photo-anchor-review.json';path.write_text(json.dumps(record))
    require_photo_review(tmp_path)
    record['status']='fail';path.write_text(json.dumps(record))
    with pytest.raises(ValueError):require_photo_review(tmp_path)
    record['status']='pass';path.write_text(json.dumps(record))
    (tmp_path/'photo-last.png').write_bytes(b'changed output')
    with pytest.raises(ValueError):require_photo_review(tmp_path)
