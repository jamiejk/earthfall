"""Overhead images must pass observable camera rules, not a generic yes/no review."""
from touchdown.texture_gate import accepted

def test_overhead_policy_rejects_oblique_landmarks_and_uncertainty():
    good=dict(camera_view='straight_down',broad_surface_texture=True,horizon='no_sky',large_landmark_visible=False,building_side_faces_visible=False)
    assert accepted(good)
    for key,value in [('camera_view','oblique'),('camera_view','unclear'),('horizon','sky_meets_ground'),('horizon','unclear'),('large_landmark_visible',True),('building_side_faces_visible',True),('broad_surface_texture',False)]:
        assert not accepted({**good,key:value})


def test_normalisation_cannot_pass_scene_composition_into_prompts(tmp_path,monkeypatch):
    import json
    import cv2,numpy as np
    from contextlib import contextmanager
    from touchdown import canonical_texture as texture,studio_brief,studio_build
    (tmp_path/'flight/run/motion').mkdir(parents=True)
    writer=cv2.VideoWriter(str(tmp_path/'flight/run/motion/earth_zoom.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),24,(64,64))
    for _ in range(70):writer.write(np.zeros((64,64,3),dtype=np.uint8))
    writer.release()
    layout=dict(cloud_exit_frame=64,target_uv=[.5,.4],colours={'background':'#697b50','land':'#697b50','water':'#2e5665'},features=[{'material':'water','polygon':[[0,0],[1,0],[1,1]]}],layout_description='Pyramids in the background and foreground dunes',expansion_description='Add more pyramids in the distance')
    (tmp_path/'scene-layout.json').write_text(json.dumps(layout))
    @contextmanager
    def runtime(*args):yield
    monkeypatch.setattr(studio_build,'runtime',runtime)
    monkeypatch.setattr(studio_brief,'chat',lambda *args,**kwargs:dict(status='pass',visible_broad_water=False,material='desert',reason='The image shows dry desert terrain without water.'))
    texture.normalise(tmp_path)
    result=json.loads((tmp_path/'scene-layout.json').read_text())
    assert result['features']==[] and result['colours']['background']=='#b7a17b'
    assert result['target_uv']==layout['target_uv'] and result['cloud_exit_frame']==64
    for text in (result['layout_description'],result['expansion_description']):
        assert 'pyramid' not in text.lower() and 'foreground' not in text.lower() and 'background' not in text.lower()
