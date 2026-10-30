from pathlib import Path
import numpy as np
from touchdown.backward_ramp import reproject,basis


def test_reprojection_identity_preserves_pixels_and_sky():
    y,x=np.mgrid[:32,:48];image=np.stack([x*4,y*5,x+y],axis=-1).astype(np.uint8)
    depth=np.full((32,48),10.,np.float32);depth[:8]=np.nan
    pose={'camera_xyz_m':[0,0,0],'aim_xyz_m':[0,1,0]}
    result,coverage=reproject(image,image,depth,pose,pose,depth)
    assert np.array_equal(result,image)
    assert coverage>.9


def test_reprojection_uses_camera_translation_and_fills_newly_exposed_region():
    guide=np.zeros((32,48,3),np.uint8);image=np.full_like(guide,200);depth=np.full((32,48),10.,np.float32)
    pose={'camera_xyz_m':[0,0,0],'aim_xyz_m':[0,1,0]}
    moved={'camera_xyz_m':[2,0,0],'aim_xyz_m':[2,1,0]}
    result,coverage=reproject(image,guide,depth,pose,moved)
    assert 0.6<coverage<0.9
    assert not result[:,0].any()
    assert result[10,30,0]==200


def test_reprojection_rejects_surface_occluded_in_later_view():
    guide=np.full((32,48,3),30,np.uint8);image=np.full_like(guide,200)
    depth=np.full((32,48),10.,np.float32);closer=np.full((32,48),2.,np.float32)
    pose={'camera_xyz_m':[0,0,0],'aim_xyz_m':[0,1,0]}
    result,coverage=reproject(image,guide,depth,pose,pose,closer)
    assert coverage==0
    assert np.array_equal(result,guide)


def test_backward_edit_ramps_model_conditioning_and_starts_from_geometry(monkeypatch,tmp_path):
    from contextlib import nullcontext
    import fakeout_earth_zoom as f
    from touchdown.comfy_job import backward_edit
    captured={}
    monkeypatch.setattr(f,'_client',lambda url:nullcontext(object()))
    monkeypatch.setattr(f,'_load',lambda graph,client,node,path:[node,0])
    def submit(client,graph,wanted,folder):
        captured.update(graph)
        return {'107':tmp_path/'result.png'}
    monkeypatch.setattr(f,'_submit',submit)
    backward_edit('local',str(tmp_path),'guide.png','later.png',.3,.7,123,'test')
    assert captured['104']['class_type']=='DualCFGGuider'
    assert captured['104']['inputs']['cfg_conds']==.3
    assert captured['104']['inputs']['cfg_cond2_negative']==1
    assert captured['104']['inputs']['cond2']==['100r0p',0]
    assert captured['104']['inputs']['cond1']==['100r1p',0]
    assert captured['105']['inputs']['latent_image']==['181',0]
    assert captured['182']['inputs']['denoise']==.7
    captured.clear()
    backward_edit('local',str(tmp_path),'guide.png','later.png',.8,1,123,'test',appearance_only=True)
    assert captured['104']['inputs']['cond1']==['190',0]
    assert captured['190']['inputs']['conditioning']==['100',0]
    assert captured['190']['inputs']['latent']==['100r1e',0]
