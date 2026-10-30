from pathlib import Path
import numpy as np
from touchdown.connected_canyon import CanyonSurface
from touchdown.earth_fall_clouds import CloudField,rays


def test_surface_has_continuous_carved_sides_and_contact(tmp_path):
    # A flat source is sufficient to verify the authored connected spur and its
    # joins without relying on ignored geographic caches.
    np.save(tmp_path/'height_detail.npy',np.full((801,801),1900,np.float32))
    np.save(tmp_path/'height_far.npy',np.full((681,681),1900,np.float32))
    s=CanyonSurface(tmp_path)
    assert s.height(0,0)==0
    assert s.height(0,-30)==0
    assert s.height(70,-30)<-100
    assert s.height(0,40)<-100
    points=np.linspace(5,35,301)
    assert np.abs(np.diff(s.height(points,np.zeros_like(points)))).max()<2


def test_clouds_move_with_a_stationary_camera():
    field=CloudField();direction=rays(96,54,45)
    _,a=field.transport([2000,-500,700],direction,250,900,layers=40,ragged_base=400,time_s=0)
    _,b=field.transport([2000,-500,700],direction,250,900,layers=40,ragged_base=400,time_s=1)
    assert np.abs(a-b).mean()>.05
    _,same=field.transport([2000,-500,700],direction,250,900,layers=40,ragged_base=400,time_s=0)
    assert np.array_equal(a,same)
