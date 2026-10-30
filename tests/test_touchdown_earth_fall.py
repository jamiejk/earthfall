import math

import numpy as np
import pytest

from touchdown.earth_fall import descent_pose
from touchdown.earth_fall_clouds import CloudField, opening_pose, rays, sphere_hit, RADIUS


def test_descent_is_high_fast_diagonal_and_fixed_lens():
    poses=[descent_pose(t) for t in np.linspace(0,1,120)]
    points=np.array([pose['position'] for pose in poses])
    delta=np.diff(points,axis=0)
    assert points[0,2]>590
    assert np.all(delta[:,1]>0) and np.all(delta[:,2]<0)
    assert np.allclose(-delta[:,2],delta[:,1])  # actual 45-degree travel
    assert np.all(points[:,0]==0)  # one heading, no lateral turns
    assert {pose['focal_length_mm'] for pose in poses}=={35.}
    assert np.linalg.norm(delta[0])>20  # >480 m/s initially at 24 fps
    assert np.linalg.norm(delta[-1])<.001
    assert points[-1,2]==pytest.approx(1.52)
    direction=np.array(poses[0]['target'])-points[0]
    assert math.degrees(math.atan2(-direction[2],direction[1]))==pytest.approx(45,abs=.2)


def test_opening_moves_in_space_with_constant_lens():
    first=opening_pose(0);last=opening_pose(1)
    assert first['position'][2]==350000
    assert last['position'][2]==2000
    assert last['position'][1]>first['position'][1]
    assert first['focal_length_mm']==last['focal_length_mm']==35
    assert last['pitch_deg']==45
    with pytest.raises(ValueError):opening_pose(1.1)


def test_sphere_intersections_distinguish_ground_and_space():
    direction=np.array([[[0,0,-1],[0,0,1]]],np.float32)
    distances=sphere_hit([0,0,350000],direction,RADIUS)
    assert distances[0,0]==pytest.approx(350000,abs=.1)
    assert np.isinf(distances[0,1])


def test_cloud_exit_depends_on_camera_height_and_is_opaque_at_splice():
    field=CloudField();direction=rays(64,36,45)
    _,trans=field.transport([0,-600,598.72],direction,400,900,layers=40,tile_m=5000)
    assert trans.max()<.001
    colour,clear=field.transport([0,-350,348.72],direction,400,900,layers=40,tile_m=5000)
    assert np.all(clear==1) and np.all(colour==0)
    last=opening_pose(1)
    _,inside=field.transport(last['position'],direction,1200,3800,spherical=True,layers=40)
    assert inside.max()<.001


def test_ragged_cloud_exit_clears_spatially_instead_of_uniformly():
    field=CloudField();direction=rays(128,72,45)
    _,inside=field.transport([0,-600,598.72],direction,250,900,layers=48,ragged_base=400)
    assert inside.max()<.001
    _,edge=field.transport([0,-401.28,400],direction,250,900,layers=48,ragged_base=400)
    assert edge.std()>.15
    assert edge.max()-edge.min()>.5
    _,clear=field.transport([0,-240,238.72],direction,250,900,layers=48,ragged_base=400)
    assert np.all(clear==1)
