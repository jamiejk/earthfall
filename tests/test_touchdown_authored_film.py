import numpy as np
import pytest

from touchdown.authored_film import cloud_mask, spur_pose, surface


def test_spur_camera_clears_route_and_settles_in_front_of_person():
    poses=[spur_pose(t) for t in np.linspace(0,1,180)]
    points=np.array([p['position'] for p in poses])
    assert np.all(np.diff(points[:,1])>0)
    assert min(z-surface(x,y) for x,y,z in points)>1.2
    assert np.linalg.norm(points[-1]-points[-2])<.003
    assert abs(points[-1,0])<.2
    assert points[-1,1]==pytest.approx(-2.8)
    assert poses[-1]['target'][1]==0
    assert surface(0,0)==0
    with pytest.raises(ValueError):spur_pose(-.1)


def test_cloud_conceals_plate_changes_and_is_deterministic():
    # The exit starts fully opaque at progress .70, then reveals the set.
    _,alpha=cloud_mask(160,90,.70)
    assert alpha.min()>.99
    first=cloud_mask(160,90,.5)
    second=cloud_mask(160,90,.5)
    assert np.array_equal(first[0],second[0])
    assert np.array_equal(first[1],second[1])
    for p in [0.,1.]:
        assert cloud_mask(160,90,p)[1].max()==0
