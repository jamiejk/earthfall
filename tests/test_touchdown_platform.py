from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer

from touchdown.platform_layout import project_layout, strip_mesh
from touchdown.terrain_quality import assess_surface_layout


def test_layout_keeps_pedestrian_geometry_and_marks_width_assumptions():
    lon, lat = -112.1079, 36.0616
    x,y = Transformer.from_crs('EPSG:4326', 'EPSG:26912', always_xy=True).transform(lon,lat)
    points = [{'lon': lon, 'lat': lat}, {'lon': lon+0.0001, 'lat': lat}]
    source = {'elements': [
        {'id': 1, 'tags': {'highway': 'footway', 'width': '4'}, 'geometry': points},
        {'id': 2, 'tags': {'highway': 'steps'}, 'geometry': points},
        {'id': 3, 'tags': {'highway': 'service'}, 'geometry': points},
        {'id': 4, 'tags': {'barrier': 'fence'}, 'geometry': points}]}
    result = project_layout(source, {'x': x, 'y': y}, 12)
    assert [path['id'] for path in result['paths']] == [1,2]
    assert result['paths'][0]['width_m'] == 4
    assert result['paths'][1]['width_source'].startswith('estimated')
    assert result['barriers'][0]['id'] == 4
    assert np.allclose(result['paths'][0]['points'][0], [0,0])


def test_surface_strip_follows_the_same_dem_height_on_both_edges():
    rows,cols = np.mgrid[:401,:401]
    height = (cols - 200).astype(np.float32) * 0.1
    vertices, faces = strip_mesh([[0,0],[10,0]], 4, height)
    assert faces and max(max(face) for face in faces) < len(vertices)
    assert np.allclose(vertices[0], [0,2,0.08])
    assert np.allclose(vertices[-1], [10,-2,0.58])


def test_shared_surface_gate_detects_missing_or_displaced_path(tmp_path: Path):
    mask = np.zeros((544,960), np.uint8)
    cv2.rectangle(mask, (100,300), (300,520), 255, -1)
    floor = np.full((544,960,3), (55,100,155), np.uint8)
    preserved = floor.copy()
    cv2.rectangle(preserved, (100,300), (300,520), (160,160,160), -1)
    moved = floor.copy()
    cv2.rectangle(moved, (140,300), (340,520), (160,160,160), -1)
    cv2.imwrite(str(tmp_path/'mask.png'),mask)
    for name, image in [('preserved',preserved),('missing',floor),('moved',moved)]:
        cv2.imwrite(str(tmp_path/f'{name}.png'),image)
    assert not assess_surface_layout(tmp_path/'preserved.png',tmp_path/'mask.png')['warnings']
    assert assess_surface_layout(tmp_path/'missing.png',tmp_path/'mask.png')['warnings']
    assert 'boundaries' in ' '.join(assess_surface_layout(tmp_path/'moved.png',tmp_path/'mask.png')['warnings'])


def test_walking_surface_bounds_grade_and_rejects_cliff_bridges():
    from touchdown.platform_layout import fit_walking_surface
    vertices, faces = strip_mesh([[0,0],[10,0]], 4, np.full((401,401), 10.0))
    vertices = [(x,y,z + x * 0.08) for x,y,z in vertices]
    fitted, kept, audit = fit_walking_surface(vertices, faces, (5,0))
    assert audit['grade'] <= 0.050001
    assert kept
    assert max(abs(a[2]-b[2]) for a,b in zip(fitted,vertices)) <= 0.6
    vertices[0] = (*vertices[0][:2], -20)
    _, kept, audit = fit_walking_surface(vertices, faces, (5,0))
    assert audit['rejected_vertices'] > 0
    assert all(0 not in face for face in kept)


def test_surface_sequence_checks_corresponding_masks(tmp_path, monkeypatch):
    from touchdown import terrain_quality as quality
    mask = np.zeros((544,960), np.uint8)
    cv2.rectangle(mask, (100,300), (300,520), 255, -1)
    floor = np.full((544,960,3), (155,100,55), np.uint8)  # RGB, coloured ground
    preserved = floor.copy()
    preserved[300:521,100:301] = 160
    for index in (0,16,32,48,64,80):
        cv2.imwrite(str(tmp_path/f'{index+1:04d}.png'), mask)
    monkeypatch.setattr(quality.V, 'frame', lambda clip,index: floor if index == 48 else preserved)
    result = quality.assess_surface_sequence(tmp_path/'clip.mp4', tmp_path)
    assert result['flagged_frames'] == [48]
    assert [s['frame'] for s in result['samples']] == [0,16,32,48,64,80]
