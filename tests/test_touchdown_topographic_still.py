from contextlib import nullcontext

import pytest


@pytest.mark.parametrize('geometry_first', [True, False])
@pytest.mark.parametrize('photo_structure', [True, False])
def test_reference_order_matches_prompt_roles(monkeypatch, tmp_path, geometry_first, photo_structure):
    import fakeout_earth_zoom as f
    from touchdown.comfy_job import topographic_still

    loaded, captured = [], {}
    monkeypatch.setattr(f, '_client', lambda url: nullcontext(object()))

    def load(graph, client, node, path):
        loaded.append(str(path))
        return [node, 0]

    def submit(client, graph, wanted, folder):
        captured.update(graph)
        return {'107': tmp_path / 'result.png'}

    monkeypatch.setattr(f, '_load', load)
    monkeypatch.setattr(f, '_submit', submit)
    topographic_still('local', str(tmp_path), 'geometry.png',
                      ['platform.jpg', 'canyon.jpg'], 'subject.png', 123,
                      geometry_first=geometry_first, photo_structure=photo_structure)
    expected = (['geometry.png', 'platform.jpg', 'canyon.jpg', 'subject.png']
                if geometry_first else
                ['platform.jpg', 'canyon.jpg', 'geometry.png', 'subject.png'])
    assert loaded == expected
    prompt = captured['100']['inputs']['text']
    geo, platform, canyon = (1, 2, 3) if geometry_first else (3, 1, 2)
    if photo_structure:
        assert f'Image {platform} is the authoritative reference for the actual built lookout' in prompt
        assert f'Image {geo} is ONLY a coarse reference for distant canyon topography' in prompt
        assert f'Image {canyon} supplies real canyon rock layers' in prompt
        assert 'Image 4 supplies only the woman identity' in prompt
        assert 'Its entire foreground platform is an incorrect modelling placeholder' in prompt
        assert 'physical structure as well as materials' in prompt
    else:
        assert f'Image {geo} is a crude untextured geometric model' in prompt
        assert f'Image {platform} is a real photograph' in prompt
        assert f'Image {canyon} documents real layered rock textures' in prompt
        assert 'Image 4 contains ONLY the woman identity' in prompt
    assert captured['100r3s']['inputs']['image'] == ['43', 0]
    assert captured['100n']['inputs']['noise_seed'] == 123


def test_requires_both_material_references():
    from touchdown.comfy_job import topographic_still
    with pytest.raises(ValueError, match='platform and a canyon'):
        topographic_still('local', '.', 'geometry.png', [], 'subject.png', 123)
