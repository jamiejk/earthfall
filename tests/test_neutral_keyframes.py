import pytest

from touchdown.neutral_keyframes import pair_graph, validate_pair


@pytest.mark.parametrize('first', range(2, 8))
def test_adjacent_reference_recipe(first):
    g = pair_graph(first, first-1, 'wide.png', 'close.png')
    validate_pair(g, first)
    assert g['371']['inputs']['fps'] == 24
    assert g['370']['inputs']['samples'] == ['365', 2]


def test_neutral_close_reference_leaves_action_free():
    g = pair_graph(2, 1, 'wide.png', 'close.png')
    for node in ['313', '343']:
        assert g[node]['inputs']['frame_idx'] == 24
        assert g[node]['inputs']['strength'] == .5
    assert 'raises both arms' in g['305']['inputs']['text']
    g['344'] = dict(class_type='LTXVAddGuide', inputs=dict(frame_idx=-1))
    with pytest.raises(ValueError, match='must not be pinned'):
        validate_pair(g, 2)


def test_unit_test_cannot_silently_become_an_eight_second_film():
    g = pair_graph(3, 2, 'wide.png', 'close.png')
    g['308']['inputs']['length'] = 193
    with pytest.raises(ValueError, match='clock'):
        validate_pair(g, 3)
