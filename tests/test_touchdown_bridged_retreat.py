import pytest
from touchdown.bridged_retreat_join import timeline, kept_indices


def test_bridge_join_drops_only_shared_endpoints_and_places_sound_ranges():
    base = {'opener': [0, 14], 'cloud': [14, 51], 'dive': [51, 135], 'swoop': [135, 280], 'end': [280, 352]}
    segments, count = timeline(base, 97, 73, 145)
    assert count == 313
    assert segments['dive'] == [51, 97]
    assert segments['swoop'] == [97, 313]
    assert segments['end'] == [265, 313]
    assert segments['cloud'] == base['cloud']


def test_bridge_rejects_cut_outside_map_section():
    with pytest.raises(ValueError, match='computed dive'):
        timeline({'dive': [51, 135]}, 180, 73, 145)


def test_retiming_keeps_both_endpoint_pins_even_when_step_does_not_divide_length():
    kept = kept_indices(73, 5)
    assert kept[0] == 0 and kept[-1] == 72
    assert len(set(kept)) == len(kept)
    assert kept[-2] == 70
    assert len(kept_indices(73, 3)) == 25
