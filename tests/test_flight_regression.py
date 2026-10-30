import copy

import pytest

from touchdown import flight_regression as regression


def test_control_pins_settings_without_mutating_saved_source(monkeypatch):
    monkeypatch.delenv('TOUCHDOWN_SPLIT', raising=False)
    source = {'seed': 123, 'video': {'split': {'parent_layout': True}, 'framing': 'three-quarter'}}
    before = copy.deepcopy(source)
    control = regression.configuration(source)
    assert source == before
    assert control['seed'] == source['seed']
    assert control['video']['framing'] == source['video']['framing']
    assert all(control['video']['split'][k] == v for k, v in regression.CONTROL.items())
    assert control['video']['split']['high_join'] == 'parent'


def test_environment_cannot_silently_change_control(monkeypatch):
    monkeypatch.setenv('TOUCHDOWN_SPLIT', '{"parent_layout": true}')
    with pytest.raises(ValueError, match='Unset TOUCHDOWN_SPLIT'):
        regression.configuration({'video': {}})


def test_frozen_preset_does_not_follow_changed_live_defaults(monkeypatch):
    monkeypatch.delenv('TOUCHDOWN_SPLIT', raising=False)
    monkeypatch.setitem(regression.flight.DEFAULTS, 'prefix_frames', 73)
    monkeypatch.setitem(regression.flight.DEFAULTS, 'high_join', 'cloud')
    cfg = regression.configuration({'video': {}})
    assert cfg['video']['split']['prefix_frames'] == 25
    assert cfg['video']['split']['high_join'] == 'parent'


def test_new_options_require_an_explicit_preset_revision(monkeypatch):
    monkeypatch.delenv('TOUCHDOWN_SPLIT', raising=False)
    monkeypatch.setitem(regression.flight.DEFAULTS, 'future_camera_option', True)
    with pytest.raises(ValueError, match='Frozen preset lacks new split settings'):
        regression.configuration({'video': {}})


@pytest.mark.parametrize('relative', ['.', 'child', '..'])
def test_experiment_cannot_write_inside_or_over_source(tmp_path, relative):
    source = tmp_path / 'source'
    with pytest.raises(ValueError, match='separate'):
        regression.check_root(source, source / relative)


def test_changed_source_is_refused(tmp_path):
    image = tmp_path / 'arrival.png'
    image.write_bytes(b'original')
    snapshot = {str(image): regression.digest(image)}
    regression.verify_snapshot(snapshot)
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Reused source changed'):
        regression.verify_snapshot(snapshot)


def test_approved_reference_cannot_be_resumed_for_generation(tmp_path, monkeypatch):
    root = tmp_path / 'approved'
    root.mkdir()
    acceptance = root / 'acceptance.json'
    acceptance.write_text('{"status":"approved_reference"}')
    before = acceptance.read_bytes()

    def unexpected_prepare(*args):
        raise AssertionError('An approved run must be refused before preparation or runtime changes')

    monkeypatch.setattr(regression, 'prepare', unexpected_prepare)
    with pytest.raises(ValueError, match='Approved reference runs are frozen'):
        regression.run(tmp_path / 'source', root)
    assert acceptance.read_bytes() == before
