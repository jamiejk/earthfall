import pytest
from touchdown.anchored_approach import approach_schedule,output_to_source,START,LENGTH
from touchdown.relay_timing import schedule,query_positions


def test_five_second_photo_has_an_explicit_time_on_the_reversed_shorter_clip():
    assert output_to_source(120)==72  # 5 s on the complete film, not on the shorter clip
    assert output_to_source(72)==120
    assert output_to_source(191)==1
    assert LENGTH==121 and START==72
    for frame in (0,71,192):
        with pytest.raises(ValueError):output_to_source(frame)


def test_short_approach_preserves_every_retained_phase_and_excludes_orbit_cloud_prompts():
    cfg=dict(place='Stonehenge',prompts={});full=schedule(cfg);short=approach_schedule(cfg)
    assert short['source_frames']==121 and short['delivery_frames']==120
    assert [x['name'] for x in short['generation_beats']]==['settle','approach','high_aerial']
    assert short['generation_beats'][-1]['end_frame']==121
    assert 'solid immovable masonry' in short['global_prompt']
    assert short['single_prompt']==short['global_prompt']+' '+' '.join(b['prompt'] for b in short['generation_beats'])
    for delivery in range(72,192):
        source=output_to_source(delivery)
        expected=next(b['name'] for b in full['generation_beats'] if b['start_frame']<=source<b['end_frame'])
        actual=next(b['name'] for b in short['generation_beats'] if b['start_frame']<=source<b['end_frame'])
        assert actual==expected
    assert short['forward_beats'][0]['start_frame']==0
    assert short['forward_beats'][-1]['end_frame']==120


def test_middle_photo_does_not_stretch_the_approach_attention_clock():
    a=query_positions(16,1,21,[0,4.5,12.5,20.5,120])
    b=query_positions(16,1,22,[0,4.5,12.5,20.5,72,120])
    assert a[:16]==b[:16]
    assert b[-2:]==[72,120]
