"""Photographic flight joins must agree and keep both source workflows verifiable."""
import json
import pytest
from test_photo_delivery import sources, encode, texture
from touchdown.authored_scene import digest
from touchdown.photo_delivery import JoinMismatch, compose
from touchdown.photo_segments import assemble


def parts(output):
    upper=output/'upper.mp4';encode(upper,[texture()]*5)
    graph=output/'upper.json';graph.write_text('{}')
    a=dict(video=str(upper),sha256=digest(upper),workflow=str(graph),workflow_sha256=digest(graph))
    b=json.loads((output/'flight/result.json').read_text())
    return a,b


def test_two_flights_meet_without_blending_and_keep_source_ancestry(sources):
    root,near=sources;a,b=parts(root);output=root/'assembly'
    r=assemble(a,b,near,output)
    assert r['source_high_frame']==4 and r['source_frames']==81
    assert r['seam']['acceptable'] and len(r['source_segments'])==2
    manifest=json.loads((output/'flight/assembly-workflows.json').read_text())
    assert not manifest['crossfade'] and not manifest['interpolation']
    delivery=compose(output,near,4)
    assert delivery['native_frames']==17


def test_a_different_high_view_cannot_be_hidden_at_the_join(sources):
    root,near=sources;a,b=parts(root)
    encode(root/'upper.mp4',[texture(2)]*5);a['sha256']=digest(root/'upper.mp4')
    with pytest.raises(JoinMismatch):assemble(a,b,near,root/'assembly')
    assert not (root/'assembly/flight/earth_zoom.mp4').exists()


def test_changing_a_segment_after_assembly_invalidates_delivery(sources):
    root,near=sources;a,b=parts(root);output=root/'assembly';assemble(a,b,near,output)
    (root/'upper.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError,match='segment changed'):compose(output,near,4)
