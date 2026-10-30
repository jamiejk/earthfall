"""Two photographic flights meeting on one verified aerial frame; no dissolves."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import cv2
from touchdown.authored_scene import digest
from touchdown import portrait_flight
from touchdown.photo_delivery import frame_at, match_metrics, video_info, JoinMismatch


def assemble(upper_record, lower_record, near_folder, output):
    """Join exact corresponding high-view endpoints, then bind both source records."""
    parts = [upper_record, lower_record]
    for part in parts:
        if digest(Path(part['video'])) != part['sha256'] or digest(Path(part['workflow'])) != part['workflow_sha256']:
            raise ValueError('Changed photographic segment source')
    upper, lower = [cv2.VideoCapture(p['video']) for p in parts]
    try:
        count, width, height, fps = video_info(upper)
        lower_count, lw, lh, lf = video_info(lower)
        if fps != 24 or lf != 24 or (width,height)!=(lw,lh):
            raise ValueError('Photographic segments must share dimensions and 24 fps')
        seam = match_metrics(frame_at(upper,count-1),frame_at(lower,0))
        if not seam['acceptable']:
            raise JoinMismatch('The shared high aerial does not match closely enough; no dissolve was added')
        folder=output/'flight';folder.mkdir(parents=True,exist_ok=True);target=folder/'earth_zoom.mp4'
        encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}',
            '-r','24','-i','pipe:0','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
        try:
            for cap,start,end in [(upper,0,count),(lower,1,lower_count)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES,start)
                for i in range(start,end):
                    ok,frame=cap.read()
                    if not ok:raise ValueError('Short photographic segment')
                    encoder.stdin.write(frame.tobytes())
        finally:
            encoder.stdin.close();code=encoder.wait()
        if code:raise RuntimeError('Photographic assembly encoding failed')
        manifest=folder/'assembly-workflows.json'
        manifest.write_text(json.dumps(dict(kind='photographic-segment-assembly',sources=parts,
            shared_high_frame=count-1,seam=seam,crossfade=False,interpolation=False),indent=2))
        near=json.loads((near_folder/'result.json').read_text())
        (folder/'provenance.json').write_text(json.dumps(dict(parent=dict(near_video_sha256=near['sha256']),source_segments=parts),indent=2))
        record=dict(video=str(target),sha256=digest(target),descent=str(target),descent_sha256=digest(target),
            workflow=str(manifest),workflow_sha256=digest(manifest),source_segments=parts,source_high_frame=count-1,
            source_frames=count+lower_count-1,seam=seam)
        (folder/'result.json').write_text(json.dumps(record,indent=2));return record
    finally:
        upper.release();lower.release()


def run(cfg,scene,output,url,near_folder):
    import fakeout_earth_zoom as f
    upper=output/'upper';upper.mkdir(parents=True,exist_ok=True)
    prompt=('One continuous fast live-action camera flight from the supplied orbital photograph down to the supplied '
      'high aerial photograph of the same region. Begin physically travelling forward and downward, fixed focal length. '
      'Accelerate toward moving cloud tops, plunge INSIDE dense rolling cloud with strong near-lens parallax, briefly '
      'surrounded by moving vapour, then emerge BELOW the clouds still TWO KILOMETRES above the ground. '
      'Finish still flying diagonally forward and DOWN at about 45 degrees over a vast regional landscape. '
      'Terrain grows continuously in perspective; clouds rush past the lens. The last photo is a wide high aerial, '
      'never a low drone or ground-level view. Natural matching land colours and light. No people, no visible monuments, '
      'no giant structures on the planet, no optical zoom, no dissolve, no cut, no frozen photograph, no pause.')
    with f._client(url) as client:
        graph={};pins=[(0,f._load(graph,client,'44',scene/'orbit.png')),(-1,f._load(graph,client,'45',scene/'high.png'))]
        wanted=f._ltx(graph,SimpleNamespace(no_upscale=True),97,pins,prompt,cfg['seed']+33,'touchdown/photo_upper')
        graph['308']['inputs'].update(width=1280,height=736)
        saved=f._submit(client,graph,tuple(wanted),upper)
    video=Path(saved[f.LTX_VIDEO]);workflow=video.parent/'graph.json'
    ur=dict(video=str(video),sha256=digest(video),workflow=str(workflow),workflow_sha256=digest(workflow))
    (upper/'result.json').write_text(json.dumps(ur,indent=2))
    lower_cfg=copy.deepcopy(cfg)
    lower_cfg['video'].update(frames=cfg['video']['near_frames']+96,pin_high=False,flight_refine=False,flight_endpoint='high')
    lower_cfg['prompts']['prefix_flight']=(
      'Continue the supplied live-action moving portrait retreat into one continuous FAST camera flight BACKWARD AND UPWARD. '
      'Preserve the supplied moving portrait frames at the beginning. Immediately after that prefix accelerate backward '
      'and UP at about 45 degrees over the same connected open ground, with strong perspective and foreground parallax. '
      f"The setting is {cfg['gate']['description']} "
      'Keep one heading and the same landmark on the same side. The person shrinks quickly into an unresolvable speck. '
      'Reveal hundreds of metres, then several kilometres of continuous plausible regional surroundings. '
      'The camera finishes TWO KILOMETRES above the ground, looking DOWN diagonally at 45 degrees, exactly in the wide '
      'regional photograph supplied as the final image. Individual monuments and people become microscopic. '
      'Keep flying rapidly all the way to the LAST frame. No cloud passage, no orbit or space in this segment. '
      'Short shutter, photographic texture, fixed focal length, no zoom, no cuts, no dissolves, no hovering, no pause, '
      'no replacing ground with roofs, no morphing foreground objects.')
    lower_output=output/'lower'
    portrait_flight.run(lower_cfg,scene,lower_output,'flight',url,near_source=near_folder)
    lr=json.loads((lower_output/'flight/result.json').read_text())
    lr=dict(video=lr['descent'],sha256=lr['descent_sha256'],workflow=lr['workflow'],workflow_sha256=lr['workflow_sha256'])
    return assemble(ur,lr,near_folder,output)
