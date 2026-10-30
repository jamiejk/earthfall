"""Check retained parent phases on their fixed clock before allowing descent generation."""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import cv2

from touchdown.authored_scene import digest
from touchdown.flight_regression import read,write
from touchdown.flight_sequence import observed_chat,FrameObservation,TransitionObservation,classify,transition_verdict
from touchdown.photo_delivery import frame_at,video_info


def parent_preview(root):
    """A held parent stays reviewable at its original clock, before descent exists."""
    from touchdown.split_flight import _checked
    record=_checked(read(root/'motion/flight/parent/result.json'))
    cap=cv2.VideoCapture(record['video'])
    try:
        n,_,_,fps=video_info(cap)
        if n<193 or fps!=24:raise ValueError('Parent preview source clock changed')
    finally:cap.release()
    folder=root/'diagnostics';folder.mkdir(exist_ok=True)
    target=folder/'parent-forward.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',record['video'],'-vf',
        'trim=end_frame=193,reverse,setpts=PTS-STARTPTS','-an','-r','24',
        '-c:v','libx264','-crf','17','-pix_fmt','yuv420p',str(target)],check=True)
    write(folder/'preview.json',dict(source=record,video=str(target),sha256=digest(target),
        source_mapping=list(range(192,-1,-1)),fps=24,source_speed=1,scope='Parent diagnostic, not delivery'))
    from touchdown.flight_report import export
    export(target,folder/'board')
    return target


def opening_verdict(observations,transitions):
    failures=[]
    for phase in observations:
        matches=sum(classify(o)==phase['expected_view'] for o in phase['observations'])
        if matches < 2:
            failures.append(f"{phase['phase']} lacks matching content in its fixed retained interval [{phase['start_frame']}, {phase['end_frame']})")
    for transition in transitions:
        if not transition_verdict(transition['observation']):
            failures.append(f"{transition['boundary']} does not establish physical camera travel")
    return dict(status='fail' if failures else 'pass',failures=failures)


def review_opening(root, cfg=None):
    from touchdown import studio_build as studio
    cfg=cfg or read(root/'experiment.json')['identity']['config']
    o=cfg['video']['split'];clock=cfg['video']['phase_contract']
    record=read(root/'motion/flight/parent/result.json');video=Path(record['video'])
    if digest(video)!=record['sha256']:raise ValueError('Parent video changed before phase review')
    folder=root/'gates/parent-phases';folder.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video));n,w,h,fps=video_info(cap)
    if fps!=24 or n!=193+o['parent_handles']:raise ValueError('Parent source clock changed')
    phases=[];boundaries=[]
    try:
        for name,label,lo,hi in zip(['orbit','clouds','high_aerial'],['orbit','cloud_cover','high_regional'],clock['bounds'][:3],clock['bounds'][1:4],strict=True):
            hi=min(hi,o['opening_frames'])
            if hi<=lo:raise ValueError('Retained parent must establish all three opening phases')
            images=[]
            for fraction in [.25,.5,.75]:
                d=round(lo+(hi-lo-1)*fraction)
                path=folder/f'{name}-{d:03d}.jpg'
                image=frame_at(cap,192-d)
                cv2.imwrite(str(path),image)
                images.append(dict(delivery_frame=d,source_frame=192-d,path=str(path),sha256=digest(path)))
            phases.append(dict(phase=name,expected_view=label,start_frame=lo,end_frame=hi,frames=images))
        for bound in clock['bounds'][1:3]:
            images=[]
            for d in range(bound-8,bound+9,4):
                path=folder/f'boundary-{bound}-{d:03d}.jpg'
                cv2.imwrite(str(path),frame_at(cap,192-d))
                images.append(dict(delivery_frame=d,source_frame=192-d,path=str(path),sha256=digest(path)))
            boundaries.append(dict(boundary=bound,frames=images))
    finally:cap.release()
    with studio.runtime('vision',folder) as metadata:
        for phase in phases:
            phase['observations']=[]
            for f in phase['frames']:
                answer=observed_chat(studio.VISION_URL,[(Path(f['path']),'ACTUAL FRAME')],
                    'Describe only the image. Orbit requires a curved planetary limb and space; high regional '
                    'requires many kilometres of connected terrain with microscopic local landmarks, no dominant '
                    'nearby building or person, and no readable facades. A close castle with distant hills is '
                    'detailed_ground. Cloud cover requires near opaque cloud surrounding the camera. Distinguish '
                    'superimposed scenes. Do not infer the requested phase or altitude.',FrameObservation,
                    folder/f"observation-{f['delivery_frame']:03d}.json",metadata)
                phase['observations'].append(answer)
        for boundary in boundaries:
            images=[(Path(f['path']),f"TIME {f['delivery_frame']/24:.3f}s") for f in boundary['frames']]
            boundary['observation']=observed_chat(studio.VISION_URL,images,
                'These are consecutive chronological samples of one transition. Is there physical camera '
                'travel with clouds moving past the lens/changing perspective, or do separate scenes change '
                'opacity, dissolve, wipe or cut? Static features remaining behind a fading veil are not travel. '
                'Use unclear if the mechanism cannot be established.',TransitionObservation,
                folder/f"transition-{boundary['boundary']}.json",metadata)
    verdict=opening_verdict(phases,boundaries)
    if digest(video)!=record['sha256']:raise ValueError('Parent changed during phase review')
    result=dict(verdict,video=str(video),video_sha256=record['sha256'],contract=clock,model=metadata,
                evaluator_sha256=digest(Path(__file__)),phases=phases,transitions=boundaries,
                scope='Retained parent opening only; full assembled video still requires review',advisory=True)
    write(folder/'report.json',result)
    return result


def review_final(root, cfg):
    """Shared app/CLI gate; validate the frame map before any visual judgement."""
    from touchdown.flight_contract import validate_mapping
    from touchdown.flight_sequence import review
    pacing=read(root/'motion/pacing.json')
    mapping=[(s,int(i)) for s,i in (r.split(':') for r in pacing['source_frame_mapping'])]
    validate_mapping(mapping,cfg['video']['split']['prefix_frames'])
    if pacing['dropped_hold_frames'] or pacing['fps']!=24:
        raise ValueError('Delivery was retimed')
    policy=read(Path(__file__).with_name('presets')/'flight-review-v1.json')
    policy['delivery_contract']=cfg['video']['phase_contract']
    return review(root/'motion/earth_zoom.mp4',root/'gates/flight-sequence',policy)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    result=review_opening(p.parse_args().root.resolve())
    print(result['status']);print('\n'.join(result['failures']))
    return 0 if result['status']=='pass' else 2


if __name__=='__main__':raise SystemExit(main())
