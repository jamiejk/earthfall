"""Preview-only opening acceleration using existing frames; no generation changes."""
from pathlib import Path
import argparse,json,hashlib,subprocess


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def opening_map(count):
    if count<13:raise ValueError('Not enough frames for this opening preview')
    # First 12 source frames (0.5s) occupy 8 output frames (0.333s).
    # Every subsequent source frame is retained once, at native 24fps.
    return [round(i*11/7) for i in range(8)]+list(range(12,count))


def main():
    import cv2
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();source=a.source.resolve();root=a.root.resolve()
    if root.exists() and any(root.iterdir()):raise ValueError('Fresh preview root required')
    root.mkdir(parents=True,exist_ok=True)
    original_sha=sha(source);cap=cv2.VideoCapture(str(source));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    if fps!=24:raise ValueError('This preview preserves the existing 24fps clock')
    mapping=opening_map(count);dropped=sorted(set(range(count))-set(mapping))
    assert mapping[8:]==list(range(12,count)) and len(mapping)==count-4
    select='not('+ '+'.join(f'eq(n\\,{i})' for i in dropped)+')'
    output=root/'opening-nudge.mp4'
    manifest=dict(source=str(source),source_sha256=original_sha,source_frames=count,fps=fps,
        output_frames=len(mapping),duration_seconds=len(mapping)/fps,source_duration_seconds=count/fps,
        source_frame_mapping=mapping,dropped_source_frames=dropped,
        changed='Only first 12 source frames: 0.5s -> 0.333s (1.5x average playback)',
        retained='All source frames 12 onward, in order, once each, at native speed',
        interpolation=False,synthetic_zoom=False,crossfades=False,new_inference_jobs=0,
        recipe_changed=False,app_promoted=False,
        export='H.264 CRF16 yuv420p; normal re-encoding may slightly change pixel values',
        script_sha256=sha(Path(__file__)))
    (root/'preview-plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-vf',f'select={select},setpts=N/(24*TB)',
        '-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-fps_mode','passthrough',
        '-movflags','+faststart',str(output)],check=True)
    cap=cv2.VideoCapture(str(output));actual=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));actual_fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    assert (actual,actual_fps)==(len(mapping),24) and sha(source)==original_sha
    (root/'export-audit.json').write_text(json.dumps(dict(output=str(output),sha256=sha(output),frames=actual,fps=actual_fps,
        source_unchanged=True,all_later_source_frames_retained=True,first_frame_retained=True,last_frame_retained=True),indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
