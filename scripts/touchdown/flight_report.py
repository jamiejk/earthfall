"""Export the same full-flight storyboard for every location and recipe."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from touchdown.authored_scene import digest
from touchdown.flight_regression import read, write


def export(video, output, review_path=None, interval=.25):
    if interval <= 0:
        raise ValueError('Storyboard interval must be positive')
    video, output = video.resolve(), output.resolve()
    source_hash = digest(video)
    report = read(review_path) if review_path else None
    if report and report['identity']['video_sha256'] != source_hash:
        raise ValueError('Review belongs to different video pixels')
    cap = cv2.VideoCapture(str(video))
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0 or count < 1:
        cap.release()
        raise ValueError('Cannot decode storyboard source')
    selected = {r['frame']:r for r in report['frames']} if report else {
        i:dict(frame=i,time=i/fps) for i in sorted(set(range(0,count,max(1,round(interval*fps)))) | {count-1})}
    labels = dict(zip([r['frame'] for r in report['frames']],report['verdict']['classified_frames'])) if report else {}
    tiles, index = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index in selected:
                frame = cv2.resize(frame,(384,221),interpolation=cv2.INTER_AREA)
                tile = Image.new('RGB',(384,249),'#151515')
                tile.paste(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),(0,28))
                ImageDraw.Draw(tile).text((7,7),f"{index/fps:.3f}s | frame {index} | {labels.get(index,'unreviewed')}",fill='white')
                tiles.append(tile)
            index += 1
    finally:
        cap.release()
    if index != count:
        raise ValueError('Short storyboard decode')
    board = Image.new('RGB',(4*384,56+249*((len(tiles)+3)//4)),'#151515')
    draw = ImageDraw.Draw(board)
    status = report['status'] if report else 'unreviewed'
    draw.text((8,7),f'Full flight | {status} | {count/fps:.3f}s | {fps:g} fps',fill='white')
    draw.text((8,29),'Required: orbit -> cloud cover -> broad 30,000-ft regional view -> detailed ground',fill='white')
    for i,tile in enumerate(tiles):
        board.paste(tile,((i%4)*384,56+(i//4)*249))
    output.mkdir(parents=True,exist_ok=True)
    board.save(output/'storyboard.jpg',quality=92)
    write(output/'summary.json',dict(video=str(video),sha256=source_hash,frames=count,fps=fps,
          duration=count/fps,review=str(review_path.resolve()) if review_path else None,status=status,
          failures=report['verdict']['failures'] if report else [],sampled_frames=sorted(selected)))
    return output/'storyboard.jpg'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--review',type=Path)
    args = parser.parse_args()
    print(export(args.video,args.output,args.review))


if __name__=='__main__':
    main()
