"""Install the example films from app/examples into the app's protected gallery.

No model calls or recursive discovery. Source films are unchanged. touchdown-app runs
this on start; run it by hand for another app data directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import cv2

REPO = Path(__file__).resolve().parents[1]
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def install(gallery):
    cases = json.loads((REPO/'app/server/reviewed_examples.json').read_text())
    gallery.mkdir(parents=True, exist_ok=True)
    public = []
    for case in cases:
        source = REPO/case['source']; name = 'reviewed-'+case['id']; target = gallery/(name+'.mp4')
        expected = digest(source)
        if target.exists() and digest(target) != expected:
            raise ValueError('Different installed example: '+str(target))
        if not target.exists(): shutil.copy2(source, target)
        cap = cv2.VideoCapture(str(source)); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
        # The poster comes from about 60% through the film: the landmark in view and the person small. The last
        # frame is the same close portrait pose in every film, which made the four cards look like one result.
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(count*0.6)); ok, image = cap.read(); cap.release()
        if not ok or not fps: raise ValueError('Cannot read example: '+str(source))
        h,w=image.shape[:2]; image=cv2.resize(image,(960,round(h*960/w)),interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(gallery/(name+'.jpg')),image): raise ValueError('Cannot save poster')
        (gallery/(name+'.json')).write_text(json.dumps({'title':case['title'],'source_sha256':expected,'reviewed':True},indent=2)+'\n')
        public.append({k:v for k,v in case.items() if k != 'source'} | {'example_url':'/api/gallery/'+name+'.mp4','poster':'/api/gallery/'+name+'.jpg','duration':f'{count/fps:.2f} sec'})
    (REPO/'app/web/reviewed-examples.json').write_text(json.dumps(public,indent=2)+'\n')
    print(f'Installed {len(public)} example films in {gallery}; sources unchanged.')
    return public
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--gallery',type=Path,required=True)
    install(p.parse_args().gallery.resolve())
