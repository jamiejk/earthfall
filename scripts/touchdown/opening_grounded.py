"""Opening-only replay with the reviewed flat-region-derived orbital photograph."""
import argparse
import importlib.util
from pathlib import Path
import shutil

DRIVER = Path(__file__).with_name('opening_rush_pinned.py')
spec = importlib.util.spec_from_file_location('pinned_opening_driver', DRIVER)
pinned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pinned)
base = pinned.base
ASSET_ROOT = base.REPO / 'outputs/touchdown/studio/studio-eilean-donan-flat-region-globe-v2-20261027'


def prepare(root):
    import cv2
    pinned.prepare(root)
    source = ASSET_ROOT / 'orbit.png'
    evidence = base.read(ASSET_ROOT / 'orbit.json')
    assert base.sha(source) == evidence['output_sha256']
    shutil.copy2(source, root / 'grounded-orbit-full.png')
    image = cv2.imread(str(source))
    h, w = image.shape[:2]
    crop_width = round(h*1280/736)
    left = (w-crop_width)//2
    image = cv2.resize(image[:, left:left+crop_width], (1280,736), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(root / 'orbit-start.png'), image)
    cw, ch = round(1280/1.8), round(736/1.8)
    x, y = (1280-cw)//2, (736-ch)//2
    # Keep the tracked central inlet under the central ray as Earth expands.
    closer = cv2.resize(image[y:y+ch, x:x+cw], (1280,736), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(root / 'closer-earth.png'), closer)
    plan = base.read(root / 'experiment.json')
    plan.update(comparison='New grounded orbital photograph and its closer pin; same opening-only LTX settings, seed, cloud prefix and later source frames',
        orbital_source=dict(path=str(source),sha256=base.sha(source),evidence=str(ASSET_ROOT/'orbit.json'),
            evidence_sha256=base.sha(ASSET_ROOT/'orbit.json'),source_size=[w,h],
            aspect_crop=[left,0,left+crop_width,h],conditioning_size=[1280,736],
            geography='Authored approximate regional layout; no measured registration'),
        changes=['Replace orbital image and derived closer keyframe',
                 'Centre closer crop on tracked destination instead of previous lower-frame zoom anchor'],
        retained=['Frozen photographic LTX 2.5 distilled and Prompt Relay','49-frame backward generation',
                  '9-frame reversed original cloud-prefix guide','Existing parent seed 2026997968',
                  'Original frames 40..195 at identical times','196-frame / 24fps / 8.167-second delivery'])
    plan['extra_photographic_guide'].update(provenance='1.8x crop of the new grounded orbital photograph',
        crop_rectangle=[x,y,cw,ch],focus_uv=[.5,.5])
    files=[root/'orbit-start.png',root/'closer-earth.png',root/'grounded-orbit-full.png',
           source,ASSET_ROOT/'orbit.json',Path(__file__).resolve()]
    plan['hashes'].update({str(p.resolve()):base.sha(p) for p in files})
    base.write(root/'experiment.json',plan)
    print('Prepared grounded opening; later source footage and timing retained',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','worker','assemble'])
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    if a.stage=='prepare':prepare(root)
    elif a.stage=='worker':pinned.worker(root)
    else:base.assemble(root)
