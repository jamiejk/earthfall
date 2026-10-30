"""Complete the archived runtime's non-generative asset; resume assembly only."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
REPO=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('stage',choices=['assets','assemble']);p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve();code=root/'frozen-code'
source=REPO/'assets/models/face_detection_yunet_2023mar.onnx';target=code/'assets/models'/source.name
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if a.stage=='assets':
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists() and digest(target)!=digest(source):raise ValueError('Existing detector differs')
 shutil.copy2(source,target)
 (root/'runtime-assets.json').write_text(json.dumps(dict(asset=str(target),source=str(source),sha256=digest(target),purpose='Historical ending crop selection only; no video/image inference change',script_sha256=digest(Path(__file__))),indent=2)+'\n')
else:
 sys.path[:0]=[str(code/'scripts'),str(code/'src')]
 from touchdown.flight_regression import read,write,verify_snapshot,summary
 from touchdown import split_flight as flight
 assert Path(flight.__file__).is_relative_to(code)
 assets=read(root/'runtime-assets.json');assert digest(target)==assets['sha256']
 manifest=read(root/'run/experiment.json');verify_snapshot(manifest['source_hashes'])
 output=root/'run/motion';cfg=manifest['config'];near=manifest['native_portrait']
 upper=read(output/'flight/parent/result.json');lower=read(output/'flight/descent/result.json')
 flight.assemble(output,near,upper,lower,flight.options(cfg))
 verify_snapshot(manifest['source_hashes']);verify_snapshot(read(root/'recipe.json')['inputs_and_code'])
 write(root/'run/status.json',dict(stage='assembly',status='rendered',publication='held for visual review',resume='No generation resubmitted'))
 write(root/'status.json',dict(status='rendered_awaiting_visual_review',app_promoted=False))
 write(root/'assembly-resume.json',dict(missing_asset_restored=True,new_inference_jobs=0,result=summary(root/'run')))
 print('Assembly completed from existing generations',flush=True)
