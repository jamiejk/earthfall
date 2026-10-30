"""One measured, non-destructive Refine Details trial on accepted Bamburgh."""
from pathlib import Path
import json,os,shlex,shutil,subprocess,time,hashlib,threading,urllib.request
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'outputs/touchdown/studio/studio-bamburgh-refine-details-v1-20261027'
SOURCE=REPO/'outputs/touchdown/studio/studio-bamburgh-complete-grounded-v1-20261027/complete-flight.mp4'
def write(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
def sha(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
 started=time.monotonic();ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'submission-start.json').exists():raise ValueError('Existing submission: preserve this attempt; use a new trial for retries')
 for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
  key,value=line.split('=',1)
  if key.startswith('TOUCHDOWN_'):os.environ[key]=json.loads(value)
 from touchdown import studio_build as studio
 from touchdown.flight_runtime import capture_runtime
 from geoblock_core.upscale import upscale_file_graph,LTX_DISTILLED_SIGMAS
 from geoblock_core.run import _client,_submit
 from geoblock_core.workflow import LTX25_MODELS
 import cv2
 cap=cv2.VideoCapture(str(SOURCE));count=int(cap.get(7));fps=cap.get(5);w=int(cap.get(3));h=int(cap.get(4));cap.release()
 assert (count,fps,w,h)==(195,24,2560,1472)
 source_sha=sha(SOURCE);padded=ROOT/'padded-source-201.mp4'
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(SOURCE),'-vf','tpad=stop_mode=clone:stop=6','-an','-c:v','libx264','-crf','0','-preset','fast','-pix_fmt','yuv420p',str(padded)],check=True)
 model_root=Path('/run/media/jamie/Image Creation/Live-Models')
 weights=[model_root/'diffusion_models'/LTX25_MODELS['transformer'],model_root/'text_encoders'/LTX25_MODELS['text_encoder'],model_root/'vae'/LTX25_MODELS['video_vae'],model_root/'loras/ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors']
 for p in weights:assert p.is_file(),p
 model_stats=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in weights]
 shutil.copy2(REPO/'outputs/touchdown/studio/studio-bamburgh-complete-grounded-v1-20261027/flight/pinned-model-paths.json',ROOT/'pinned-model-paths.json')
 os.environ['TOUCHDOWN_COMFY_SERVER_CMD']+=' --deterministic --extra-model-paths-config '+shlex.quote(str(ROOT/'pinned-model-paths.json'))
 write('experiment.json',dict(source=str(SOURCE),source_sha256=source_sha,source_frames=count,fps=fps,source_size=[w,h],process_frames=201,padding='Six repeated last frames; trim after processing, no source retiming',target=[3840,2208],tile=[1024,576],temporal_window=97,overlap=.5,lora_strength=1,seed=2081561474,sigmas=LTX_DISTILLED_SIGMAS,weights=model_stats,fresh_weight_sha256=False,app_promoted=False))
 capture_runtime(ROOT);write('status.json',dict(stage='runtime',status='starting'))
 with studio.runtime('comfy',ROOT):
  ready=time.monotonic()
  with urllib.request.urlopen(studio.COMFY_URL+'/object_info',timeout=60) as r:info=json.load(r)
  write('object-info.json',info)
  with _client(studio.VIDEO_URL) as client:
   asset=client.assets.from_file(str(padded))
   g=upscale_file_graph(asset,method='ltx-detail',target='4k',prefix='earthfall/bamburgh_refine_details_v1',seed=2081561474,fps=24)
   # Pin the trained 1024x576 tile and exact-aspect 4K canvas rather than the old size calculator's HD tile.
   for key in ['400','449','450']:g.pop(key,None)
   g['443']['inputs']['device']='cpu'
   g['444']['inputs']['text']='Sharp photographic detail, crisp natural texture, fine surface detail, clean edges, natural film grain, high resolution footage. Preserve the source framing, motion, lighting and colours.'
   g['445']['inputs']['text']='blurry, soft, plastic, smeared detail, oversharpened halos, warped faces'
   g['451']['inputs'].update(width=3840,height=2208,length=201)
   g['452']['inputs'].update(image=['402',0],width=3840,height=2208)
   g['456']['inputs'].update(tile_frames=97,use_streaming=True,use_tiled_encode=True,tile_size=512,tile_overlap=64,latent_downscale_factor=1)
   g['459']['inputs'].update(tile_width=1024,tile_height=576,overlap_frac=.5,tile_frames=97,canvas_device='cpu')
   g['460']['inputs'].update(tile_size=512,overlap=64,temporal_size=64,temporal_overlap=16)
   # Validate actual installed classes, required inputs, links and selectable model values before inference.
   for key,n in g.items():
    assert n['class_type'] in info,(key,n['class_type'])
    required=info[n['class_type']]['input'].get('required',{})
    assert set(required)<=set(n['inputs']),(key,set(required)-set(n['inputs']))
    for name,value in n['inputs'].items():
     if isinstance(value,list):assert value[0] in g,(key,name,value)
     elif name in required and isinstance(required[name][0],list):assert value in required[name][0],(key,name,value)
   write('submitted-graph.json',g)
   inference_start=time.monotonic();write('submission-start.json',dict(wall_time=time.time(),prep_and_runtime_seconds=inference_start-started,runtime_ready_seconds=ready-started));write('status.json',dict(stage='refine',status='generating'))
   stop=threading.Event();samples=[]
   def monitor():
    while not stop.is_set():
     try:
      result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
      samples.append(dict(elapsed_seconds=round(time.monotonic()-inference_start,1),gpu=result.stdout.strip()));write('resource-samples.json',samples)
     except Exception:pass
     stop.wait(15)
   thread=threading.Thread(target=monitor,daemon=True);thread.start()
   try:saved=_submit(client,g,('490',),ROOT/'generation')
   finally:stop.set();thread.join(timeout=12)
   processing=time.monotonic()-inference_start
   raw=Path(saved['490']);write('generation-result.json',dict(video=str(raw),sha256=sha(raw),processing_seconds=processing))
 write('status.json',dict(stage='export',status='trimming_padding'))
 final=ROOT/'bamburgh-refine-details-4k.mp4'
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(raw),'-vf','trim=end_frame=195,setpts=PTS-STARTPTS','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(final)],check=True)
 cap=cv2.VideoCapture(str(final));output=[int(cap.get(7)),cap.get(5),int(cap.get(3)),int(cap.get(4))];cap.release();assert output==[195,24,3840,2208],output
 assert sha(SOURCE)==source_sha
 write('timing.json',dict(processing_seconds=processing,total_wall_seconds=time.monotonic()-started,frames_delivered=195,frames_processed=201,source_duration_seconds=195/24,output=str(final),output_sha256=sha(final),source_unchanged=True,output_info=output))
 write('status.json',dict(stage='complete',status='awaiting_visual_review'));print(json.dumps(json.loads((ROOT/'timing.json').read_text())),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if ROOT.exists():write('failure.json',dict(type=type(e).__name__,message=str(e)));write('status.json',dict(stage='failed',status='held'))
  raise
