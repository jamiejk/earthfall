"""Memory-bounded measured recovery of the full-film Refine Details trial."""
from pathlib import Path
import json,os,shlex,shutil,subprocess,time,threading,urllib.request,hashlib
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'outputs/touchdown/studio/studio-bamburgh-refine-details-chunked-v2-20261027'
SOURCE=REPO/'outputs/touchdown/studio/studio-bamburgh-complete-grounded-v1-20261027/complete-flight.mp4'
CHUNKS=[(0,49,49),(48,97,49),(96,145,49),(144,195,57)]
def write(name,value):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,default=str)+'\n')
def sha(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
 started=time.monotonic()
 if (ROOT/'submission-start.json').exists():raise ValueError('Preserve previous attempt; do not rerun')
 for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
  k,v=line.split('=',1)
  if k.startswith('TOUCHDOWN_'):os.environ[k]=json.loads(v)
 from touchdown import studio_build as studio
 from touchdown.flight_runtime import capture_runtime
 from geoblock_core.upscale import upscale_file_graph,LTX_DISTILLED_SIGMAS
 from geoblock_core.run import _client,_submit
 source_sha=sha(SOURCE);paths=ROOT/'pinned-model-paths.json'
 shutil.copy2(REPO/'outputs/touchdown/studio/studio-bamburgh-complete-grounded-v1-20261027/flight/pinned-model-paths.json',paths)
 os.environ['TOUCHDOWN_COMFY_SERVER_CMD']+=' --cache-none --deterministic --extra-model-paths-config '+shlex.quote(str(paths))
 write('experiment.json',dict(source=str(SOURCE),source_sha256=source_sha,recovery_of='studio-bamburgh-refine-details-v1-20261027',reason='Kernel OOM killed whole-film decode/4K preprocessing before sampling',chunks=CHUNKS,target=[3840,2208],tile=[1024,576],window=49,overlap=.5,strength=1,seed=2081561474,sigmas=LTX_DISTILLED_SIGMAS,cache='none',assembly='Drop first shared frame of chunks 2-4; remove six padded last frames; direct joins, no blending',known_risk='Independent chunks can change appearance at three boundaries; visual review required',source_clock=[195,24,8.125],app_promoted=False))
 for i,(first,last,length) in enumerate(CHUNKS):
  part=ROOT/f'chunk-{i+1}';part.mkdir(exist_ok=True);pad=length-(last-first)
  vf=f'trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS'+(f',tpad=stop_mode=clone:stop={pad}' if pad else '')
  subprocess.run(['ffmpeg','-v','error','-y','-i',str(SOURCE),'-vf',vf,'-an','-c:v','libx264','-preset','fast','-crf','0','-pix_fmt','yuv420p',str(part/'input.mp4')],check=True)
 capture_runtime(ROOT);write('status.json',dict(stage='runtime',status='starting'))
 stop=threading.Event();samples=[]
 def monitor():
  while not stop.is_set():
   try:
    gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10).stdout.strip()
    available=next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    samples.append(dict(elapsed_seconds=round(time.monotonic()-started,1),gpu=gpu,available_ram_kib=int(available)));write('resource-samples.json',samples)
   except Exception:pass
   stop.wait(15)
 thread=threading.Thread(target=monitor,daemon=True);thread.start();results=[]
 try:
  with studio.runtime('comfy',ROOT):
   with urllib.request.urlopen(studio.COMFY_URL+'/object_info',timeout=60) as r:info=json.load(r)
   write('object-info.json',info)
   with _client(studio.VIDEO_URL) as client:
    write('submission-start.json',dict(wall_time=time.time(),prep_and_runtime_seconds=time.monotonic()-started))
    for i,(first,last,length) in enumerate(CHUNKS):
     part=ROOT/f'chunk-{i+1}';asset=client.assets.from_file(str(part/'input.mp4'))
     g=upscale_file_graph(asset,method='ltx-detail',target='4k',prefix=f'earthfall/bamburgh_refine_v2_chunk_{i+1}',seed=2081561474,fps=24)
     for key in ['400','449','450']:g.pop(key,None)
     g['443']['inputs']['device']='cpu'
     g['444']['inputs']['text']='Sharp photographic detail, crisp natural texture, fine surface detail, clean edges, natural film grain, high resolution footage. Preserve the source framing, motion, lighting and colours.'
     g['445']['inputs']['text']='blurry, soft, plastic, smeared detail, oversharpened halos, warped faces'
     g['451']['inputs'].update(width=3840,height=2208,length=length)
     g['452']['inputs'].update(image=['402',0],width=3840,height=2208)
     g['456']['inputs'].update(tile_frames=49,use_streaming=True,use_tiled_encode=True,tile_size=512,tile_overlap=64,latent_downscale_factor=1)
     g['459']['inputs'].update(tile_width=1024,tile_height=576,overlap_frac=.5,tile_frames=49,canvas_device='cpu')
     g['460']['inputs'].update(tile_size=512,overlap=64,temporal_size=64,temporal_overlap=16)
     for key,n in g.items():
      assert n['class_type'] in info,(key,n['class_type'])
      required=info[n['class_type']]['input'].get('required',{})
      assert set(required)<=set(n['inputs']),(key,set(required)-set(n['inputs']))
      for name,value in n['inputs'].items():
       if isinstance(value,list):assert value[0] in g,(key,name,value)
       elif name in required and isinstance(required[name][0],list):assert value in required[name][0],(key,name,value)
     write(f'chunk-{i+1}/submitted-graph.json',g);write('status.json',dict(stage='refine',chunk=i+1,status='generating'))
     job_start=time.monotonic();saved=_submit(client,g,('490',),part/'generation');raw=Path(saved['490'])
     result=dict(chunk=i+1,source_first=first,source_last_exclusive=last,processed_frames=length,job_seconds=time.monotonic()-job_start,video=str(raw),sha256=sha(raw));results.append(result);write(f'chunk-{i+1}/result.json',result);write('chunk-results.json',results)
     print('COMPLETE CHUNK',i+1,result['job_seconds'],flush=True)
 finally:stop.set();thread.join(timeout=12)
 write('status.json',dict(stage='export',status='assembling'))
 command=['ffmpeg','-v','error','-y']
 for result in results:command+=['-i',result['video']]
 filters=[]
 for i,(first,last,length) in enumerate(CHUNKS):filters.append(f'[{i}:v]trim=start_frame={0 if i==0 else 1}:end_frame={last-first},setpts=PTS-STARTPTS[v{i}]')
 filters.append(''.join(f'[v{i}]' for i in range(4))+'concat=n=4:v=1:a=0[out]')
 final=ROOT/'bamburgh-refine-details-4k.mp4';command+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(final)]
 subprocess.run(command,check=True)
 import cv2
 cap=cv2.VideoCapture(str(final));out=[int(cap.get(7)),cap.get(5),int(cap.get(3)),int(cap.get(4))];cap.release();assert out==[195,24,3840,2208],out;assert sha(SOURCE)==source_sha
 timing=dict(total_wall_seconds=time.monotonic()-started,job_seconds_sum=sum(x['job_seconds'] for x in results),chunks=results,source_unchanged=True,output=str(final),output_sha256=sha(final),output_info=out,output_duration_seconds=8.125,boundary_delivery_frames=[49,97,145],quality='awaiting visual review')
 write('timing.json',timing);write('status.json',dict(stage='complete',status='awaiting_visual_review'));print(json.dumps(timing),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:write('failure.json',dict(type=type(e).__name__,message=str(e)));write('status.json',dict(stage='failed',status='held'));raise
