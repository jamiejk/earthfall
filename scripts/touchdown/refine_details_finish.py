"""Resume only the failed final chunk, then assemble the timed refinement trial."""
from pathlib import Path
import json,os,time,subprocess,shutil,shlex,hashlib,threading
REPO=Path(__file__).resolve().parents[2]
OLD=REPO/'outputs/touchdown/studio/studio-bamburgh-refine-details-chunked-v2-20261027'
ROOT=REPO/'outputs/touchdown/studio/studio-bamburgh-refine-details-final-v3-20261027'
def read(p):return json.loads(p.read_text())
def write(n,d):(ROOT/n).write_text(json.dumps(d,indent=2,default=str)+'\n')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 started=time.monotonic()
 if (ROOT/'submission-start.json').exists():raise ValueError('Existing submission; do not overwrite')
 for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
  k,v=line.split('=',1)
  if k.startswith('TOUCHDOWN_'):os.environ[k]=json.loads(v)
 from touchdown import studio_build as studio
 from touchdown.flight_runtime import capture_runtime
 from geoblock_core.run import _client,_submit
 experiment=read(OLD/'experiment.json');source=Path(experiment['source']);results=read(OLD/'chunk-results.json');assert len(results)==3
 for result in results:assert sha(Path(result['video']))==result['sha256']
 assert sha(source)==experiment['source_sha256'];input_video=OLD/'chunk-4/input.mp4'
 graph=read(OLD/'chunk-4/submitted-graph.json')
 graph['456']['inputs']['tile_frames']=57;graph['459']['inputs']['tile_frames']=57
 graph['490']['inputs']['filename_prefix']='earthfall/bamburgh_refine_final_chunk_v3'
 write('experiment.json',dict(recovery_of=str(OLD),preserved_chunks=[1,2,3],input=str(input_video),input_sha256=sha(input_video),change='Guide and sampler temporal window 49 -> 57; same 57-frame input now fits one tiled-encode window',source_sha256=experiment['source_sha256'],target=[3840,2208],app_promoted=False))
 shutil.copy2(OLD/'pinned-model-paths.json',ROOT/'pinned-model-paths.json')
 os.environ['TOUCHDOWN_COMFY_SERVER_CMD']+=' --cache-none --deterministic --extra-model-paths-config '+shlex.quote(str(ROOT/'pinned-model-paths.json'))
 capture_runtime(ROOT);write('status.json',dict(stage='runtime',status='starting'))
 stop=threading.Event();samples=[]
 def monitor():
  while not stop.is_set():
   try:
    gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10).stdout.strip()
    ram=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    samples.append(dict(elapsed_seconds=time.monotonic()-started,gpu=gpu,available_ram_kib=ram));write('resource-samples.json',samples)
   except Exception:pass
   stop.wait(15)
 thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:
  with studio.runtime('comfy',ROOT):
   with _client(studio.VIDEO_URL) as client:
    graph['401']['inputs']['file']=client.assets.from_file(str(input_video));write('submitted-graph.json',graph)
    job_started=time.monotonic();write('submission-start.json',dict(wall_time=time.time(),prep_and_runtime_seconds=job_started-started));write('status.json',dict(stage='refine',chunk=4,status='generating'))
    saved=_submit(client,graph,('490',),ROOT/'generation');raw=Path(saved['490']);seconds=time.monotonic()-job_started
    final_result=dict(chunk=4,source_first=144,source_last_exclusive=195,processed_frames=57,job_seconds=seconds,video=str(raw),sha256=sha(raw));write('chunk-4-result.json',final_result);results.append(final_result)
 finally:stop.set();thread.join(timeout=12)
 write('chunk-results.json',results);write('status.json',dict(stage='export',status='assembling'))
 cmd=['ffmpeg','-v','error','-y']
 for result in results:cmd+=['-i',result['video']]
 filters=[]
 for i,result in enumerate(results):
  length=result['source_last_exclusive']-result['source_first'];filters.append(f'[{i}:v]trim=start_frame={0 if i==0 else 1}:end_frame={length},setpts=PTS-STARTPTS[v{i}]')
 filters.append(''.join(f'[v{i}]' for i in range(4))+'concat=n=4:v=1:a=0[out]')
 final=ROOT/'bamburgh-refine-details-4k.mp4';cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(final)]
 subprocess.run(cmd,check=True)
 import cv2
 c=cv2.VideoCapture(str(final));info=[int(c.get(7)),c.get(5),int(c.get(3)),int(c.get(4))];c.release();assert info==[195,24,3840,2208],info;assert sha(source)==experiment['source_sha256']
 first=read(OLD/'submission-start.json');whole=REPO/'outputs/touchdown/studio/studio-bamburgh-refine-details-v1-20261027';initial=read(whole/'submission-start.json')
 timing=dict(successful_job_seconds_sum=sum(x['job_seconds'] for x in results),final_chunk_seconds=seconds,finish_attempt_wall_seconds=time.monotonic()-started,elapsed_since_chunked_recovery_start=time.time()-(first['wall_time']-first['prep_and_runtime_seconds']),elapsed_since_initial_whole_film_start=time.time()-(initial['wall_time']-initial['prep_and_runtime_seconds']),chunks=results,source_unchanged=True,output=str(final),output_sha256=sha(final),output_info=info,output_duration_seconds=8.125,boundary_frames=[49,97,145],processed_frames_successful=sum(x['processed_frames'] for x in results),quality='awaiting visual review')
 write('timing.json',timing);write('status.json',dict(stage='complete',status='awaiting_visual_review'));print(json.dumps(timing),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:write('failure.json',dict(type=type(e).__name__,message=str(e)));write('status.json',dict(stage='failed',status='held'));raise
