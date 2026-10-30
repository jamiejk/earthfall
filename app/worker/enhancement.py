"""Supervise a separate refinement subprocess using the existing worker queue."""
import hashlib,json,shlex,subprocess,time
from .runner import kill_build,output_dir

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def run_enhancement_job(job,rep,cfg,log=print):
    if not cfg.enhancement:
        rep.finish(job['id'],'failed',error='This worker does not support enhancement.');return 'failed'
    root=output_dir(cfg,{})/('enhance-'+job['id'].lower());root.mkdir(parents=True,exist_ok=True)
    proc=None;stages={'source':'running'}
    try:
        if rep.progress(job['id'],stages,'Loading the original film'):
            rep.finish(job['id'],'cancelled');return 'cancelled'
        source=root/'source.mp4';rep.source_video(job['id'],source)
        if sha(source)!=job['source_sha256']:raise ValueError('Original video changed during download')
        stages={'source':'done','refine':'pending'}
        command=[s.format(python=cfg.python,input=str(source),output=str(root),repo=str(cfg.repo)) for s in shlex.split(cfg.enhance_build_cmd)]
        with (root/'worker.log').open('a') as log_file:
            proc=subprocess.Popen(command,cwd=cfg.cwd(),env=cfg.env(),stdout=log_file,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                progress={}
                try:progress=json.loads((root/'progress.json').read_text())
                except (OSError,ValueError):pass
                stages=progress.get('stages',stages)
                if rep.progress(job['id'],stages,progress.get('message','Preparing resolution enhancement')):
                    kill_build(proc);rep.finish(job['id'],'cancelled',message='Enhancement cancelled. The original is unchanged.');return 'cancelled'
                if proc.poll() is not None:break
                time.sleep(cfg.tick_s)
        if proc.returncode:raise RuntimeError(f"Enhancement exited {proc.returncode}. "+(root/'worker.log').read_text()[-4000:])
        outcome=json.loads((root/'outcome.json').read_text());video=(root/outcome['video']).resolve()
        if not video.is_relative_to(root.resolve()) or not video.is_file():raise ValueError('Invalid enhanced result path')
        if sha(video)!=outcome['video_sha256'] or outcome['source_sha256']!=job['source_sha256'] or not outcome['clock_preserved']:raise ValueError('Enhancement evidence mismatch')
        if rep.progress(job['id'],stages,'Saving the enhanced film'):
            rep.finish(job['id'],'cancelled');return 'cancelled'
        rep.result(job['id'],video);rep.review(job['id'],outcome)
        rep.finish(job['id'],'done',message='Enhanced film ready. The original is unchanged.');return 'done'
    except Exception as exc:
        rep.finish(job['id'],'failed',error=f'Enhancement failed; original unchanged. {exc}');return 'failed'
    finally:
        if proc is not None and proc.poll() is None:kill_build(proc)
