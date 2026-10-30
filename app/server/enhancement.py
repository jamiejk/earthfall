"""Source-bound optional refinement jobs; originals are never overwritten."""
import hashlib,json,time
from pathlib import Path
from fastapi import Depends,HTTPException
from fastapi.responses import FileResponse

WORKFLOW='refine-details-v1'
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def source_path(settings,jobs,job):
    parent=jobs.get(job['parent_id'])
    if not parent or parent['status']!='done' or parent['workflow']==WORKFLOW:
        raise HTTPException(409,'The original film is no longer available.')
    path=settings.jobs_dir/parent['id']/parent['result']
    if not parent['result'] or not path.is_file():raise HTTPException(409,'The original film is missing.')
    if digest(path)!=job['source_sha256']:raise HTTPException(409,'The original film changed after enhancement was queued.')
    return path

def completion_error(settings,jobs,job):
    try:
        source_path(settings,jobs,job)
        folder=settings.jobs_dir/job['id'];report=json.loads((folder/'review.json').read_text())
        if report.get('source_sha256')!=job['source_sha256'] or report.get('video_sha256')!=digest(folder/job['result']):return 'Enhancement evidence does not match its source and output.'
        if not isinstance(report.get('source_frames'), int) or report['source_frames'] < 1 or not report.get('source_fps') or not report.get('clock_preserved') or report.get('source_frames')!=report.get('output_frames') or report.get('source_fps')!=report.get('output_fps'):return 'Enhancement changed the source frame clock.'
    except (OSError,ValueError,HTTPException):return 'Enhancement verification is missing or the original changed.'
    return None

def install(app,settings,jobs,user,worker,job_or_404,public_job):
    app.state.enhance_worker_seen=0.0
    def available():return bool(settings.worker_token and time.time()-app.state.enhance_worker_seen<45)
    @app.get('/api/jobs/{job_id}/enhancement',dependencies=[Depends(user)])
    def status(job_id:str):
        parent=job_or_404(job_id)
        if parent['workflow']==WORKFLOW:raise HTTPException(400,'Open the original film to manage its enhancement.')
        child=jobs.latest_enhancement(job_id)
        return {'available':available(),'eligible':parent['status']=='done' and bool(parent['result']),'job':public_job(jobs,child) if child else None,'estimate':'About 45–55 minutes for an 8-second film on the tested RTX 4090. The original remains available.'}
    @app.post('/api/jobs/{job_id}/enhance',dependencies=[Depends(user)])
    def enqueue(job_id:str):
        parent=job_or_404(job_id)
        if parent['status']!='done' or not parent['result'] or parent['workflow']==WORKFLOW:raise HTTPException(409,'Enhancement requires a completed original film.')
        path=settings.jobs_dir/parent['id']/parent['result']
        if not path.is_file():raise HTTPException(409,'The original film is missing.')
        sha=digest(path)
        # Serialize repeat clicks across API processes as well as threads.
        with jobs.lock:
            jobs.conn.execute('BEGIN IMMEDIATE')
            try:
                previous=jobs.latest_enhancement(job_id)
                if previous and previous['source_sha256']==sha and previous['status'] in ('queued','running','done'):
                    result=previous
                else:
                    if not available():raise HTTPException(503,'The enhancement worker is offline. Your original film is unchanged.')
                    if jobs.active_count()>=settings.max_queue:raise HTTPException(429,'The queue is full; try again later.')
                    result=jobs.create(parent['preset'],'','',{'title':parent['place'].get('title',parent['preset'])},workflow=WORKFLOW,parent_id=job_id,source_sha256=sha)
                jobs.conn.execute('COMMIT')
            except BaseException:
                jobs.conn.execute('ROLLBACK');raise
        return public_job(jobs,result)
    @app.get('/api/worker/jobs/{job_id}/source-video',dependencies=[Depends(worker)])
    def download_source(job_id:str):
        child=job_or_404(job_id)
        if child['workflow']!=WORKFLOW:raise HTTPException(400,'Not an enhancement job.')
        return FileResponse(source_path(settings,jobs,child),media_type='video/mp4')
