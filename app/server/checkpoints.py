"""Authenticated, artifact-bound vision/operator checkpoints for the canonical worker."""
import hashlib,json,math,re,time
from pathlib import Path
from fastapi import Depends,HTTPException,Request,UploadFile,File
from fastapi.responses import FileResponse

WORKFLOW='photographic-grounded-v1'
# Checkpoints each canonical recipe passes, in order (canonical_build.RECIPES). The app workflow name stays
# photographic-grounded-v1; the build records which recipe ran in its outcome.
RECIPES={'photographic-grounded-v1':('flight','layout','texture','expand-1','expand-2','expand-3','project','orbit','final'),
         'photographic-generic-orbit-v1':('flight','final')}
IDS=RECIPES['photographic-grounded-v1']

def order(base,checkpoint_id):
    """The checkpoints that must be approved before this one: the grounded order once any grounded-only checkpoint
    has been published for the job (or for one only it has), otherwise the generic-orbit order."""
    generic=RECIPES['photographic-generic-orbit-v1']
    grounded=checkpoint_id not in generic or any((base/name).exists() for name in IDS if name not in generic)
    ids=IDS if grounded else generic
    return ids[:ids.index(checkpoint_id)]
ASSET=re.compile(r'^[a-z][a-z0-9-]{0,50}\.(png|jpg|mp4)$')

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(path.read_text())
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def validate_layout(layout):
    def uv(p):return isinstance(p,list) and len(p)==2 and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=1 for v in p)
    if not isinstance(layout,dict):raise ValueError('A reviewed layout is required.')
    frame=layout.get('cloud_exit_frame')
    if type(frame) is not int or frame<40:raise ValueError('Choose a clear cloud-exit frame at frame 40 or later.')
    if not uv(layout.get('target_uv')):raise ValueError('Set the destination within the plan.')
    colours=layout.get('colours',{})
    if not isinstance(colours,dict) or not 1<=len(colours)<=8 or 'background' not in colours or any(not re.fullmatch(r'#[0-9a-fA-F]{6}',str(v)) for v in colours.values()):raise ValueError('Use valid material colours.')
    features=layout.get('features')
    if not isinstance(features,list) or len(features)>32:raise ValueError('Invalid layout features.')
    for feature in features:
        if not isinstance(feature,dict) or feature.get('material') not in colours:raise ValueError('Unknown material.')
        points=feature.get('polygon',[])
        if not isinstance(points,list) or not 3<=len(points)<=128 or not all(uv(p) for p in points):raise ValueError('Each region needs 3–128 points inside the plan.')
    for field in ('layout_description','expansion_description'):
        if not isinstance(layout.get(field),str) or not 10<=len(layout[field].strip())<=3000:raise ValueError('Describe the layout and its wider surroundings.')
    return {**layout,'measured_geography':False,'provenance':layout.get('provenance','Derived and checked against this job’s fresh flight')}

def install(app,settings,jobs,user,worker,job_or_404):
    def folder(job_id):return settings.jobs_dir/job_id/'checkpoints'
    def job(job_id,active=False):
        value=job_or_404(job_id)
        if value['workflow']!=WORKFLOW:raise HTTPException(400,'Not a canonical workflow job.')
        if active and (value['status']!='running' or value['cancel_requested']):raise HTTPException(409,'This job is no longer accepting review changes.')
        return value
    def checked_id(checkpoint_id):
        if checkpoint_id not in IDS:raise HTTPException(400,'Unknown checkpoint.')
    def public(job_id,data):
        return {**data,'assets':[{**a,'url':f"/api/jobs/{job_id}/checkpoints/{data['id']}/assets/{a['name']}"} for a in data['assets']]}
    @app.post('/api/worker/jobs/{job_id}/checkpoints/{checkpoint_id}/assets/{name}',dependencies=[Depends(worker)])
    async def upload(job_id:str,checkpoint_id:str,name:str,file:UploadFile=File(...)):
        job(job_id,True);checked_id(checkpoint_id)
        if not ASSET.fullmatch(name):raise HTTPException(400,'Invalid review asset.')
        with jobs.lock:
            if (folder(job_id)/checkpoint_id/'request.json').exists():raise HTTPException(409,'Published review evidence is frozen.')
        dest=folder(job_id)/checkpoint_id/name;dest.parent.mkdir(parents=True,exist_ok=True)
        temporary=dest.with_suffix('.part');size=0
        try:
            with temporary.open('wb') as stream:
                while chunk:=await file.read(1<<20):
                    size+=len(chunk)
                    if size>512*1024*1024:raise HTTPException(413,'Review asset too large.')
                    stream.write(chunk)
            with jobs.lock:
                job(job_id,True)
                if (dest.parent/'request.json').exists():raise HTTPException(409,'Published review evidence is frozen.')
                temporary.replace(dest)
        finally:temporary.unlink(missing_ok=True)
        return {'sha256':sha(dest)}
    @app.post('/api/worker/jobs/{job_id}/checkpoint',dependencies=[Depends(worker)])
    async def publish(job_id:str,request:Request):
        data=await request.json();checkpoint_id=data.get('id');checked_id(checkpoint_id)
        if len(json.dumps(data))>60000 or not isinstance(data.get('assets'),list) or not 1<=len(data['assets'])<=8:raise HTTPException(400,'Invalid checkpoint.')
        with jobs.lock:
            job(job_id,True);base=folder(job_id)/checkpoint_id
            # Earlier stages must have been explicitly accepted, in order.
            for previous in order(folder(job_id),checkpoint_id):
                decision=folder(job_id)/previous/'decision.json'
                if not decision.exists() or read(decision).get('continue') is not True:raise HTTPException(409,'Previous checkpoint is not approved.')
            for asset in data['assets']:
                name=asset.get('name','')
                if not ASSET.fullmatch(name) or not (base/name).is_file() or sha(base/name)!=asset.get('sha256'):raise HTTPException(409,'Review evidence changed or is incomplete.')
            frozen={k:data[k] for k in ('id','assets','title','description')}
            frozen['evidence_sha256']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
            if (base/'request.json').exists() and read(base/'request.json')!=frozen:raise HTTPException(409,'Checkpoint already has different evidence.')
            write(base/'request.json',frozen);write(folder(job_id)/'current.json',{'id':checkpoint_id})
        return public(job_id,frozen)
    @app.get('/api/jobs/{job_id}/checkpoint',dependencies=[Depends(user)])
    def current(job_id:str):
        value=job(job_id);path=folder(job_id)/'current.json'
        if not path.exists():return {'checkpoint':None}
        base=folder(job_id)/read(path)['id'];data=read(base/'request.json')
        decision=read(base/'decision.json') if (base/'decision.json').exists() else None
        return {'checkpoint':public(job_id,data),'decision':decision,'can_decide':value['status']=='running' and not value['cancel_requested'] and decision is None}
    @app.post('/api/worker/jobs/{job_id}/checkpoints/{checkpoint_id}/decision',dependencies=[Depends(worker)])
    @app.post('/api/jobs/{job_id}/checkpoints/{checkpoint_id}/decision',dependencies=[Depends(user)])
    async def decide(job_id:str,checkpoint_id:str,request:Request):
        checked_id(checkpoint_id);data=await request.json()
        with jobs.lock:
            job(job_id,True);base=folder(job_id)/checkpoint_id
            if not (base/'request.json').exists():raise HTTPException(409,'Checkpoint not ready.')
            evidence=read(base/'request.json')
            if data.get('evidence_sha256')!=evidence['evidence_sha256']:raise HTTPException(409,'Review evidence changed. Reload before deciding.')
            if type(data.get('continue')) is not bool or not isinstance(data.get('reason'),str) or not 5<=len(data['reason'].strip())<=2000:raise HTTPException(422,'Record a review finding before approving or holding.')
            decision={k:data[k] for k in ('continue','reason','evidence_sha256')}
            if request.url.path.startswith('/api/worker/'):
                report=data.get('review_report','')
                if not isinstance(report,str) or len(report)>200000 or hashlib.sha256(report.encode()).hexdigest()!=data.get('record_sha256'):
                    raise HTTPException(422,'Automatic review evidence is missing or changed.')
                try:answer=json.loads(report)['answer']
                except (ValueError,KeyError,TypeError):raise HTTPException(422,'Invalid automatic review report.')
                if data['continue'] and answer.get('status')!='pass':raise HTTPException(409,'Vision did not pass this checkpoint.')
                decision.update(reviewer='qwen',review_report=json.loads(report),record_sha256=data['record_sha256'])
            else:decision['reviewer']='operator' 
            if checkpoint_id=='layout' and data['continue']:
                try:decision['layout']=validate_layout(data.get('layout'))
                except ValueError as exc:raise HTTPException(422,str(exc)) from exc
            if (base/'decision.json').exists():
                old=read(base/'decision.json')
                if any(old.get(k)!=v for k,v in decision.items()):raise HTTPException(409,'Decision already recorded.')
                return old
            decision['reviewed_at']=time.time();write(base/'decision.json',decision)
        return decision
    @app.get('/api/worker/jobs/{job_id}/checkpoints/{checkpoint_id}/decision',dependencies=[Depends(worker)])
    def worker_decision(job_id:str,checkpoint_id:str):
        job(job_id);checked_id(checkpoint_id);path=folder(job_id)/checkpoint_id/'decision.json'
        return read(path) if path.exists() else None
    @app.get('/api/jobs/{job_id}/checkpoints/{checkpoint_id}/assets/{name}',dependencies=[Depends(user)])
    def asset(job_id:str,checkpoint_id:str,name:str):
        job(job_id);checked_id(checkpoint_id);base=folder(job_id)/checkpoint_id
        if not ASSET.fullmatch(name) or not (base/'request.json').exists() or name not in [a['name'] for a in read(base/'request.json')['assets']]:raise HTTPException(404,'Unknown review asset.')
        return FileResponse(base/name)
