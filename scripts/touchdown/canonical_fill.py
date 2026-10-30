"""One unchanged native Fill pass per owned runtime, releasing GPU for Qwen review.

Graph, prompt, seeds, masks and model validation are the canonical expand recipe.
Only runtime lifetime differs: no GPU is held while the vision reviewer runs.
"""
import os,shlex,time,uuid
import urllib.request
from touchdown.scene_grounded_workflow import (REPO,environment,verify,case,model_checks,read,write,sha)

def expand_one(root,index):
    environment();verify(root)
    from touchdown import fill_outpaint as fill,studio_build as studio,flux3
    from touchdown.flight_runtime import capture_runtime
    selected,cfg=case(root);folder=root/'terrain';model_checks(folder)
    layout=read(root/'scene-layout.json');target=layout['target_uv']
    paths=folder/'fill-paths.json'
    write(paths,dict(scene_fill=dict(is_default=True,**{role:str((REPO/'models/FLUX-Fill'/role).resolve())
        for role in ('diffusion_models','text_encoders','vae')})))
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']+=' --deterministic --extra-model-paths-config '+shlex.quote(str(paths))
    capture_runtime(folder)
    (folder/'decisions').mkdir(exist_ok=True)
    with studio.runtime('comfy',folder):
        for index in [index]:
            verify(root)
            source=folder/('texture.png' if index==1 else f'expand-{index-1}/raw-outpaint.png')
            part=folder/f'expand-{index}'
            graph,photo=fill.prepare(source,part)
            graph['23']['inputs']['text']=(
                'Extend this vertical NADIR satellite orthophotograph beyond every edge. Keep the '
                'same completely straight-down view, uniform ground scale, illumination and texture. '
                f'Continue the existing connected geographic surface of {cfg["words"]["region"]}. '
                +layout['expansion_description']+' '
                'Preserve existing land/water boundaries through the central source area. New '
                'surroundings form a much wider region. Fine small-scale overhead topographic texture. '
                'No sky, horizon, oblique side faces, close foreground, giant landmark, repeated '
                'castle, labels, frame or inset photograph.')
            fill.validate(graph)
            ui=read(part/'workflow.json')
            next(n for n in ui['nodes'] if n['id']==23)['widgets_values'][0]=graph['23']['inputs']['text']
            write(part/'workflow.json',ui)
            graph['17']['inputs']['image']=flux3._upload(studio.COMFY_URL,photo)
            write(part/'submitted-graph.json',graph)
            response=flux3._call(studio.COMFY_URL,'/prompt',dict(prompt=graph,client_id=uuid.uuid4().hex,
                extra_data={'extra_pnginfo':{'workflow':ui}}))
            write(part/'submission.json',response);pid=response['prompt_id'];started=time.monotonic()
            write(root/'status.json',dict(stage='expand',index=index,status='generating',prompt_id=pid))
            print('START expansion',index,pid,flush=True)
            while True:
                history=flux3._call(studio.COMFY_URL,f'/history/{pid}').get(pid,{})
                if history.get('status',{}).get('status_str')=='error':
                    write(part/'history.json',history);raise RuntimeError('Native Fill failed')
                if history.get('status',{}).get('completed'):break
                if time.monotonic()-started>1800:raise TimeoutError('Native Fill timed out')
                time.sleep(2)
            write(part/'history.json',history)
            for node,name in [('9','raw-outpaint'),('47','padding-mask'),('48','padded-input')]:
                item=history['outputs'][node]['images'][0]
                with urllib.request.urlopen(studio.COMFY_URL+'/view?'+urllib.parse.urlencode(item),timeout=60) as r:
                    (part/f'{name}.png').write_bytes(r.read())
            write(part/'geometry.json',dict(expansions=index,original_target_uv=target,
                nominal_target_uv=[.5+(v-.5)/(2**index) for v in target],
                original_source_width_fraction=1/(2**index),raw_sha256=sha(part/'raw-outpaint.png'),
                source_sha256=sha(source),exact_pixel_preservation=False,elapsed_seconds=round(time.monotonic()-started,1)))
    verify(root)
    write(root/'status.json',dict(stage='expand',status='complete'))

