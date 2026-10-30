"""Run bounded continuity alternatives with fixed seeds and no candidate selection."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route',choices=['retreat','portrait','clay'],required=True)
    p.add_argument('--endpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--terrain-stage',type=Path);p.add_argument('--imagery',type=Path)
    p.add_argument('--comfy-python',required=True);p.add_argument('--url',default='http://127.0.0.1:8191')
    p.add_argument('--blender',default='blender');p.add_argument('--seed',type=int,default=7)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.route=='clay' and (not a.terrain_stage or not a.imagery): p.error('clay needs --terrain-stage and --imagery')
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2]/'src')])}
    records=[]
    def job(name,**kw):
        params={'url':a.url,**kw};records.append({'job':name,'parameters':params})
        (a.output/'jobs.json').write_text(json.dumps(records,indent=2))
        completed=subprocess.run([a.comfy_python,'-m','touchdown.comfy_job',name,json.dumps(params)],env=env,capture_output=True,text=True)
        if completed.returncode: raise RuntimeError(completed.stderr[-4000:])
        return Path(json.loads(completed.stdout.strip().splitlines()[-1])['result'])
    if a.route=='retreat':
        prompt=('One continuous real camera shot. The woman remains standing completely still on the same '
          'pedestrian platform, facing the camera with her feet firmly planted. Starting at eye level, '
          'the camera moves smoothly backwards in a perfectly straight line away from her and steadily '
          'rises above the platform, revealing more of the surrounding overlook. The camera keeps looking '
          'at her as she becomes smaller. Real rigid stonework, railings and landscape keep their shapes '
          'and positions. No orbit, no sideways movement, no turn, no walking, no dissolve. '
          'Natural daylight, photographic detail.')
        clip=job('ltx',folder=str(a.output/'generation'),frames=73,pins=[[0,str(a.endpoint)]],prompt=prompt,seed=a.seed,prefix='geoblock/touchdown_retreat_trial')
        out=a.output/'approach.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(clip),'-vf','reverse','-an','-c:v','libx264','-crf','14',str(out)],check=True)
    else:
        if a.route=='portrait':
            subprocess.run([sys.executable,'-m','touchdown.portrait_scene','--endpoint',str(a.endpoint),'--output',str(a.output/'scene')],check=True,env=env)
            guide=a.output/'scene'/'guide.mp4';look=a.output/'scene'/'0000.png';end=a.endpoint;frames=73
            prompt=('One continuous gentle straight camera dolly toward a woman standing completely still '
              'on the same platform. Follow the guide camera movement exactly. Keep the same connected '
              'platform, stonework, railings, trees and canyon throughout. Her feet stay planted; '
              'she breathes gently and blinks. No walking, no turning, no scenery changes, '
              'natural daylight and photographic detail.')
        else:
            meta=json.loads((a.terrain_stage/'scene'/'terrain_guide.json').read_text());scene=a.output/'scene'
            command=[a.blender,'-b','-t','4','-noaudio','--python',str(Path(__file__).with_name('terrain_blender.py')),'--',
              '--terrain',str(a.terrain_stage/'dem'),'--imagery',str(a.imagery),'--output',str(scene),
              '--heading',str(meta['heading']),'--target-x',str(meta['target_x']),'--target-y',str(meta['target_y']),
              '--clay','--person-proxy']
            layout=a.terrain_stage/'platform_layout.json'
            if layout.exists(): command.extend(['--platform-layout',str(layout)])
            subprocess.run(command,check=True)
            end=job('layout_portrait',folder=str(a.output/'endpoint'),guide=str(scene/'guide_frames'/'0081.png'),identity=str(a.endpoint),seed=20261009)
            # Appearance must match the starting view; an eye-level endpoint
            # here can freeze that composition while guide geometry moves.
            look=job('layout_portrait',folder=str(a.output/'aerial-appearance'),
              guide=str(scene/'guide_frames'/'0001.png'),identity=str(end),seed=20261010,
              prompt=('Image 1 is an untextured 3D aerial layout of the Grand Canyon from several hundred '
                'metres above the rim. Make it a real aerial photograph from exactly image 1 camera '
                'viewpoint. Preserve all ridge silhouettes, ravines, paths, camera angle and scale '
                'from image 1. Image 2 supplies only the natural rock and concrete material appearance '
                'and daylight. Do not copy its eye-level viewpoint or its large person. The standing '
                'person at the destination is microscopic at this altitude; do not add a large figure. '
                'Natural rocky canyon surfaces and concrete pedestrian paths, photographic detail, '
                'no white clay surfaces.'))
            guide=scene/'guide.mp4';frames=81
            prompt=('Render the untextured 3D guide as one continuous real photographic camera descent. '
              'Follow its exact camera movement and preserve the platform edges, cliff silhouettes and '
              'stationary human position and scale. Replace the grey human proxy with the same woman '
              'in the endpoint photograph, mustard yellow jacket, green trousers, boots. Her feet stay '
              'planted at the proxy feet positions throughout. Rigid coherent concrete and natural '
              'rocky canyon, realistic depth and parallax, no walking, no bending structures, '
              'no invented railings.')
        clip=job('terrain_v2v',guide=str(guide),first=str(look),last=str(end),output=str(a.output/'ltx'),seed=a.seed,frames=frames,prompt=prompt)
        out=a.output/'approach.mp4';shutil.copy(clip,out)
    print(out,flush=True)

if __name__=='__main__':main()
