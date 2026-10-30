"""Photographic references and an unpinned living end over the connected guide."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from touchdown.authored_scene import read_scene,generate_asset,digest


def paint(folder,url,manifest,only=None):
    cfg=read_scene(manifest);base=Path('outputs/touchdown/authored-canyonlands').resolve()
    for name,index in [('wide_photo',0),('arrival_photo',96)]:
        if only and name!=only:continue
        source=folder/'frames'/f'{index:04d}.png'
        refs=[source,Path(cfg['subject']),base/'master_look_reference.png']+[Path(r['local_path']) for r in cfg['references']]
        prompt=('Make image 1 a convincing real travel photograph of Canyonlands. '
            'Preserve its exact camera angle, scale, continuous sandstone landform, cliff edges, foreground '
            'and horizon positions. The terrain is one connected eroded landscape, never separate pillars '
            'pasted on a flat photo. Images 4, 5 and 6 are real location references: use their irregular '
            'strata, fissures, broken ledges, talus, sparse grey-green scrub and natural atmospheric detail '
            'to replace the smooth CGI appearance. Image 3 sets the restrained cream/copper palette, '
            'cool storm light and camera-left sunlight. Natural textures at the correct scale, no tiling, '
            'no giant grains or repeating patterns. One photograph, not a collage. ')
        if name=='arrival_photo':
            prompt+=('Image 2 identifies the person: retain her exact face, curly hair and red raincoat over '
                'a grey hoodie and black trousers. She occupies the same place and size as in image 1. '
                'Relight and integrate her into the whole photograph, with natural skin, soft shared '
                'illumination, grounded feet and a plausible contact shadow. Natural relaxed expression, '
                'a slight breeze in the coat and curls. Remove cutout edges and compositing artifacts. ')
        else:
            from PIL import Image,ImageOps
            layout=folder/'wide_camera_layout.png'
            ImageOps.grayscale(Image.open(source)).convert('RGB').save(layout)
            refs=[layout,base/'master_look_reference.png']
            prompt=('A real aerial photograph from 600 metres above a continuous Canyonlands sandstone '
                'promontory, camera looking down at 45 degrees. Image 1 is the mandatory camera and '
                'landform layout: preserve its steep downward view and the connected spur and ravines. '
                'Image 2 contains material and atmosphere samples only: use its cream/copper sandstone '
                'colours, warm light from the left and cool shadows. This is an entirely new aerial '
                'view, not an eye-level landscape. The image contains only terrain, no sky or horizon. '
                'Real irregular layered cliffs, talus, natural erosion, fissures, scrub and open '
                'sandstone benches at the correct aerial scale. One coherent photographic landscape, '
                'no collage, no text, no separate columns, no repeating texture, no large person.')
        spec=dict(name=name,width=2688,height=1536,seed=2026102208+index,models=cfg['models'],
                  context_parent=str(folder/'master.png'),references=[str(p) for p in refs],prompt=prompt)
        generate_asset(url,spec,folder)


def video(folder,url,upscale=False):
    from geoblock_core.workflow import ShotRequest,_add_ltx25,LTX_VIDEO
    from geoblock_core.run import _client,_submit
    prompt=('One continuous fast cinematic drone descent, following the camera guide from high over '
        'a continuous naturally eroded Canyonlands sandstone promontory down to the same woman in a red '
        'raincoat, grey hoodie and black trousers. The scene is one real connected canyon landscape, '
        'cream sandstone, copper cliff bands, irregular weathering, sparse scrub, cool storm sky and '
        'warm sunlight from the left. Real photographic travel footage, natural parallax and depth. '
        'As the camera arrives and settles, a breeze visibly moves her red coat hem, sleeve folds and '
        'curly hair. She breathes, blinks once and makes a small natural head turn toward the camera. '
        'She is a living person physically standing in the scene, both feet grounded, no frozen cutout. '
        'Preserve her face and clothing. The scenery stays stable, no scenery changes, no extra people, '
        'no transitions or dissolves. Slow only near the end of the guide.')
    output=folder/'living';output.mkdir(exist_ok=True)
    with _client(url) as client:
        graph={'50':{'class_type':'LoadVideo','inputs':{'file':client.assets.from_file(str(folder/'approach.mp4'))}},
               '60':{'class_type':'GetVideoComponents','inputs':{'video':['50',0]}}}
        refs=[]
        for k,name in enumerate(['wide_photo','arrival_photo']):
            load=str(40+k*2);pre=str(41+k*2)
            graph[load]={'class_type':'LoadImage','inputs':{'image':client.assets.from_file(str(folder/f'{name}.png'))}}
            graph[pre]={'class_type':'LTXVPreprocess','inputs':{'image':[load,0],'img_compression':18}}
            refs.append([pre,0])
        req=ShotRequest(name='living-canyon',output_prefix='touchdown/connected_living_canyon',seed=2026102217,
                        ltx_upscale=upscale,pin_last_keyframe=False)
        # The close photographic reference supplies skin, lighting and geology;
        # the wide start and near-arrival pin maintain the journey. No final pin.
        _add_ltx25(graph,req,121,refs[1],refs[1],[(0,refs[0]),(96,refs[1])],prompt,layout=True)
        for node in graph.values():
            if node['class_type']=='LTXAddVideoICLoRAGuide' and node['inputs']['frame_idx']==0:
                node['inputs']['strength']=.55
        (output/'parameters.json').write_text(json.dumps(dict(prompt=prompt,seed=req.seed,frames=121,
            layout_strength=.55,final_pin=False,mid_pin=96,upscale=upscale,
            sources={name:digest(folder/name) for name in ['approach.mp4','wide_photo.png','arrival_photo.png']}),indent=2))
        saved=_submit(client,graph,(LTX_VIDEO,),output)
    path=Path(saved[LTX_VIDEO]);(output/'result.json').write_text(json.dumps(dict(video=str(path)),indent=2));print(path,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,default=Path('outputs/touchdown/earth-fall-v4'))
    p.add_argument('--stage',choices=['paint','video'],required=True);p.add_argument('--url')
    p.add_argument('--manifest',type=Path,default=Path('shots/authored/canyonlands.toml'));p.add_argument('--upscale',action='store_true')
    p.add_argument('--only',choices=['wide_photo','arrival_photo'])
    a=p.parse_args();folder=a.folder.resolve()
    if a.stage=='paint':paint(folder,a.url or 'http://127.0.0.1:8192',a.manifest,a.only)
    else:video(folder,a.url or 'http://127.0.0.1:8193',a.upscale)

if __name__=='__main__':main()
