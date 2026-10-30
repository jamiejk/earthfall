"""Overhead guidance v2: typed surface description, map-only geometry reference.

A named variant of the accepted texture stage. No oblique landmark photograph or
free-form scene composition is passed into the texture/expansion prompts.
"""
from pathlib import Path
from typing import Literal
import json,os,shlex
from pydantic import BaseModel,ConfigDict,Field

class Surface(BaseModel):
    model_config=ConfigDict(extra='forbid')
    status:Literal['pass','unclear']
    visible_broad_water:bool
    material:Literal['desert','grassland','forest','rock','snow','urban','mixed']
    reason:str=Field(min_length=10,max_length=1200)

MATERIALS={
 'desert':('#b7a17b','Pale tan arid sand and stone, subtle dry drainage patterns and sparse scrub'),
 'grassland':('#697b50','Muted green-brown grassland, irregular field patterns and small drainage features'),
 'forest':('#50654a','Irregular muted green forest canopy and natural clearings'),
 'rock':('#928675','Weathered grey-brown rock and natural fine drainage patterns'),
 'snow':('#d7d9d6','Pale snow and ice with fine natural surface texture'),
 'urban':('#9b988f','Fine small-scale grey roof blocks, streets and green spaces'),
 'mixed':('#8c896a','Muted natural green-brown land with fine irregular surface texture')}

def normalise(root):
    import cv2
    from touchdown import studio_build as studio
    from touchdown.studio_brief import chat
    from touchdown.canonical_build import read,write,Held,sha
    layout=read(root/'scene-layout.json');folder=root/'layout-normalization';folder.mkdir()
    source=root/'flight/run/motion/earth_zoom.mp4'
    cap=cv2.VideoCapture(str(source));cap.set(1,layout['cloud_exit_frame']);ok,image=cap.read();cap.release()
    if not ok:raise Held('Cannot read the selected landscape frame')
    photo=folder/'landscape.png';cv2.imwrite(str(photo),image)
    with studio.runtime('vision',folder):
        surface=chat(studio.VISION_URL,[(photo,'LANDSCAPE MATERIALS')],
            'Describe broad ground materials and whether a broad body of water is actually VISIBLE. '
            'Ignore monuments, people, sky and clouds. Desert sand is desert, not green grassland. '
            'Do not invent a river/lake from knowing the location. Select unclear if the ground cannot be read.',
            Surface,folder/'surface.json')
    if surface['status']!='pass':raise Held('Unclear regional surface: '+surface['reason'])
    colour,description=MATERIALS[surface['material']]
    # Model-written composition text is not an instruction channel for image generation.
    features=layout['features'] if surface['visible_broad_water'] else []
    background=layout['colours']['background'] if surface['visible_broad_water'] else colour
    if surface['visible_broad_water'] and background==layout['colours'].get('land'):background=colour
    result={**layout,'colours':{'background':background,'land':colour,'water':'#2e5665'},'features':features,
        'layout_description':'Preserve the flat plan’s broad land/water shapes and positions. '+description+'. All surface detail is viewed vertically from above at uniform map scale.',
        'expansion_description':description+'. Continue the existing broad land/water arrangement without adding isolated landmarks or inventing new water bodies.',
        'provenance':'overhead-guidance-v2: existing destination UV/polygons plus independent material observation; no measured geography',
        'guidance_variant':'overhead-guidance-v2','surface_material':surface['material']}
    write(folder/'normalization.json',dict(source_layout=layout,source_frame_sha256=sha(photo),surface=surface,normalised_layout=result))
    write(root/'scene-layout.json',result)

def generate(root):
    from touchdown import scene_grounded_workflow as g,studio_build as studio
    from touchdown.authored_scene import generate_asset
    g.environment();g.verify(root)
    selected,cfg=g.case(root);layout=g.read(root/'scene-layout.json');folder=root/'terrain'
    description=MATERIALS[layout['surface_material']][1]
    spec=dict(name='texture',references=[str(folder/'regional-layout.png')],models=cfg['models'],seed=cfg['seed']+1,width=1920,height=1088,
        prompt='One realistic satellite orthophotograph looking vertically STRAIGHT DOWN onto a broad ground region. '
        'Use the flat drawing ONLY as the plan of land and water shapes; replace flat colour with natural fine photographic surface texture. '
        +description+'. '+layout['layout_description']+' '
        'Uniform map scale and vertical camera at every edge. A continuous ground surface fills the image. '
        'Buildings, if any, are tiny roof marks. No recognisable individual monument, no foreground, background, horizon or sky. '
        'No oblique view, perspective recession, building side faces, hero landmark, map labels, text or symbols.')
    g.write(folder/'texture-spec.json',spec)
    g.write(folder/'texture-variant.json',dict(variant='overhead-guidance-v2',changes=['typed surface materials','map-only image reference','no model-authored scene composition'],seed=spec['seed']))
    os.environ['TOUCHDOWN_COMFY_SERVER_CMD']+=' --deterministic --extra-model-paths-config '+shlex.quote(str(root/'flight/pinned-model-paths.json'))
    with studio.runtime('comfy',folder):generate_asset(studio.COMFY_URL,spec,folder)
    g.verify(root)
