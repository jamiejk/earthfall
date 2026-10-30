"""Recorded local vision decisions for the canonical app automation layer.

These replace the assistant's prior per-run inspection, not the generation recipe.
Unclear or failed evidence holds the job; there are no pre-filled approvals.
"""
from pathlib import Path
from typing import Literal
import json
from pydantic import BaseModel,Field,ConfigDict,field_validator

class Finding(BaseModel):
    model_config=ConfigDict(extra='forbid')
    status:Literal['pass','fail','unclear']
    reason:str=Field(min_length=10,max_length=1500)
    geometry_preserved:bool
    usable_for_next_stage:bool
    visible_text:bool

class Region(BaseModel):
    model_config=ConfigDict(extra='forbid')
    material:Literal['land','water']
    polygon:list[list[float]]=Field(min_length=3,max_length=24)
    @field_validator('polygon')
    @classmethod
    def points(cls,points):
        if any(len(p)!=2 or any(not 0<=v<=1 for v in p) for p in points):raise ValueError('Invalid polygon coordinates')
        return points

class Plan(BaseModel):
    model_config=ConfigDict(extra='forbid')
    status:Literal['pass','fail','unclear']
    reason:str=Field(min_length=10,max_length=1000)
    background:Literal['land','water']
    features:list[Region]=Field(max_length=8)
    target_uv:list[float]=Field(min_length=2,max_length=2)
    layout_description:str=Field(min_length=10,max_length=1800)
    expansion_description:str=Field(min_length=10,max_length=1800)
    @field_validator('target_uv')
    @classmethod
    def point(cls,p):
        if any(not 0<=v<=1 for v in p):raise ValueError('Invalid destination position')
        return p

class Exit(BaseModel):
    model_config=ConfigDict(extra='forbid')
    status:Literal['pass','fail','unclear']
    frame:int
    reason:str=Field(min_length=10,max_length=1000)


def samples(video,folder,indices):
    import cv2
    folder.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(str(video));count=int(cap.get(7));images=[]
    try:
        for index in sorted(set(min(count-1,max(0,int(i))) for i in indices)):
            cap.set(1,index);ok,frame=cap.read()
            if not ok:raise ValueError('Cannot decode visual-review frame')
            height,width=frame.shape[:2]
            frame=cv2.resize(frame,(960,round(height*960/width)))
            path=folder/f'frame-{index:04}.jpg';cv2.imwrite(str(path),frame)
            images.append((path,f'FRAME {index} / {index/24:.3f}s'))
    finally:cap.release()
    return images,count


def review(root,name,assets,description):
    from touchdown import studio_build as studio
    from touchdown.studio_brief import chat
    from touchdown.canonical_build import sha
    folder=root/'automatic-reviews'/name;folder.mkdir(parents=True,exist_ok=True)
    if name=='texture':
        from touchdown.texture_gate import check
        result=check(root/assets[0]['path'],folder)
        result['record']=str(result['record'].relative_to(root))
        return result
    if name=='final':
        from touchdown.opening_motion_gate import check
        motion,record=check(root/assets[0]['path'],folder/'opening-motion')
        if motion['status']=='fail':
            return dict(continue_=False,reason=motion['reason'],finding=motion,
                        record=str(record.relative_to(root)),record_sha256=sha(record))
    images=[]
    for asset in assets:
        path=root/asset['path']
        if path.suffix=='.mp4':
            import cv2
            cap=cv2.VideoCapture(str(path));count=int(cap.get(7));cap.release()
            indices=[0,4,8,12,20,32,38,39,40,44,64,count-24,count-1] if name=='final' else [0,16,32,40,48,56,64,80,96,120,count-24,count-1]
            frames,_=samples(path,folder/'frames',indices);images+=frames
        else:images.append((path,path.stem))
    # Include the parent guide for appearance/geometry comparisons.
    parent=None
    if name.startswith('expand-'):
        index=int(name[-1]);parent=root/('terrain/texture.png' if index==1 else f'terrain/expand-{index-1}/raw-outpaint.png')
    elif name=='orbit':parent=root/'terrain/globe-guide.png'
    if parent is not None:images.insert(0,(parent,'PARENT GUIDE: compare broad geometry, account for changed scale'))
    with studio.runtime('vision',folder) as runtime:
        if name=='layout':
            frames,count=samples(root/assets[0]['path'],folder/'exit-candidates',[64])
            selection=chat(studio.VISION_URL,frames,'Decide whether this listed frame provides readable LANDSCAPE geometry after the cloud passage: connected landforms, coastlines or inland land/water patterns from which a rough plan can be inferred. An oblique or low view with sky and a visible landmark IS acceptable. Do not require high altitude, a microscopic landmark or a 30,000-foot view. Opaque cloud, an isolated portrait or unreadable land/water relationships are not usable. Return its actual frame index and pass only if usable; otherwise unclear.',Exit,folder/'exit-selection.json')
            if selection['status']!='pass' or selection['frame'] not in [int(p.stem.split('-')[1]) for p,_ in frames]:
                return dict(continue_=False,reason='No confident cloud-exit selection. '+selection['reason'],record=str((folder/'exit-selection.json').relative_to(root)),record_sha256=sha(folder/'exit-selection.json'))
            frame=selection['frame'];selected,_=samples(root/assets[0]['path'],folder/'chosen-exit',[frame])
            answer=chat(studio.VISION_URL,selected,'Author a SIMPLE CREATIVE flat land/water guide inspired by this landscape. This is NOT map recovery, georeferencing or an accurate perspective transformation. Approximation and distortion are EXPECTED; no real-world coordinates or surveyed accuracy are requested. The accepted workflow used a hand-drawn cartoon-like land/water silhouette as a conditioning guide. Do that same artistic planning task: choose a land or water background, sketch the broad visible connected landforms with a few coarse polygons, and mark the approximate destination. Use normalized DRAWING coordinates [x,y] between 0 and 1, x to the right and y down; these are positions on your invented flat drawing, not actual terrain coordinates. Keep the broad relationship of water and land; do not add unseen islands or lakes. Use at most 8 polygons with at most 16 points each. If broad water is absent, use land background and no polygons. Describe landscape materials and how the region should continue outside the drawing. Pass if you can author a usable rough guide; oblique perspective and inability to measure true geography are not failures for this task. Fail or unclear only if the source is opaque cloud, an isolated person, or the land/water relationship cannot be read at all.',Plan,folder/'layout-response.json')
            layout=dict(cloud_exit_frame=frame,target_uv=answer['target_uv'],colours={'background':'#697b50' if answer['background']=='land' else '#2e5665','land':'#697b50','water':'#2e5665'},features=answer['features'],layout_description=answer['layout_description'],expansion_description=answer['expansion_description'],provenance='Qwen-derived approximate plan from this fresh flight; not surveyed geography',measured_geography=False)
            return dict(continue_=answer['status']=='pass',reason=answer['reason'],layout=layout,record=str((folder/'layout-response.json').relative_to(root)),record_sha256=sha(folder/'layout-response.json'),model=runtime)
        prompt=(f'Assess this stage of an Earth-to-landmark flight: {name}. Required: {description} '
                'Return pass only if the supplied evidence supports usable geometry and continuity for the next stage. Uncertain evidence means unclear, never automatic pass. '
                'For intermediate texture/outpaint/project guides, low detail or minor text is a recorded defect, not alone a failure if broad geometry remains useful; the orbital FINISH and FINAL film must contain no visible map labels/text. '
                'For outpaints, the central prior region is intentionally half the image width and height; compare ancestry at that scale. '
                'For chronological frames, look for increasing destination scale and viewpoint change, not merely drifting clouds or opacity transitions. A clear cut, dissolve or stalled opening fails. '
                'For flight before grounded opening replacement, assess usable cloud passage and landing; its old orbital start will be replaced. Report visible limitations candidly.')
        answer=chat(studio.VISION_URL,images,prompt,Finding,folder/'response.json')
    accepted=answer['status']=='pass' and answer['geometry_preserved'] and answer['usable_for_next_stage'] and not (answer['visible_text'] and name in ('orbit','final'))
    return dict(continue_=accepted,reason=answer['reason'],finding=answer,record=str((folder/'response.json').relative_to(root)),record_sha256=sha(folder/'response.json'),model=runtime)
