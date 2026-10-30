"""Independent camera observations for overhead textures; never grade reference photos."""
from pathlib import Path
from typing import Literal
import json
from pydantic import BaseModel,ConfigDict,Field

class CameraObservation(BaseModel):
    model_config=ConfigDict(extra='forbid')
    camera_view:Literal['straight_down','oblique','unclear']
    horizon:Literal['sky_meets_ground','no_sky','unclear']
    large_landmark_visible:bool
    building_side_faces_visible:bool
    broad_surface_texture:bool
    reason:str=Field(min_length=10,max_length=1500)

def accepted(answer):
    return (answer['camera_view']=='straight_down' and answer['broad_surface_texture']
            and answer['horizon']=='no_sky'
            and not any(answer[k] for k in ('large_landmark_visible','building_side_faces_visible')))

def check(image,folder):
    from touchdown import studio_build as studio
    from touchdown.studio_brief import chat
    from touchdown.authored_scene import digest
    folder.mkdir(parents=True,exist_ok=True)
    report=folder/'camera-observation.json'
    with studio.runtime('vision',folder) as runtime:
        answer=chat(studio.VISION_URL,[(image,'CANDIDATE IMAGE ONLY')],
            'Describe the visible camera angle and content of this ONE image. Do not judge attractiveness or recognisability. '
            'straight_down means a vertical map-like overhead surface with no perspective recession toward the top; '
            'oblique means looking forward/diagonally, visible monument side faces or perspective depth. '
            'A horizon is SKY meeting the ground. A coastline, riverbank or texture edge is NOT a horizon. If all pixels are ground/water with no sky, select no_sky. '
            'A pyramid dominating the image is a large landmark, even if photographed from above. '
            'broad_surface_texture means a wide map-like region with only small-scale natural/manmade details. '
            'State what is actually visible. No reference photograph or requested outcome is provided.',
            CameraObservation,report)
    passed=accepted(answer)
    data=json.loads(report.read_text())
    data['answer']['status']='pass' if passed else 'fail'
    data['decision_policy']='straight_down AND broad_surface_texture AND no sky boundary/large landmark/side faces'
    report.write_text(json.dumps(data,indent=2)+'\n')
    return dict(continue_=passed,reason=answer['reason'],finding=answer,record=report,record_sha256=digest(report),model=runtime)
