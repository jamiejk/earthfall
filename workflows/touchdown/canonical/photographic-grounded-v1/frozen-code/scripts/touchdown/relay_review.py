"""Traceable frame comparison and advisory local checks for the timing trial."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, ConfigDict, Field

from touchdown.authored_scene import digest
from touchdown.relay_timing import NAMES, BOUNDS
from touchdown.opening_motion import opening_motion

TIMES = [0, .75, 1.5, 2, 2.5, 3.5, 4.5, 5, 6.5, 191/24]


class Stage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_id: str
    stage: Literal["orbit", "above_clouds", "inside_cloud", "high_aerial", "low_aerial", "person_prominent", "unclear"]
    evidence: str = Field(min_length=5,max_length=400)


class Stages(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[Stage]


def sheet(root, variants=("baseline","relay")):
    font=ImageFont.load_default(size=18)
    canvas=Image.new("RGB",(640*len(variants),2840),(18,20,23));draw=ImageDraw.Draw(canvas)
    draw.text((12,10),"LTX 2.5 — eight-second photographic timing experiment",font=font,fill="white")
    for column,variant in enumerate(variants):
        result=json.loads((root/variant/"result.json").read_text())
        label=("5-second photo anchor" if result["photo_anchor"] else "shorter approach, no middle anchor") if "photo_anchor" in result else ("one global prompt" if variant=="baseline" else "prompts scheduled over time")
        draw.text((column*640+12,40),variant+": "+label,font=font,fill="white")
        cap=cv2.VideoCapture(str(root/variant/"earth_zoom.mp4"))
        try:
            for row,t in enumerate(TIMES):
                frame=min(round(t*24),191);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read()
                if not ok:raise ValueError("Cannot decode comparison")
                phase=next(NAMES[i] for i in range(5) if BOUNDS[i] <= frame < BOUNDS[i+1])
                x=column*640+12;y=72+row*274
                draw.text((x,y),f"{frame/24:.3f}s — requested {phase}",font=font,fill="white")
                pic=ImageOps.contain(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(616,244))
                canvas.paste(pic,(x,y+24))
        finally:cap.release()
    canvas.save(root/"comparison.jpg",quality=93)


def review(root, source, variants=("baseline","relay")):
    for line in Path("/tmp/touchdown-generation-runtime.env").read_text().splitlines():
        key,value=line.split("=",1);os.environ[key]=json.loads(value)
    from touchdown.studio_build import runtime,VISION_URL,sample_preview
    from touchdown import vision_check as vision
    from touchdown.studio_brief import chat,EndingReview,POSES,FRAMINGS,independent_delivery_review,high_view_review
    setup=json.loads((source/"setup.json").read_text());old=json.loads((source/"outcome.json").read_text())
    sheet(root,variants)
    with runtime("vision",root):
        for variant in variants:
            folder=root/variant;delivery=json.loads((folder/"delivery.json").read_text());video=Path(delivery["video"])
            if digest(video)!=delivery["sha256"]:raise ValueError("Compared video changed")
            opening=opening_motion(video)
            (folder/"opening-motion.json").write_text(json.dumps(opening,indent=2))
            evidence=vision.prepare(video,TIMES,folder/"gates/timing/evidence")
            answers=[]
            for offset in (0,5):
                batch=evidence["frames"][offset:offset+5]
                result=chat(VISION_URL,[(Path(f["image"]),f["frame_id"]) for f in batch],
                    "Describe each labelled frame independently. Return exactly one observation for each supplied frame ID. "
                    "Classify the ACTUAL visible scale: orbit has Earth curvature, above_clouds is looking over cloud tops, "
                    "inside_cloud is a largely obscured view from within cloud, high_aerial shows many kilometres of detailed "
                    "ground far below with local structures microscopic, low_aerial has a large landmark or near ground, "
                    "person_prominent means a clearly recognisable foreground person. Use unclear for ambiguous evidence. "
                    "Do not infer altitude from a desired camera move. Describe what you see, not a requested timeline.",
                    Stages,folder/f"gates/timing/batch-{offset}.json")
                if sorted(o["frame_id"] for o in result["observations"])!=sorted(f["frame_id"] for f in batch):
                    raise ValueError("Timing checker did not return the supplied frames exactly")
                answers.extend(result["observations"])
            (folder/"previews").mkdir(exist_ok=True)
            for name,t in [("orbit",0),("approach",4.5),("portrait",191/24)]:
                sample_preview(video,t,folder/f"previews/{name}.jpg")
            sample_preview(source/"scene/arrival.png",0,folder/"previews/arrival.jpg")
            high=[]
            for i,t in enumerate([2.5,3,3.5]):
                pic=folder/f"gates/high/frame-{i}.jpg";sample_preview(video,t,pic)
                high.append(high_view_review(VISION_URL,pic,pic.with_suffix(".json")))
            near_start=(delivery["native_start_frame"]/24 if delivery["native_start_frame"] is not None else 6.5)
            ending=vision.prepare(video,[near_start+(191/24-near_start)*i/4 for i in range(5)],folder/"gates/ending/evidence")
            controls=chat(VISION_URL,[(Path(f["image"]),f["frame_id"]) for f in ending["frames"]],
                f"Chronological ending frames. Requested action: {POSES[setup['ending']]} "
                f"Requested final crop: {FRAMINGS[setup['framing']]}. Judge crop from the last frame and action across "
                "the sequence. Arms must remain open for an arms-open hold. Do not certify facial sharpness. Say unclear "
                "if the images do not establish the requested action.",EndingReview,folder/"gates/ending/report.json")
            raw_controls=controls
            controls,crop,orbit=independent_delivery_review(VISION_URL,Path(ending["frames"][-1]["image"]),
                folder/"previews/orbit.jpg",controls,setup["framing"],folder/"gates/ending")
            timing_report=dict(status="human review required",video_sha256=digest(video),
                observations=[dict(**o,seconds=next(f["time_seconds"] for f in evidence["frames"] if f["frame_id"]==o["frame_id"]))
                              for o in answers])
            # This is an experiment even when the advisory checks pass. Human
            # motion/photographic review remains necessary before promotion.
            body=dict(status="held",checks_passed=False,setup_id=setup["id"],
                engine="LTX 2.5 timed photographic experiment: "+variant,
                message="Eight-second timing experiment. Review clouds, high aerial timing, motion and final framing.",
                input_hashes=old["input_hashes"],references=old["references"],arrival_review=old["arrival_review"],
                reused_arrival_sha256=digest(source/"scene/arrival.png"),source_generation=json.loads((folder/"result.json").read_text()),
                video="earth_zoom.mp4",video_sha256=digest(video),duration=8.0,
                width=delivery["delivered_size"][0],height=delivery["delivered_size"][1],delivery=delivery,
                opening_motion_review=opening,
                motion_review=dict(status=opening["status"],reason=opening["reason"]+" Human review of the complete flight is still required."),
                timing_review=timing_report,cloud_exit_reviews=high,raw_ending_review=raw_controls,
                ending_review=controls,final_crop_observation=crop,orbital_scale_review=orbit,
                facial_clarity="Native moving portrait retained when the measured seam agrees; human sharpness review required.")
            (folder/"outcome.json").write_text(json.dumps(body,indent=2))
            (folder/"provenance.json").write_text((root/"provenance.json").read_text())
            print("REVIEWED",variant,json.dumps(timing_report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--sheet-only",action="store_true")
    p.add_argument("--variants",nargs="+",choices=["baseline","relay"],default=["baseline","relay"])
    a=p.parse_args()
    if a.sheet_only:sheet(a.root.resolve(),a.variants)
    else:review(a.root.resolve(),a.source.resolve(),a.variants)


if __name__=="__main__":main()
