"""Matched photographic LTX control/Prompt Relay experiment, outside app defaults."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from types import SimpleNamespace

import cv2

from touchdown.authored_scene import digest
from touchdown.photo_delivery import match_metrics, frame_at, video_info
from touchdown.photo_gate import require_arrival_gate
from touchdown.photo_only import asset_plan, verify_lineage
from touchdown.relay_timing import schedule, wire_relay

REPO = Path(__file__).resolve().parents[2]
RUNTIME = REPO / "outputs/touchdown/relay-runtime"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def assemble(folder, near_record, arrival_sha, portrait_frames=73, opening_handles=0):
    record = json.loads((folder/"result.json").read_text())
    for item in (record, near_record):
        if digest(Path(item["video"])) != item["sha256"] or digest(Path(item["workflow"])) != item["workflow_sha256"]:
            raise ValueError("Experiment source or graph changed")
    a, b = cv2.VideoCapture(record["video"]), cv2.VideoCapture(near_record["video"])
    try:
        count, w, h, fps = video_info(a)
        nc, nw, nh, nfps = video_info(b)
        if opening_handles not in (0,48) or (count, fps, nc, nfps) != (193+opening_handles,24,73,24) or abs(w/h-nw/nh) > .002:
            raise ValueError("Unexpected eight-second source or portrait timing")
        images = [frame_at(a, i) for i in range(count)]
        if portrait_frames not in (25,73):
            raise ValueError("Use a reviewed 25- or 73-frame photographic portrait prefix")
        seam = match_metrics(images[portrait_frames-1], frame_at(b,0))
        # A mismatched arrival is kept as a raw experiment, never disguised.
        native = seam["acceptable"]
        size = (nw,nh) if native else (w,h)
        dest = folder/"earth_zoom.mp4"
        enc = subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","bgr24",
            "-s",f"{size[0]}x{size[1]}","-r","24","-i","pipe:0","-an","-c:v","libx264",
            "-crf","17","-preset","fast","-pix_fmt","yuv420p","-movflags","+faststart",str(dest)], stdin=subprocess.PIPE)
        try:
            for source_frame in range(192,0,-1):
                im = frame_at(b,portrait_frames-1-source_frame) if native and source_frame < portrait_frames else images[source_frame]
                if im.shape[1::-1] != size:
                    im = cv2.resize(im,size,interpolation=cv2.INTER_LANCZOS4)
                enc.stdin.write(im.tobytes())
        finally:
            enc.stdin.close(); code=enc.wait()
        if code:
            raise RuntimeError("Experiment export failed")
        verify=cv2.VideoCapture(str(dest)); info=video_info(verify); verify.release()
        if info[0] != 192 or info[3] != 24:
            raise ValueError("Export did not satisfy the eight-second clock")
        result=dict(video=str(dest),sha256=digest(dest),flight=record,native_portrait=near_record,
                    arrival_sha256=arrival_sha,seam=seam,native_portrait_used=native,
                    native_start_frame=193-portrait_frames if native else None,native_first_frame=0,
                    native_frames=portrait_frames-1 if native else 0,portrait_prefix_frames=portrait_frames,
                    flight_source_mapping=list(range(192,0,-1)),output_frames=192,delivered_fps=24,
                    discarded_orbital_source_frames=list(range(193,count)),
                    opening_handle_frames=opening_handles,source_frames=count,
                    duration=8.0,delivered_size=list(size),source_speed=1.0,native_speed=1.0,
                    interpolation=False,crossfade=False,face_composite=False,three_dimensional_guides=False)
        write(folder/"delivery.json",result)
        return result
    finally:
        a.release();b.release()


def run(source, root, variants, portrait_frames=73, opening_handles=0):
    # Load private environment before importing runtime constants; never log values.
    for line in Path("/tmp/touchdown-generation-runtime.env").read_text().splitlines():
        key,value=line.split("=",1);os.environ[key]=json.loads(value)
    from touchdown.studio_build import runtime, VIDEO_URL
    import fakeout_earth_zoom as f
    cfg=json.loads((source/"scene-config.json").read_text());scene=source/"scene"
    if cfg["place"].lower() != "stonehenge":
        raise ValueError("This bounded landscape experiment is for the reviewed Stonehenge setup")
    near=source/"motion/near";nr=json.loads((near/"result.json").read_text())
    parent=json.loads((near/"provenance.json").read_text())["parent"]
    arrival_sha=digest(scene/"arrival.png")
    if parent["arrival_sha256"] != arrival_sha:
        raise ValueError("The native portrait belongs to a different arrival")
    if digest(Path(nr["video"])) != nr["sha256"] or digest(Path(nr["workflow"])) != nr["workflow_sha256"]:
        raise ValueError("The native portrait changed")
    lineage=verify_lineage(cfg,scene,asset_plan(cfg,scene));require_arrival_gate(cfg,scene,root)
    if portrait_frames not in (25,73):
        raise ValueError("This bounded comparison supports 25 or 73 native portrait frames")
    timing=schedule(cfg,opening_handles);timing["native_portrait_frames"]=portrait_frames;write(root/"timing.json",timing)
    node=RUNTIME/"custom_nodes/TouchdownPromptRelay"
    upstream=json.loads((node/"upstream.json").read_text())
    for name,meta in upstream["files"].items():
        if digest(node/name) != meta["sha256"]:
            raise ValueError("Pinned Prompt Relay source changed")
    shutil.copyfile(REPO/"scripts/touchdown/relay_node.py",node/"__init__.py")
    config=RUNTIME/"extra-paths.json"
    write(config,dict(touchdown_relay=dict(custom_nodes=str(node.parent))))
    os.environ["TOUCHDOWN_COMFY_SERVER_CMD"] += " --extra-model-paths-config "+shlex.quote(str(config))
    prefix=root/"near-retreat-prefix.mp4"
    root.mkdir(parents=True,exist_ok=True)
    subprocess.run(["ffmpeg","-v","error","-y","-i",nr["video"],"-vf",
        f"trim=end_frame={portrait_frames},reverse,setpts=PTS-STARTPTS","-an","-c:v","libx264","-crf","12",str(prefix)],check=True)
    # The photographic prefix and endpoint are appended reference planes,
    # each kept at its actual source time rather than stretching the clock.
    guide_frames=[0]+[8*i-3.5 for i in range(1,(portrait_frames-1)//8+1)]+[timing["source_frames"]-1]
    write(root/"provenance.json",dict(source=str(source),arrival_sha256=arrival_sha,near=nr,
        orbit_sha256=digest(scene/"orbit.png"),photographic_roots=lineage,upstream=upstream,
        node_sha256=digest(node/"__init__.py"),timing_module_sha256=digest(Path(__file__).with_name("relay_timing.py")),
        models=cfg["models"],ltx_models=f.LTX25_MODELS,seed=cfg["seed"]+32,frames=timing["source_frames"],fps=24,opening_handle_frames=opening_handles,
        portrait_prefix_frames=portrait_frames,retained_native_source_frames=list(range(portrait_frames-1)),
        three_dimensional_guides=False,comparison="Identical text, seed, photographic references, frames and resolution; temporal weighting differs"))
    with runtime("comfy",root):
        for variant in variants:
            folder=root/variant;folder.mkdir(parents=True,exist_ok=True)
            if (folder/"result.json").exists():
                assemble(folder,nr,arrival_sha,portrait_frames,opening_handles);continue
            started=time.monotonic();print("START",variant,flush=True)
            with f._client(VIDEO_URL) as client:
                graph={"50":dict(class_type="LoadVideo",inputs=dict(file=client.assets.from_file(str(prefix)))),
                       "51":dict(class_type="GetVideoComponents",inputs=dict(video=["50",0]))}
                orbit=f._load(graph,client,"46",scene/"orbit.png")
                wanted=f._ltx(graph,SimpleNamespace(no_upscale=True),timing["source_frames"],[(0,["51",0]),(-1,orbit)],
                              timing["single_prompt"],cfg["seed"]+32,"touchdown/stonehenge_timed_"+variant)
                graph["308"]["inputs"].update(width=1280,height=736)
                wire_relay(graph,timing,folder/"attention-audit.json",guide_frames,enabled=variant=="relay")
                saved=f._submit(client,graph,tuple(wanted),folder)
            video=Path(saved[f.LTX_VIDEO]);workflow=video.parent/"graph.json"
            write(folder/"result.json",dict(video=str(video),sha256=digest(video),workflow=str(workflow),
                workflow_sha256=digest(workflow),variant=variant,elapsed_seconds=round(time.monotonic()-started,1)))
            delivery=assemble(folder,nr,arrival_sha,portrait_frames,opening_handles)
            print("COMPLETE",variant,delivery["duration"],"seconds; native portrait:",delivery["native_portrait_used"],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=REPO/"outputs/touchdown/studio/studio-9kaxjysulpa1")
    p.add_argument("--output",type=Path,default=REPO/"outputs/touchdown/studio/stonehenge-ltx-relay-20261023")
    p.add_argument("--variants",nargs="+",choices=["baseline","relay"],default=["baseline","relay"])
    p.add_argument("--portrait-frames",type=int,choices=[25,73],default=73)
    p.add_argument("--opening-handles",type=int,choices=[0,48],default=0)
    a=p.parse_args();run(a.source.resolve(),a.output.resolve(),a.variants,a.portrait_frames,a.opening_handles)


if __name__ == "__main__":
    main()
