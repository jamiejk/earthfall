"""Conservative opening-camera check: coherent travel, not changing grain.

Coordinates are measured at 640 px width. This rejects a tracked hold; a
textureless/occluded view is unclear and requires visual review, never a pass.
It cannot establish photographic quality or distinguish a digital zoom.
"""
from __future__ import annotations

import cv2
import numpy as np


def pair_motion(first, second):
    def gray(frame):
        frame=cv2.resize(frame,(640,round(frame.shape[0]*640/frame.shape[1])))
        return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    a,b=gray(first),gray(second)
    points=cv2.goodFeaturesToTrack(a,maxCorners=400,qualityLevel=.015,minDistance=8)
    missing=dict(status="unclear",reason="Insufficient coherent tracked features")
    if points is None or len(points)<30:return missing
    nxt,ok,_=cv2.calcOpticalFlowPyrLK(a,b,points,None,winSize=(25,25),maxLevel=4)
    back,back_ok,_=cv2.calcOpticalFlowPyrLK(b,a,nxt,None,winSize=(25,25),maxLevel=4)
    valid=(ok.ravel()>0)&(back_ok.ravel()>0)&(np.linalg.norm(back-points,axis=2).ravel()<1)
    src,dst=points[valid,0],nxt[valid,0]
    if len(src)<30:return missing
    matrix,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=1.5)
    if matrix is None:return missing
    chosen=inliers.ravel().astype(bool);n=int(chosen.sum());ratio=n/len(src)
    hull=cv2.convexHull(src[chosen]);coverage=cv2.contourArea(hull)/(a.shape[0]*a.shape[1])
    if n<30 or ratio<.6 or coverage<.2:return dict(**missing,inliers=n,inlier_fraction=round(ratio,3),coverage=round(coverage,3))
    tracked=src[chosen];projected=tracked@matrix[:,:2].T+matrix[:,2]
    travel=float(np.median(np.linalg.norm(projected-tracked,axis=1)))
    return dict(status="moving" if travel>=.75 else "hold",median_camera_travel_px=round(travel,3),
                inliers=n,inlier_fraction=round(ratio,3),coverage=round(coverage,3))


def opening_motion(video):
    cap=cv2.VideoCapture(str(video))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS)
        frames=[]
        for t in (0,.25,.5,.75,1):
            cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*fps));ok,frame=cap.read()
            if not ok:raise ValueError("Cannot decode opening-motion evidence")
            frames.append(frame)
        intervals=[dict(start_seconds=i/4,end_seconds=(i+1)/4,**pair_motion(a,b))
                   for i,(a,b) in enumerate(zip(frames,frames[1:]))]
        states=[i["status"] for i in intervals]
        status="fail" if "hold" in states else "pass" if all(s=="moving" for s in states) else "unclear"
        return dict(status=status,reason={"fail":"Opening contains a measured camera hold.",
            "pass":"Coherent camera travel measured in every opening quarter second.",
            "unclear":"Opening camera motion needs human review; tracking was inconclusive."}[status],
            method="Forward/backward feature tracks and spatially supported robust affine fit at 640 px width",
            threshold_px_per_quarter_second=.75,intervals=intervals)
    finally:cap.release()
