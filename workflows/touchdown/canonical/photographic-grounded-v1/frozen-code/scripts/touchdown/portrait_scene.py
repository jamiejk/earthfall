"""Small deterministic 2.5D dolly from a portrait; approximate depths, not reconstruction."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from PIL import Image
from . import video as V


def extract_person(image, coat_max_y_fraction=1.0):
    h,w=image.shape[:2];hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    coat=(hsv[:,:,0]>15)&(hsv[:,:,0]<40)&(hsv[:,:,1]>110)&(hsv[:,:,2]>70)
    coat[int(h*coat_max_y_fraction):]=False
    coat[:int(h*.3)]=False;coat[:, :int(w*.4)]=False;coat[:,int(w*.6):]=False
    count,labels,stats,_=cv2.connectedComponentsWithStats(coat.astype(np.uint8))
    if count<2: raise ValueError('No central mustard coat found in this bounded test')
    x,y,bw,bh,area=stats[1+np.argmax(stats[1:,4])]
    if area<100: raise ValueError('Coat region too small')
    x0=max(0,x-bw//2);x1=min(w,x+bw+bw//2)
    y0=max(0,y-int(bh*.6));y1=min(h,y+int(bh*2.8))
    mask=np.zeros((h,w),np.uint8)
    cv2.grabCut(cv2.cvtColor(image,cv2.COLOR_RGB2BGR),mask,(int(x0),int(y0),int(x1-x0),int(y1-y0)),np.zeros((1,65),float),np.zeros((1,65),float),5,cv2.GC_INIT_WITH_RECT)
    silhouette=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)).astype(np.uint8)
    n,components=cv2.connectedComponents(silhouette)
    candidates=components[coat & (components>0)]
    if not len(candidates): raise ValueError('Extracted person does not include coat')
    selected=int(np.bincount(candidates).argmax())
    silhouette=(components==selected).astype(np.uint8)
    ys,xs=np.where(silhouette)
    if len(xs)<100: raise ValueError('Person extraction failed')
    return silhouette,(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))


def dolly_maps(depth, distance):
    """Inverse projection for a straight backward translation; no frame blending."""
    h,w=depth.shape;yy,xx=np.mgrid[:h,:w]
    # current pixel ray intersects source surface at source axial depth.
    scale=1+distance/np.maximum(depth,.1)
    return ((xx-(w-1)/2)*scale+(w-1)/2).astype(np.float32),((yy-(h-1)/2)*scale+(h-1)/2).astype(np.float32)


def ground_maps(shape,height,distance):
    h,w=shape;yy,xx=np.mgrid[:h,:w];focal=w/(2*np.tan(np.pi/6))
    z=height*focal/np.maximum(yy-(h-1)/2,1)
    scale=z/np.maximum(z-distance,.1)
    return ((xx-(w-1)/2)*scale+(w-1)/2).astype(np.float32),((yy-(h-1)/2)*scale+(h-1)/2).astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--endpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=73);p.add_argument('--travel',type=float,default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    image=np.asarray(Image.open(a.endpoint).convert('RGB').resize((960,544)))
    person,box=extract_person(image);h,w=image.shape[:2];focal=w/(2*np.tan(np.pi/6));person_depth=focal*1.7/max(box[3]-box[1],1)
    # Ground plane calibrated to the extracted feet; this is inferred camera height.
    height=person_depth*(box[3]-(h-1)/2)/focal
    if not .5<height<2.5: raise ValueError('Portrait cannot support the assumed upright ground plane')
    floor=((cv2.cvtColor(image,cv2.COLOR_RGB2HSV)[:,:,1]<65)|person.astype(bool))&(np.indices((h,w))[0]>h*.65)
    hole=cv2.dilate(person,np.ones((7,7),np.uint8))*255
    background=cv2.inpaint(cv2.cvtColor(image,cv2.COLOR_RGB2BGR),hole,5,cv2.INPAINT_TELEA)
    background=cv2.cvtColor(background,cv2.COLOR_BGR2RGB)
    yy=np.indices((h,w))[0];depth=np.full((h,w),60,np.float32)
    # Each layer is projected independently and composited with its own mask.
    Image.fromarray(person*255).save(a.output/'person_mask.png');Image.fromarray(background).save(a.output/'background.png')
    records=[]
    with V.Writer(a.output/'guide.mp4') as writer:
        for i in range(a.frames):
            t=i/(a.frames-1);smooth=t*t*(3-2*t);distance=a.travel*(1-smooth)
            mx,my=dolly_maps(depth,distance)
            frame=cv2.remap(background,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            gmx,gmy=ground_maps((h,w),height,distance)
            ground=cv2.remap(background,gmx,gmy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            ground_alpha=cv2.remap(floor.astype(np.uint8),gmx,gmy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE).astype(bool)
            frame[ground_alpha]=ground[ground_alpha]
            pmx,pmy=dolly_maps(np.full((h,w),person_depth,np.float32),distance)
            pixels=cv2.remap(image,pmx,pmy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            alpha=cv2.remap(person,pmx,pmy,cv2.INTER_NEAREST).astype(bool);frame[alpha]=pixels[alpha]
            if i==a.frames-1: frame=image  # exact endpoint; no layered approximation at the pin
            Image.fromarray(frame).save(a.output/f'{i:04d}.png')
            writer.write(cv2.resize(frame,(V.W,V.H)));records.append({'frame':i,'camera_retreat_m':distance})
    (a.output/'scene.json').write_text(json.dumps({'person_box':box,'person_depth_m':person_depth,'camera_height_m':height,'background_depth_m':60,'travel_m':a.travel,'poses':records,'limitations':['Estimated depth; rails remain on background layer','Background borders reflected; unknown floor borders replicated','Person hole filled by deterministic inpaint','Only a small dolly; cannot replace the full swoop']},indent=2))

if __name__=='__main__':main()
