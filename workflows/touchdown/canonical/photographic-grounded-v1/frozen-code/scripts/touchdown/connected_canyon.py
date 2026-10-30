"""One connected reference-coloured canyon surface, sampled from cached terrain."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np


class CanyonSurface:
    def __init__(self,folder,heading=195.,landing=(-14.,4.)):
        self.near=np.load(Path(folder)/'height_detail.npy')
        self.far=np.load(Path(folder)/'height_far.npy')
        self.heading=math.radians(heading);self.landing=landing
        self.datum=float(self._sample(self.near,1,np.array(landing[0]),np.array(landing[1])))

    @staticmethod
    def _sample(array,step,east,north):
        n=array.shape[0]
        x=np.clip((n-1)/2+east/step,0,n-1.00001)
        y=np.clip((n-1)/2-north/step,0,n-1.00001)
        x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
        fx=x-x0;fy=y-y0
        return ((array[y0,x0]*(1-fx)+array[y0,x0+1]*fx)*(1-fy)+
                (array[y0+1,x0]*(1-fx)+array[y0+1,x0+1]*fx)*fy)

    def height(self,x,y):
        x=np.asarray(x);y=np.asarray(y)
        east=self.landing[0]+x*math.cos(self.heading)+y*math.sin(self.heading)
        north=self.landing[1]-x*math.sin(self.heading)+y*math.cos(self.heading)
        blend=np.clip((395-np.maximum(np.abs(east),np.abs(north)))/60,0,1)
        blend=blend*blend*(3-2*blend)
        result=self._sample(self.near,1,east,north)*blend+self._sample(self.far,50,east,north)*(1-blend)-self.datum
        # Art-direct a continuous lookout spur in the cached terrain. This
        # removes the broad empty apron without adding separate cliff objects.
        def smooth(value):
            value=np.clip(value,0,1);return value*value*(3-2*value)
        width=6+np.maximum(0,-y)*.16
        side=smooth((np.abs(x)-width)/12)
        end=smooth((y-4)/18)
        corridor=np.maximum(side,end)
        local=smooth((y+1000)/180)*(1-smooth((y-650)/350))*(1-smooth((np.abs(x)-700)/400))
        result-=corridor*local*(150+25*np.sin(y*.009)*np.sin(x*.007))
        # Flatten centimetres of relief where the supplied person stands.
        contact=np.clip((x*x+y*y)/4,0,1)
        return result*contact
