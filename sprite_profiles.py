"""Pairwise silhouette profiles learned by type; resample both views from one shared draw."""
import torch
from torch import nn
from torch.nn import functional as F

from prepare import TYPES
from sprite_detail import DetailedGenerator

SLICES=12
GROUPS=64
SHAPE=2*2*SLICES


def describe(pairs):
    """Horizontal centre and half-width in height units at twelve levels per view."""
    n=len(pairs)
    views=pairs.reshape(n,2,3,pairs.shape[-2],pairs.shape[-1])
    mask=views.amin(2)<.88
    height,width=mask.shape[-2:]
    x=torch.arange(width,device=pairs.device)
    y=torch.arange(height,device=pairs.device)
    top=torch.where(mask.any(-1),y,height).amin(-1).clamp_max(height-1)
    bottom=torch.where(mask.any(-1),y,0).amax(-1)
    span=(bottom-top+1).clamp_min(1)
    lo=torch.where(mask.any(-2),x,width).amin(-1)
    hi=torch.where(mask.any(-2),x,0).amax(-1)
    center=(lo+hi)*.5
    rows=(y-top[:,:,None]+.5)/span[:,:,None]
    blocks=[]
    for k in range(SLICES):
        distance=(rows-(k+.5)/SLICES).abs()
        use=distance<=torch.maximum(distance.masked_fill(~mask.any(-1),float("inf")).amin(-1,keepdim=True),
                                    distance.new_tensor(.62/SLICES))
        occupied=(mask & use[:,:,:,None]).any(-2)
        left=torch.where(occupied,x,width).amin(-1)
        right=torch.where(occupied,x,0).amax(-1)
        blocks.append(torch.stack([((left+right)*.5-center)/span,
                                   ((right-left+1)*.5/span).clamp_min(.012)],dim=2))
    return torch.stack(blocks,-1),top,span,center


def pigment(pairs):
    pixels=pairs.reshape(len(pairs),2,3,-1).permute(0,1,3,2).flatten(1,2).add(1).mul(.5)
    high=pixels.amax(2)
    low=pixels.amin(2)
    chroma=high-low
    weight=((high>.16)&(low<.90)).float()*(.04+chroma.square())
    return (pixels*weight[:,:,None]).sum(1)/weight.sum(1,keepdim=True).clamp_min(.01)


class ProfileGenerator(nn.Module):
    def __init__(self, learned_palette=False):
        super().__init__()
        self.drawing=DetailedGenerator()
        self.drawing.base.expressive=True
        self.drawing.base.anatomy=True
        self.register_buffer("centers",torch.zeros(GROUPS,SHAPE+3))
        self.register_buffer("basis",torch.zeros(GROUPS,5,SHAPE+3))
        self.register_buffer("freq",torch.ones(len(TYPES),len(TYPES),GROUPS)/GROUPS)
        self.register_buffer("type_color",torch.full((len(TYPES),len(TYPES),3),.5))
        self.learned_palette=learned_palette
        if learned_palette:
            self.register_buffer("pigment_freq", torch.full((len(TYPES),len(TYPES),24),1/24))

    def forward(self,z,y,traits=None):
        # Split dense batches to bound the memory used by the shared 3-D renderer.
        if len(z)>6:
            groups = [None] * len(z.split(6)) if traits is None else traits.split(6)
            return torch.cat([self.forward(a,b,c) for a,b,c in zip(z.split(6),y.split(6),groups)])
        i,j=self.drawing.base.type_ids(y)
        if self.learned_palette:
            body=self.drawing.base
            # A shared draw from pigments characteristic of this combination.
            frequency=self.pigment_freq[i,j]
            background=self.pigment_freq.diagonal(dim1=0,dim2=1).mean(1)
            odds=frequency.pow(1.65)/background.clamp_min(.002).pow(.65)
            pick=body.gumbel(odds,z[:,8:32],1.5)
            colors=body.tones[pick].clone()
            if traits is not None:
                colors[:,0]=torch.lerp(colors[:,0], traits[:,3:], .79)
            rgb,_=body.render(z,y,body.choose(z,y),hard=True,colors=colors)
            base=self.drawing.detail(F.pad(rgb.flatten(1,2),(16,16,16,16),value=1))
        else:
            base=self.drawing(z,y)
        return self.apply_profile(base,z,i,j,traits)

    def apply_profile(self,base,z,i,j,traits=None):
        """Fit paired horizontal outlines to a sampled TRAIN profile."""
        p,top,span,center=describe(base)
        affinity=(self.centers[None,:,:SHAPE]-p.flatten(1)[:,None]).square().mean(2)
        logits=self.freq[i,j].clamp_min(1e-7).log()-4*affinity
        choice=self.drawing.base.gumbel(logits.softmax(1),z[:,24:88],.85)
        sample=self.centers[choice]+.64*(self.basis[choice]*z[:,88:93,None].clamp(-2,2)).sum(1)
        goal=sample[:,:SHAPE].reshape_as(p)
        strength=.30 if self.learned_palette else .54
        target_mid=p[:,:,0]+strength*(goal[:,:,0]-p[:,:,0]).clamp(-.24,.24)
        factor=(goal[:,:,1].clamp_min(.035)/p[:,:,1].clamp_min(.035)).clamp(.54,1.95).pow(strength)
        if traits is not None:
            current=p[:,:,1].amax(2)*2
            stretch=(traits[:,:1]/current.clamp_min(.2)).clamp(.7,1.5).sqrt()
            band=torch.linspace(-1,1,SLICES,device=z.device)
            factor=factor*stretch[:,:,None]*(1+.18*traits[:,1,None,None]*band)
        # Inverse horizontal deformation, shared conditional sample and independent projections.
        n,_,h,w=base.shape
        axis=torch.arange(w,device=z.device,dtype=z.dtype)
        level=((axis-top[:,:,None]+.5)/span[:,:,None]*SLICES-.5).clamp(0,SLICES-1)
        low=level.long()
        def rows(a):
            return torch.lerp(a.gather(2,low),a.gather(2,(low+1).clamp_max(SLICES-1)),level-low)
        xp=(axis-center[:,:,None])/span[:,:,None]
        src=(xp[:,:,None,:]-rows(target_mid)[:,:,:,None])/rows(factor)[:,:,:,None]+rows(p[:,:,0])[:,:,:,None]
        src=src*span[:,:,None,None]+center[:,:,None,None]
        grid=torch.stack([2*src/(w-1)-1,(2*axis/(h-1)-1)[None,None,:,None].expand_as(src)],-1)
        rgb=1+F.grid_sample(base.reshape(n*2,3,h,w)-1,grid.reshape(n*2,h,w,2),
                            mode="nearest",align_corners=True).reshape(n,2,3,h,w)
        if self.learned_palette:
            # Occupy more of the native 96-pixel canvas for small bodies. Preserve
            # aspect ratio, discrete pixels, and the pose in each projection.
            mask=rgb.amin(2)<.88
            left=torch.where(mask.any(-2),axis,w).amin(-1)
            right=torch.where(mask.any(-2),axis,0).amax(-1)
            midx=(left+right)*.5
            midy=top+(span-1)*.5
            longest=torch.maximum(right-left+1,span)
            zoom=(61/longest.clamp_min(1)).clamp(1.,1.65)
            xx=(axis-(w-1)/2)/zoom[:,:,None]+midx[:,:,None]
            yy=(axis-(h-1)/2)/zoom[:,:,None]+midy[:,:,None]
            source=torch.stack(torch.broadcast_tensors(xx[:,:,None,:]*2/(w-1)-1,yy[:,:,:,None]*2/(h-1)-1),-1)
            return 1+F.grid_sample(rgb.flatten(0,1)-1,source.reshape(n*2,h,w,2),mode="nearest",
                                   align_corners=True).reshape_as(base)
        # Transfer TRAIN chroma while keeping linework, eyes, and luminance steps intact.
        old=pigment(base)
        target=.68*self.type_color[i,j]+.32*sample[:,SHAPE:].clamp(0,1)
        def chroma(color):
            return color-color.mean(1,keepdim=True)
        delta=(chroma(target)-chroma(old)).clamp(-.38,.38)
        rgb=(rgb+1)*.5
        value=rgb.mean(2,keepdim=True)
        weight=(3.7*value*(1-value)).clamp(0,1)*(rgb.amin(2,keepdim=True)<.90)
        rgb=(rgb+.92*delta[:,None,:,None,None]*weight).clamp(0,1)
        return (2*rgb-1).flatten(1,2)
