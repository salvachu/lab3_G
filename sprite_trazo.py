"""Paired, type-conditioned shape and pigment reconstruction on the native 96px grid."""
import torch
from torch import nn
from torch.nn import functional as F
from prepare import TYPES
from sprite_ficha import FichaGenerator


class OutlinePainter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode=nn.Sequential(nn.Conv2d(4,20,5,padding=2),nn.SiLU(),
                                  nn.Conv2d(20,32,4,2,1),nn.SiLU())
        self.small=nn.Sequential(nn.Conv2d(32,40,4,2,1),nn.SiLU())
        self.paired=nn.Sequential(nn.Conv2d(80,64,3,padding=1),nn.SiLU(),
                                  nn.Conv2d(64,80,3,dilation=2,padding=2))
        self.tags=nn.Linear(len(TYPES),80)
        self.merge=nn.Sequential(nn.Conv2d(72,32,3,padding=1),nn.SiLU(),
                                 nn.Conv2d(32,64,3,padding=1),nn.PixelShuffle(2),nn.SiLU())
        self.output=nn.Sequential(nn.Conv2d(20,20,3,padding=1),nn.SiLU(),nn.Conv2d(20,4,1))
        nn.init.zeros_(self.output[-1].bias)
        nn.init.zeros_(self.output[-1].weight)

    def forward(self,src,tags,hard=False):
        n=len(src)
        rgb=src.reshape(-1,3,96,96)
        mask=(rgb.amin(1,keepdim=True)<.91).float()
        inputs=torch.cat([rgb,mask],1)
        skip=self.encode(inputs)
        small=self.small(skip).reshape(n,80,24,24)
        core=(small+self.paired(small)+self.tags(tags)[:,:,None,None]).reshape(-1,40,24,24)
        merged=self.merge(torch.cat([skip,F.interpolate(core,scale_factor=2,mode="nearest")],1))
        pred=self.output(torch.cat([inputs,merged],1))
        nearby=F.avg_pool2d(mask,7,1,3)
        fill=F.avg_pool2d(rgb*mask,7,1,3)/nearby.clamp_min(.01)
        foundation=torch.where(mask.bool(),rgb,fill)
        color=(foundation+.85*pred[:,:3].tanh()).clamp(-1,1)
        logits=3.2*(nearby-.45)+1.2*(mask-.5)+pred[:,3:]
        alpha=(logits>0).float() if hard else logits.sigmoid()
        return (1+alpha*(color-1)).reshape_as(src),logits


class TrazoGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=FichaGenerator(detailed=True,all_bodies=True)
        self.painter=OutlinePainter()

    def refine(self,draft,tags):
        painted,logits=self.painter(draft,tags,hard=True)
        old=draft.reshape(-1,3,96,96)
        new=painted.reshape_as(old)
        overlap=(old.amin(1,keepdim=True)<.91)&(logits>0)
        return torch.where(overlap,torch.lerp(old,new,.42),new).reshape_as(draft)

    def forward(self,z,tags):
        return self.refine(self.base(z,tags),tags)
