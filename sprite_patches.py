"""Refinador local de cuatro pasadas condicionado por el dibujo libre del VQGAN."""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from sprite_plans import PlanGenerator

SIDE=24
K=512


def tiles(pairs):
    return F.pixel_unshuffle(pairs.reshape(-1,3,96,96),4)


def tile_dist(x,table):
    return F.conv2d(x,(-2/48)*table[:,:,None,None],table.square().mean(1))+x.square().mean(1,keepdim=True)


class PatchGenerator(PlanGenerator):
    def __init__(self,groups,temperature=0.9):
        super().__init__(groups,temperature)
        self.register_buffer("table",torch.zeros(K,48))
        self.context_embedding=nn.Embedding(K+1,24)
        self.patch_net=nn.Sequential(nn.Conv2d(48+24,80,3,padding=1),nn.GroupNorm(8,80),nn.SiLU(),
                                     nn.Conv2d(80,80,3,padding=1),nn.GroupNorm(8,80),nn.SiLU(),
                                     nn.Conv2d(80,K,1))
        coord=torch.arange(SIDE)
        self.register_buffer("order",2*(coord[:,None]%2)+coord[None,:]%2)

    def predict(self,coarse,known):
        c=self.context_embedding(known).permute(0,3,1,2)
        return self.patch_net(torch.cat([coarse,c],1))-tile_dist(coarse,self.table)/0.11

    @torch.inference_mode()
    def forward(self,z,y):
        return self.refine(self.pair_ids(z,y),z,y)

    @torch.inference_mode()
    def refine(self,ids,z,y):
        coarse=tiles(self.tokenizer.decode(ids,y)[0])
        active=F.interpolate((ids.reshape(-1,1,12,12)!=0).float(),scale_factor=2)[:,0].bool()
        known=torch.full_like(active,K,dtype=torch.long)
        known[~active]=0
        draws=[]
        for row in z.detach().cpu().contiguous().numpy():
            seed=int.from_bytes(hashlib.sha256(b"patch"+row.tobytes()).digest()[:8],"little")
            draws.append(torch.rand(2,SIDE,SIDE,generator=torch.Generator().manual_seed(seed)))
        uniform=torch.stack(draws).to(z.device).reshape_as(active)
        for stage in range(4):
            loc=(self.order==stage)&active
            logits=self.predict(coarse,known).permute(0,2,3,1)[loc]
            values,candidates=logits.topk(8)
            probs=(values/0.75).softmax(-1)
            rank=(probs.cumsum(1)<uniform[loc][:,None]).sum(1,keepdim=True).clamp_max(7)
            known[loc]=candidates.gather(1,rank).flatten()
        detail=self.table[known].permute(0,3,1,2)
        return F.pixel_shuffle(detail,4).reshape(-1,6,96,96)
