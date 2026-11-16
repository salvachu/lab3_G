"""Disposición conjunta de frente y espalda aprendida exclusivamente de TRAIN."""
import hashlib

import torch
from torch import nn

from prepare import TYPES
from sprite_tokens import GRID,LENGTH,TokenGenerator,sequence


class PlanGenerator(TokenGenerator):
    def __init__(self, groups, temperature=0.9):
        super().__init__(temperature)
        self.register_buffer("means",torch.zeros(groups,2,GRID,GRID))
        self.register_buffer("occupancy",torch.zeros_like(self.means))
        self.plan_distribution=nn.Linear(len(TYPES),groups)
        self.layout_features=nn.Sequential(nn.Conv2d(2,48,3,padding=1),nn.SiLU(),nn.Conv2d(48,128,3,padding=1))
        self.plan_embedding=nn.Embedding(groups,128)

    def context(self,groups):
        local=self.layout_features(self.means[groups]).flatten(2).transpose(1,2)
        layout=local.repeat_interleave(2,dim=1)+self.plan_embedding(groups)[:,None]
        prob=sequence(self.occupancy[groups]).clamp(0.02,0.98)
        return layout,(1-prob).log()-prob.log()

    def sample_groups(self,z,y):
        draws=[]
        for row in z.detach().cpu().contiguous().numpy():
            digest=hashlib.sha256(b"layout"+row.tobytes()).digest()
            draws.append(int.from_bytes(digest[:6],"little")/2**48)
        prob=self.plan_distribution(y).softmax(-1)
        return (prob.cumsum(-1)<z.new_tensor(draws)[:,None]).sum(1).clamp_max(len(self.means)-1)

    def pair_ids(self,z,y):
        groups=self.sample_groups(z,y)
        layout,bias=self.context(groups)
        return self.sample_ids(z,y,layout,bias)

    def forward(self,z,y):
        return self.tokenizer.decode(self.pair_ids(z,y),y)[0]


@torch.no_grad()
def fit_plans(masks,tokens):
    x=masks.flatten(1)
    centers=x[torch.randperm(len(x),device=x.device)[:96]]
    for iteration in range(24):
        groups=torch.cdist(x,centers).argmin(1)
        counts=torch.bincount(groups,minlength=len(centers))
        new=torch.zeros_like(centers).index_add_(0,groups,x)/counts.clamp_min(1)[:,None]
        centers=torch.where((counts>0)[:,None],new,centers)
        if iteration in (15,20):
            centers=centers[counts>=24]
    groups=torch.cdist(x,centers).argmin(1)
    counts=torch.bincount(groups,minlength=len(centers))
    means=torch.zeros_like(centers).index_add_(0,groups,x)/counts[:,None]
    active=(tokens!=0).float().reshape(-1,GRID*GRID,2).transpose(1,2).reshape(len(x),-1)
    occupancy=torch.zeros_like(means).index_add_(0,groups,active)/counts[:,None]
    return groups,means.reshape(-1,2,GRID,GRID),occupancy.reshape(-1,2,GRID,GRID),counts
