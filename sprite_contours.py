"""Un contorno conjunto, muestreado con el z de Plan, guía la ocupación de ambas vistas."""
import torch
from torch import nn
from torch.nn import functional as F

from prepare import TYPES
from sprite_plans import PlanGenerator
from sprite_tokens import sequence


def layer(inputs,outputs,stride=1):
    return nn.Sequential(nn.Conv2d(inputs,outputs,3,stride,1),nn.GroupNorm(8,outputs),nn.SiLU())


class PairedContour(nn.Module):
    latent=32

    def __init__(self,groups):
        super().__init__()
        self.group_embedding=nn.Embedding(groups,24)
        self.type_embedding=nn.Linear(len(TYPES),24,bias=False)
        self.encoder=nn.Sequential(layer(4,32,2),layer(32,48,2),layer(48,64,2),nn.Flatten())
        self.posterior=nn.Linear(64*6*6+24,self.latent*2)
        self.start=nn.Linear(self.latent+24,64*6*6)
        self.stages=nn.ModuleList([layer(66,64),layer(66,48),layer(50,32)])
        self.out=nn.Sequential(layer(32,32),nn.Conv2d(32,2,3,padding=1))

    def condition(self,groups,y):
        return self.group_embedding(groups)+self.type_embedding(y)

    def encode(self,target,mean,cond):
        coarse=F.interpolate(mean,size=target.shape[-2:],mode="bilinear",align_corners=False)
        features=self.encoder(torch.cat([target,coarse],1))
        mu,logvar=self.posterior(torch.cat([features,cond],1)).chunk(2,1)
        return mu,logvar.clamp(-7,3)

    def decode(self,z,mean,cond):
        x=self.start(torch.cat([z,cond],1)).reshape(-1,64,6,6)
        for stage in self.stages:
            x=F.interpolate(x,scale_factor=2,mode="nearest")
            coarse=F.interpolate(mean,size=x.shape[-2:],mode="bilinear",align_corners=False)
            x=stage(torch.cat([x,coarse],1))
        return self.out(x)+torch.logit(coarse.clamp(0.05,0.95))


class ContourGenerator(PlanGenerator):
    def __init__(self,groups,temperature=0.9):
        super().__init__(groups,temperature)
        self.contour=PairedContour(groups)

    @torch.inference_mode()
    def render_components(self,z,y):
        groups=self.sample_groups(z,y)
        logits=self.contour.decode(z[:,:self.contour.latent],self.means[groups],self.contour.condition(groups,y))
        mask=logits.sigmoid()
        pool=sequence(F.max_pool2d(mask,4)).clamp(0.002,0.998)
        layout,bias=self.context(groups)
        guided=0.25*bias-2.8*torch.logit(pool)
        ids=self.sample_ids(z,y,layout,guided)
        pair=self.tokenizer.decode(ids,y)[0].reshape(-1,2,3,96,96)
        # Un margen de un píxel permite conservar trazos justo fuera de la guía.
        alpha=F.interpolate(logits,scale_factor=2,mode="bilinear",align_corners=False)
        alpha=(2*F.max_pool2d(alpha,3,1,1)).sigmoid()
        return pair,alpha

    def forward(self,z,y):
        pair,alpha=self.render_components(z,y)
        return (1+alpha[:,:,None]*(pair-1)).flatten(1,2)
