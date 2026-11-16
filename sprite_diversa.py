"""Sample paired proportions, anatomical family and markings from types and noise."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from prepare import TYPES
from sprite_ficha import FichaGenerator, ghost_pair
from sprite_profiles import SHAPE
from sprite_cuerpos import paired_bodies
from sprite_trazo import OutlinePainter, refine_pair

FAMILIES=8
PATCH=6
SIZE=FAMILIES+SHAPE+6+2*PATCH**2


class BodyPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean",torch.zeros(SIZE))
        self.register_buffer("std",torch.ones(SIZE))
        self.encoder=nn.Linear(SIZE+len(TYPES)+12,256)
        self.blocks=nn.ModuleList(nn.Sequential(nn.LayerNorm(256),nn.Linear(256,384),nn.SiLU(),nn.Linear(384,256))
                                  for _ in range(3))
        self.output=nn.Linear(256,SIZE)

    def forward(self,x,time,types):
        angle=time*torch.arange(1,7,device=x.device)*math.pi
        h=self.encoder(torch.cat([x,angle.sin(),angle.cos(),types],1))
        for block in self.blocks:
            h=h+.5*block(h)
        return self.output(h)

    def sample(self,z,types):
        # Expand the seed deterministically to independent normal coordinates.
        seeds=[int.from_bytes(hashlib.sha256(row.tobytes()).digest()[:8],"little")
               for row in z.detach().cpu().contiguous().numpy()]
        x=torch.stack([torch.randn(SIZE,generator=torch.Generator().manual_seed(s)) for s in seeds]).to(z)
        for step in range(24):
            t=x.new_full((len(x),1),step/24)
            velocity=self(x,t,types)
            midpoint=self(x+velocity/48,t+1/48,types)
            x=x+midpoint/24
        return x*self.std+self.mean


class DiverseGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.ficha=FichaGenerator(detailed=True)
        self.prior=BodyPrior()
        self.insect=FichaGenerator(detailed=True,all_bodies=True)
        self.painter=OutlinePainter()
        self.ficha.base.drawing.base.posed=True
        self.ficha.base.drawing.base.show_limbs=True

    def forward(self,z,tags):
        if len(z)>5:
            return torch.cat([self(a,b) for a,b in zip(z.split(5),tags.split(5))])
        sample=self.prior.sample(z,tags)
        family,outline,card,markings=sample.split([FAMILIES,SHAPE,6,2*PATCH**2],1)
        net=self.ficha
        body=net.base.drawing.base
        i,j=body.type_ids(tags)
        prior=body.topo[i,j].clamp_min(.004)
        global_rate=body.topo.diagonal(dim1=0,dim2=1).mean(1).clamp_min(.004)
        # Combine the conditional flow draw with the observed family odds by type.
        logits=1.55*prior.log()-.75*global_rate.log()+.9*family
        family=body.gumbel(logits.softmax(1),z[:,:8],1.)
        card[:,3:]=torch.lerp(card[:,3:],net.describe(z,tags)[:,3:],.58)
        card=card.clamp(card.new_tensor([.38,-.75,.12,.04,.04,.04]),card.new_tensor([2.4,.75,.98,.95,.95,.95]))
        bug=(tags[:,TYPES.index("bug")]>0)&(tags[:,TYPES.index("poison")]>0)
        ghost=(tags[:,TYPES.index("ghost")]>0)&(tags.sum(1)==1)
        spectral=ghost&(z[:,63]>.0)
        psychic_ghost=(tags[:,TYPES.index("ghost")]>0)&(tags[:,TYPES.index("psychic")]>0)
        sea_dragon=(tags[:,TYPES.index("water")]>0)&(tags[:,TYPES.index("dragon")]>0)
        # Retain the inked spectral bodies, and admit learned quadrupeds,
        # bipeds and shelled creatures on alternate draws of this dual type.
        other_family=(family==1)|(family==4)|(family==7)
        drawn=bug|(ghost&~spectral)|(psychic_ghost&~(other_family&(z[:,63]>-.75)))|(sea_dragon&(z[:,63]<0))
        classic=~(drawn|spectral)
        rgb=z.new_empty(len(z),6,96,96)
        def finish(draft,y,traits,model):
            corrected=model.finish(draft,y,traits)
            return torch.lerp(draft,corrected,.44)
        if classic.any():
            y,traits=tags[classic],card[classic]
            draft=net.base(z[classic],y,traits,family[classic],outline[classic])
            rgb[classic]=finish(draft,y,traits,net)
        for mask,use_bodies in ((drawn,True),(spectral,False)):
            if mask.any():
                a,y,traits=z[mask],tags[mask],card[mask]
                model=self.insect if use_bodies else net
                draft=paired_bodies(a,y,traits) if use_bodies else ghost_pair(a,traits,detailed=True)
                i,j=model.base.drawing.base.type_ids(y)
                draft=model.base.apply_profile(draft,a,i,j,traits,outline[mask])
                rgb[mask]=finish(draft,y,traits,model)
        if psychic_ghost.any():
            rgb[psychic_ghost]=refine_pair(self.painter,rgb[psychic_ghost],tags[psychic_ghost])
        views=rgb.reshape(-1,3,96,96)
        variation=F.interpolate(markings.reshape(-1,1,PATCH,PATCH),size=(96,96),mode="bilinear",align_corners=False)
        paint=(views.amin(1,keepdim=True)<.88)&(views.mean(1,keepdim=True)>-.55)
        ramp=(variation.clamp(-.48,.48)*5).round()/5
        return torch.where(paint,(views+.42*ramp).clamp(-1,1),views).reshape_as(rgb)
