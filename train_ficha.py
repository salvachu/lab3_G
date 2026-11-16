"""Fit TRAIN descriptions, then train a descriptor-conditioned adversarial colour refiner."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm

from model import Discriminator, LATENT
from prepare import TYPES
from sprite_ficha import FIELDS, GROUPS, FichaGenerator, measure
from train_estructura import dataset


@torch.no_grad()
def fit_descriptions(model, traits, labels, weights):
    # Equal species weights prevent a species with many game variants dominating a type.
    scale=traits.std(0).clamp_min(.08)
    x=traits/scale
    centers=x[torch.multinomial(weights,GROUPS,replacement=False)]
    for _ in range(28):
        group=torch.cdist(x,centers).argmin(1)
        mass=x.new_zeros(GROUPS).index_add_(0,group,weights)
        sums=torch.zeros_like(centers).index_add_(0,group,x*weights[:,None])
        centers=torch.where(mass[:,None]>0,sums/mass[:,None].clamp_min(.0001),centers)
    model.centers.copy_(centers*scale)
    var=torch.zeros_like(centers).index_add_(0,group,(traits-model.centers[group]).square()*weights[:,None])
    model.spread.copy_((var/mass[:,None].clamp_min(.0001)).sqrt())
    hist=F.one_hot(group,GROUPS)*weights[:,None]
    prior=mass/mass.sum()
    all_types=labels.T@hist + 2*prior
    single=all_types/all_types.sum(1,keepdim=True)
    for i in range(len(TYPES)):
        for j in range(i,len(TYPES)):
            exact=(labels[:,i]*labels[:,j])*(labels.sum(1)==(1 if i==j else 2))
            counts=exact@hist+1.5*(single[i]+single[j])
            model.freq[i,j]=model.freq[j,i]=counts/counts.sum()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,default=600)
    parser.add_argument("--figuras",action="store_true")
    parser.add_argument("--todos",action="store_true",help="Usar los planes corporales nuevos para todos los tipos")
    parser.add_argument("--desde",type=Path)
    parser.add_argument("--run",type=Path,default=Path("runs/ficha"))
    args=parser.parse_args()
    args.run.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(7267)
    data=dataset()
    y=data["y"].cuda()
    x=data["images"].cuda().flatten(1,2)
    traits=torch.cat([measure(b.float()/127.5-1) for b in x.split(64)])
    _,indices,counts=np.unique(data["species"],return_inverse=True,return_counts=True)
    weights=y.new_tensor(1/counts[indices])
    model=FichaGenerator(detailed=args.figuras or args.todos,all_bodies=args.todos).cuda()
    base=torch.load("runs/perfiles_pigmentos/modelo.pt",weights_only=True,map_location="cpu")
    if base["data"]!=data["data"]:
        raise ValueError("La base y las fichas deben usar la misma partición")
    model.base.load_state_dict(base["generator"])
    model.base.eval().requires_grad_(False)
    fit_descriptions(model,traits,y,weights)
    disc=Discriminator().cuda()
    disc.condition=spectral_norm(nn.Linear(len(TYPES)+len(FIELDS),128,bias=False)).cuda()
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=.0006,betas=(.5,.99))
    od=torch.optim.Adam(disc.parameters(),lr=.0002,betas=(.0,.99))
    latest=sorted(args.run.glob("step_*.pt"))
    start=0
    if latest or args.desde:
        state=torch.load(latest[-1] if latest else args.desde,map_location="cpu",weights_only=True)
        if latest:
            model.detailed=state.get("detailed",False)
            model.all_bodies=state.get("all_bodies",False)
        if state["data"]!=data["data"]:
            raise ValueError("Cambió la partición")
        for obj,key in ((model,"generator"),(disc,"discriminator"),(opt,"opt"),(od,"od")):
            obj.load_state_dict(state[key])
        start=state["step"]
    if args.steps<=start:
        return
    with torch.no_grad():
        rng=torch.Generator(device="cuda").manual_seed(16691)
        idx=torch.multinomial(weights,384,replacement=True,generator=rng)
        cond=y[idx].clone()
        if model.all_bodies:
            cond[:len(TYPES)]=torch.eye(len(TYPES),device=y.device)
        else:
            cond[::5]=0
            cond[::5,TYPES.index("ghost")]=1
        z=torch.randn(len(cond),LATENT,device="cuda",generator=rng)
        sampled=model.describe(z,cond)
        drafts=torch.cat([model.draft(a,b,c) for a,b,c in zip(z.split(4),cond.split(4),sampled.split(4))])
        drafts=F.interpolate(drafts,size=(64,64),mode="nearest")
    for step in range(start+1,args.steps+1):
        rng=torch.Generator(device="cuda").manual_seed(91001+step)
        ri=torch.multinomial(weights,16,replacement=True,generator=rng)
        fi=torch.randint(len(cond),(16,),device="cuda",generator=rng)
        real=x[ri].float()/127.5-1
        # Remove shading detail, then restore real marked pixels under their own descriptors.
        blurred=F.avg_pool2d(real,3,1,1,count_include_pad=False)
        degraded=torch.where(real.amin(1,keepdim=True)<.96,.65*blurred+.35*real,real)
        recon=model.finish(degraded,y[ri],traits[ri])
        fake=model.finish(drafts[fi],cond[fi],sampled[fi])
        a=torch.cat([y[ri],traits[ri]],1)
        b=torch.cat([cond[fi],sampled[fi]],1)
        disc.requires_grad_(True)
        od.zero_grad(set_to_none=True)
        pos,neg=disc(torch.cat([real,fake.detach()]),torch.cat([a,b])).chunk(2)
        (F.relu(1-pos).mean()+F.relu(1+neg).mean()).backward()
        od.step()
        disc.requires_grad_(False)
        opt.zero_grad(set_to_none=True)
        figure=real.reshape(-1,3,64,64).amin(1,keepdim=True)<.95
        error=(recon-real).abs().reshape(-1,3,64,64)
        loss=(error*figure).sum()/(3*figure.sum()).clamp_min(1)-.008*disc(fake,b).mean()
        loss.backward()
        opt.step()
        if step%200==0 or step==args.steps:
            path=args.run/f"step_{step:06}.pt"
            tmp=path.with_suffix(".tmp")
            torch.save(dict(architecture="paired_ficha",generator=model.state_dict(),data=data["data"],
                            discriminator=disc.state_dict(),opt=opt.state_dict(),od=od.state_dict(),step=step,
                            campos=FIELDS,pares=len(x),especies=len(counts),detailed=model.detailed,
                            all_bodies=model.all_bodies),tmp)
            tmp.replace(path)
            print(step,float(loss.detach()),flush=True)


if __name__=="__main__":
    main()
