"""Fit paired proportions and representative pigments on the training split only."""
import argparse
from pathlib import Path

import numpy as np
import torch

from generate import render
from prepare import TYPES
from sprite_profiles import GROUPS, SHAPE, ProfileGenerator, describe, pigment
from train_estructura import dataset
from train_sprite_latent import CASES, SEEDS


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=Path("runs/detalle_1200/detalle_01200.pt"))
    parser.add_argument("--out",type=Path,default=Path("runs/perfiles_pareados/modelo.pt"))
    parser.add_argument("--pigmentos",action="store_true")
    args=parser.parse_args()
    args.out.parent.mkdir(exist_ok=True,parents=True)
    data=dataset()
    model=ProfileGenerator(learned_palette=args.pigmentos).cuda().eval()
    previous=torch.load(args.base,map_location="cpu",weights_only=True)
    model.drawing.load_state_dict(previous["generator"])
    features=[]
    for block in data["images"].split(64):
        rgb=block.cuda().float()/127.5-1
        shape=describe(rgb)[0].flatten(1)
        features.append(torch.cat([shape,pigment(rgb)],1))
    x=torch.cat(features)
    _,species,counts=np.unique(data["species"],return_inverse=True,return_counts=True)
    weights=torch.tensor(1./counts[species],device="cuda",dtype=x.dtype)
    # Use geometry for grouping; keep pigment coordinates for the conditional draw.
    rng=torch.Generator(device="cuda").manual_seed(6815)
    centers=x[torch.multinomial(weights,GROUPS,replacement=False,generator=rng)].clone()
    for _ in range(32):
        group=torch.cdist(x[:,:SHAPE],centers[:,:SHAPE]).argmin(1)
        mass=x.new_zeros(GROUPS).index_add_(0,group,weights)
        sums=torch.zeros_like(centers).index_add_(0,group,weights[:,None]*x)
        centers=torch.where(mass[:,None]>0,sums/mass.clamp_min(1e-7)[:,None],centers)
    model.centers.copy_(centers)
    mass=x.new_zeros(GROUPS).index_add_(0,group,weights)
    marginal=mass/mass.sum()
    y=data["y"].cuda()
    def distribution(w,prior,prior_color,strength):
        hist=x.new_zeros(GROUPS).index_add_(0,group,w)
        return ((hist+strength*prior)/(w.sum()+strength),
                ((x[:,SHAPE:]*w[:,None]).sum(0)+strength*prior_color)/(w.sum()+strength))
    global_color=(x[:,SHAPE:]*weights[:,None]).sum(0)/weights.sum()
    singles=[distribution(weights*y[:,i],marginal,global_color,3.) for i in range(len(TYPES))]
    if args.pigmentos:
        tones=model.drawing.base.tones[:,0]
        # Nearest TRAIN palette in chroma and brightness, with softer assignment
        # for samples falling midway between two learned pigments.
        def coordinates(rgb):
            return rgb-.72*rgb.mean(1,keepdim=True)
        membership=(-torch.cdist(coordinates(x[:,SHAPE:]),coordinates(tones)).square()/.012).softmax(1)
        pigments=(y.T*weights)@membership
        pigments=(pigments+.01)/(pigments.sum(1,keepdim=True)+.24)
    for i in range(len(TYPES)):
        for j in range(i,len(TYPES)):
            shared=y[:,i]*y[:,j]
            exact=shared*(y.sum(1)==(1 if i==j else 2))
            prior=.5*(singles[i][0]+singles[j][0])
            prior_color=.5*(singles[i][1]+singles[j][1])
            fr,col=distribution(weights*exact,prior,prior_color,4.)
            model.freq[i,j].copy_(fr)
            model.freq[j,i].copy_(fr)
            model.type_color[i,j].copy_(col)
            model.type_color[j,i].copy_(col)
            if args.pigmentos:
                counts_color=(weights*exact)@membership+2*(pigments[i]+pigments[j])
                model.pigment_freq[i,j]=model.pigment_freq[j,i]=counts_color/counts_color.sum()
    for k in range(GROUPS):
        assigned=group==k
        if assigned.sum()<2:
            continue
        delta=x[assigned]-centers[k]
        cov=delta.T@(delta*weights[assigned,None])/mass[k]
        values,vectors=torch.linalg.eigh(cov)
        model.basis[k].copy_(vectors[:,-5:].T*values[-5:].clamp_min(0).sqrt()[:,None])
    tmp=args.out.with_suffix(".tmp")
    torch.save(dict(architecture="paired_profiles",generator=model.state_dict(),data=data["data"],
                    base=str(args.base),base_step=previous["base_step"],detail_steps=previous["detail_steps"],
                    training_pairs=len(x),species=len(counts),clusters=GROUPS,learned_palette=args.pigmentos),tmp)
    tmp.replace(args.out)
    render(model,CASES,SEEDS,args.out.with_suffix(".png"))
    print(f"{len(x)} pares, {len(counts)} especies, {GROUPS} grupos",flush=True)


if __name__=="__main__":
    main()
