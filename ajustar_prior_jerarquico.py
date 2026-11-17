"""Fit a type-conditional mixture on TRAIN posterior fields, pooling distinct species."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from prepare import TYPES
from sprite_jerarquica import HierarchicalGenerator, PosteriorEncoder
from train_estructura import dataset
from train_jerarquica import save


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,default=Path("runs/jerarquica/checkpoint.pt"))
    p.add_argument("--output",type=Path,default=Path("pesos/jerarquica.pth"))
    p.add_argument("--metodo",choices=("gaussian","convex"),default="convex")
    args=p.parse_args()
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    data=dataset()
    if state["data"]!=data["data"]: raise ValueError("Cambió la partición")
    groups=256
    model=HierarchicalGenerator(groups).cuda().eval()
    model.load_state_dict(dict(model.state_dict(),**state["net"]))
    enc=PosteriorEncoder().cuda().eval()
    enc.load_state_dict(state["encoder"])
    fields=[]; features=[]
    y=data["y"].cuda()
    for i,chunk in enumerate(data["images"].split(64)):
        x=chunk.cuda().float()/127.5-1
        tags=y[i*64:i*64+len(x)]
        zeros=[x.new_zeros(len(x),k,s,s) for s,k in zip(model.sizes,model.latent)]
        emb=[]
        model.decode(tags,zeros,enc(x),encoded=emb)
        fields.append(torch.cat(emb,1))
        masks=(x.amin(2)<.92).float()
        feature=[2*F.avg_pool2d(masks,8).flatten(1)]
        for flat,level in zip(emb,zeros):
            block=F.adaptive_avg_pool2d(flat.reshape_as(level),2).flatten(1)
            feature.append(.8*block)
        color=((x*masks[:,:,None]).sum((1,3,4))/masks.sum((1,2,3)).clamp_min(1)[:,None])
        features.append(torch.cat([*feature,2*color],1))
    fields=torch.cat(fields)
    features=torch.cat(features)
    features=(features-features.mean(0))/features.std(0).clamp_min(.16)
    # Allocate greater weight to the complete outlines and their gross colours.
    features[:,:128]*=1.35
    features[:,-3:]*=3
    torch.manual_seed(96142)
    species=torch.tensor(np.unique(data["species"],return_inverse=True)[1],device="cuda")
    weight=1./torch.bincount(species)[species]
    centres=features[torch.multinomial(weight,groups,replacement=False)]
    for _ in range(22):
        dist=torch.cdist(features,centres)
        idx=dist.argmin(1)
        mass=torch.bincount(idx,weights=weight,minlength=groups)
        avg=torch.zeros_like(centres).index_add_(0,idx,features*weight[:,None])/mass[:,None].clamp_min(.001)
        centres=torch.where(mass[:,None]>0,avg,centres)
    dist=torch.cdist(centres,features)
    neighbours=[]
    for _ in range(6):
        ix=dist.argmin(1)
        neighbours.append(ix)
        dist.masked_fill_(species[None]==species[ix,None],float("inf"))
    near=torch.stack(neighbours,1)
    weights=fields.new_tensor([.36,.23,.16,.11,.08,.06])
    values=fields[near]
    mu=(values*weights[None,:,None]).sum(1)
    model.centres.copy_(mu)
    if args.metodo=="convex":
        model.directions.copy_(.72*(values[:,:5]-mu[:,None]))
    else:
        residual=(values-mu[:,None])*weights.sqrt()[None,:,None]
        _,vec=torch.linalg.eigh(residual@residual.transpose(1,2))
        model.directions.copy_(vec[:,:,-5:].transpose(1,2)@residual)
    counts=F.one_hot(idx,groups).float()*weight[:,None]
    global_rate=counts.sum(0)+.05
    global_rate/=global_rate.sum()
    for i in range(len(TYPES)):
        for j in range(i+1):
            if i==j:
                score=(y[:,i]*(1+.8*(y.sum(1)==1)))@counts+global_rate
            else:
                exact=(y[:,i]*y[:,j])@counts
                common=((y[:,i]@counts+global_rate)*(y[:,j]@counts+global_rate)/global_rate).sqrt()
                score=exact+2.*common/common.sum()
            freq=(score/score.sum()).pow(1.6)/global_rate.pow(.6)
            model.frequencies[i,j]=model.frequencies[j,i]=freq/freq.sum()
    save(dict(architecture="paired_hierarchical",components=groups,prior_method=args.metodo,step=state["step"],
              data=data["data"],generator=model.state_dict()),args.output)
    print(args.output,state["step"],flush=True)


if __name__=="__main__":
    main()
