"""Learn paired spatial distributions from TRAIN and adversarially refine their decoder."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import Discriminator
from prepare import TYPES
from sample_cycles import load_generator
from sprite_mosaico import GROUPS, ORIGINS, PRINTS, MosaicGenerator
from train_estructura import dataset
from train_sprite_latent import codec_loss


def centers(values, species, groups, iterations=16):
    # Balance alternate versions of a species during codebook fitting.
    weight = torch.bincount(species)[species].float().rsqrt()
    mean = values[torch.multinomial(weight, groups, replacement=False)]
    for _ in range(iterations):
        dist = torch.cdist(values, mean)
        choice = dist.argmin(1)
        mass = torch.bincount(choice, weights=weight, minlength=groups)
        new = torch.zeros_like(mean).index_add_(0, choice, values*weight[:,None])
        mean = torch.where(mass[:,None]>0,new/mass[:,None].clamp_min(.01),mean)
    # A prototype blends the closest examples from different species. No local
    # code vector in the export is taken directly from one training example.
    dist = torch.cdist(mean, values)
    neighbours = []
    for _ in range(4):
        idx = dist.argmin(1)
        neighbours.append(idx)
        dist.masked_fill_(species[None,:]==species[idx,None],float("inf"))
    neighbours = torch.stack(neighbours,1)
    blend = mean.new_tensor([.48, .24, .17, .11])
    mean = (values[neighbours]*blend[None,:,None]).sum(1)
    assigned = torch.cdist(values,mean).argmin(1)
    return mean, assigned, neighbours


@torch.no_grad()
def fit(model, codes, tags, species, palettes):
    model.mean.copy_(codes.mean((0,2,3),keepdim=True))
    model.scale.copy_(codes.std((0,2,3),keepdim=True).clamp_min(.10))
    field = (codes-model.mean)/model.scale
    global_field = model.descriptor(field)
    mean, group, neighbours = centers(global_field,species,GROUPS)
    model.layouts.copy_(mean)
    delta = global_field[neighbours]-mean[:,None]
    model.basis[:,:4].copy_(delta*.48)
    model.palettes.copy_((palettes[neighbours]*mean.new_tensor([.52,.24,.15,.09])[None,:,None,None]).sum(1))
    weights = 1 / torch.bincount(species)[species]
    onehot = F.one_hot(group,GROUPS).float()
    frequency = onehot.T @ (weights[:,None]*tags)
    total = onehot.T @ weights
    base = (total+.2)/(total.sum()+.2*GROUPS)
    for i in range(len(TYPES)):
        for j in range(i+1):
            if i == j:
                mass = frequency[:,i]+1.5*base
            else:
                exact = onehot.T @ (weights*tags[:,i]*tags[:,j])
                fallback = ((frequency[:,i]+base)*(frequency[:,j]+base)/(total+.1)).sqrt()
                mass = exact + 2.5*fallback/fallback.sum().clamp_min(.001)
            model.frequency[i,j] = model.frequency[j,i] = mass/mass.sum()
    for q,(y,x) in enumerate(ORIGINS):
        patches = field[:,:,y:y+5,x:x+5].flatten(1)
        local, ids, near = centers(patches,species,PRINTS)
        model.patches[q].copy_(local.reshape_as(model.patches[q]))
        model.local_layout[q].copy_(global_field[near].mean(1))
        joint = torch.bincount(group*PRINTS+ids,weights=weights,minlength=GROUPS*PRINTS).reshape(GROUPS,PRINTS)
        soft = (-torch.cdist(mean,model.local_layout[q]).square()/global_field.size(1)*4).softmax(1)
        joint += .75*soft
        model.given_layout[q].copy_(joint/joint.sum(1,keepdim=True))
    return field


def save(obj, dest):
    tmp = dest.with_suffix(".tmp")
    torch.save(obj,tmp)
    tmp.replace(dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--codec",type=Path,default=Path("runs/latente_pareado/step_006000.pt"))
    args=p.parse_args()
    out=Path("runs/mosaico")
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(73731)
    data=dataset()
    tags=data["y"].cuda()
    species=torch.tensor(np.unique(data["species"],return_inverse=True)[1],device="cuda")
    teacher=load_generator(args.codec,"cuda")
    net=MosaicGenerator().cuda()
    net.codec.load_state_dict(teacher.codec.state_dict())
    with torch.no_grad():
        codes=torch.cat([teacher.codec.encode(x.cuda().float()/127.5-1) for x in data["images"].split(64)])
    checkpoint=out/"checkpoint.pt"
    disc=Discriminator().cuda()
    params=list(net.codec.decode_detail.parameters())
    opt=torch.optim.AdamW(params,lr=.00015)
    dop=torch.optim.Adam(disc.parameters(),lr=.00012,betas=(0.,.95))
    start=0
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if state["data"]!=data["data"]: raise ValueError("Cambió la partición")
        net.load_state_dict(state["generator"])
        disc.load_state_dict(state["disc"])
        opt.load_state_dict(state["opt"])
        dop.load_state_dict(state["dop"])
        start=state["step"]
        field=(codes-net.mean)/net.scale
    else:
        from train_boceto import quantize
        # Recompute the shared eight-colour targets using only this split.
        _,pal=quantize(data["images"])
        field=fit(net,codes,tags,species,pal.cuda())
    def checkpoint_at(step):
        inference=dict(architecture="paired_mosaic",generator=net.state_dict(),step=step,data=data["data"])
        save(dict(**inference,disc=disc.state_dict(),opt=opt.state_dict(),dop=dop.state_dict()),checkpoint)
        save(inference,out/f"modelo_{step:05}.pth")
    if start == 0:
        checkpoint_at(0)
    weight=torch.bincount(species)[species].float().rsqrt()
    for step in range(start+1,args.steps+1):
        torch.manual_seed(799000+step)
        ix=torch.multinomial(weight,20,replacement=True)
        real=data["images"][ix.cpu()].cuda().float()/127.5-1
        y=tags[ix]
        with torch.no_grad():
            rough=net.compose(net.descriptor(field[ix]), None, target=field[ix])*net.scale+net.mean
            alpha=torch.rand(len(ix),1,1,1,device="cuda")*.75
            degraded=torch.lerp(codes[ix],rough,alpha).clamp(-1,1)
            sampled,_=net.sample_codes(torch.randn(len(ix),96,device="cuda"),y)
        recon,logits=net.codec.decode(degraded)
        fake,_=net.codec.decode(sampled)
        disc.requires_grad_(True)
        good,bad=disc(torch.cat([real.flatten(1,2),fake.detach()]),y.repeat(2,1)).chunk(2)
        dis_loss=F.relu(1-good).mean()+F.relu(1+bad).mean()
        dop.zero_grad(set_to_none=True)
        dis_loss.backward()
        dop.step()
        disc.requires_grad_(False)
        loss=codec_loss(real,recon,logits)-.043*disc(fake,y).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params,3.)
        opt.step()
        if step%250==0 or step==args.steps:
            checkpoint_at(step)
            print(step,round(loss.item(),4),flush=True)


if __name__=="__main__":
    main()
