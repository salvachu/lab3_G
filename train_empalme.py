"""Fit disjoint paired part dictionaries and train the conditional seam painter."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import Discriminator
from prepare import TYPES
from sprite_empalme import COLORS, END, INK_GROUPS, SIZE, START, SpliceGenerator, outline_features
from train import train_rows
from train_boceto import quantize
from train_estructura import dataset


def representatives(rows):
    by_species={}
    for i,row in enumerate(rows):
        quality=(int(row["source_generation"]),row["is_shiny"]=="0",
                 row["form_name"]==row["species_name"],row["pair_id"])
        sp=row["species_id"]
        if sp not in by_species or by_species[sp][0]<quality:
            by_species[sp]=(quality,i)
    indices=[i for _,i in by_species.values()]
    np.random.default_rng(2231).shuffle(indices)
    return indices[::2],indices[1::2]


def save(state,path):
    tmp=path.with_suffix(".tmp")
    torch.save(state,tmp)
    tmp.replace(path)


@torch.no_grad()
def fit_inks(net,pal,tags,species):
    samples=pal[:,1:].flatten(1)
    weight=torch.bincount(species)[species].float().reciprocal()
    centre=samples[torch.multinomial(weight,INK_GROUPS,replacement=False)]
    for _ in range(22):
        group=torch.cdist(samples,centre).argmin(1)
        total=torch.bincount(group,weights=weight,minlength=INK_GROUPS)
        sums=torch.zeros_like(centre).index_add_(0,group,weight[:,None]*samples)
        centre=torch.where(total[:,None]>0,sums/total[:,None].clamp_min(.001),centre)
    net.ink.copy_(F.pad(centre.reshape(INK_GROUPS,8,3),(0,0,1,0),value=1.))
    counts=F.one_hot(group,INK_GROUPS).float()*weight[:,None]
    base=counts.sum(0)+.05
    base/=base.sum()
    frequencies=tags.T@counts
    for i in range(len(TYPES)):
        for j in range(i+1):
            if i==j:
                mass=frequencies[i]+2*base
            else:
                shared=(tags[:,i]*tags[:,j])@counts
                mix=((frequencies[i]+base)*(frequencies[j]+base)/base).sqrt()
                mass=shared+3*mix/mix.sum()
            odds=(mass/mass.sum()).pow(1.65)/base.pow(.65)
            net.ink_weights[i,j]=net.ink_weights[j,i]=odds/odds.sum()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=Path("runs/empalme_color"))
    parser.add_argument("--steps",type=int,default=900)
    args=parser.parse_args()
    output=args.run
    output.mkdir(exist_ok=True,parents=True)
    data=dataset()
    rows=train_rows()
    torch.manual_seed(7801)
    heads,bodies=representatives(rows)
    net=SpliceGenerator(len(heads),len(bodies)).cuda()
    ids,pal=quantize(data["images"])
    ids,pal=ids.cuda(),pal.cuda()
    pal=F.pad(pal,(0,0,1,0),value=1.)
    tags=data["y"].cuda()
    species=torch.tensor(np.unique(data["species"],return_inverse=True)[1],device="cuda")
    fit_inks(net,pal,tags,species)
    for role,idx,cut in (("head",heads,slice(0,END)),("body",bodies,slice(START,SIZE))):
        getattr(net,role+"_tiles").copy_(ids[idx,:,cut])
        getattr(net,role+"_types").copy_(tags[idx])
        getattr(net,role+"_shape").copy_(outline_features(ids[idx,:,START:END]!=0))
        getattr(net,role+"_colors").copy_(pal[idx])
        assert tags[idx].sum(0).min()>0
    ids_manifest={k:[rows[i]["pair_id"] for i in v] for k,v in (("head",heads),("body",bodies))}
    (output/"fuentes_train.json").write_text(json.dumps(ids_manifest),encoding="utf8")
    disc=Discriminator().cuda()
    opt=torch.optim.AdamW(net.painter.parameters(),lr=.00065)
    dop=torch.optim.Adam(disc.parameters(),lr=.00015,betas=(0.,.99))
    state_file=output/"checkpoint.pt"
    start=0
    if state_file.exists():
        state=torch.load(state_file,map_location="cpu",weights_only=True)
        if state["data"]!=data["data"]: raise ValueError("Cambió la partición")
        weights=state["generator"]
        for field in ("ink","ink_weights"):
            weights.setdefault(field,getattr(net,field))
        net.load_state_dict(weights)
        disc.load_state_dict(state["discriminator"])
        opt.load_state_dict(state["opt"])
        dop.load_state_dict(state["dop"])
        start=state["step"]
    def checkpoint(step):
        state=dict(architecture="paired_splice",generator=net.state_dict(),data=data["data"],step=step,
                   heads=len(heads),bodies=len(bodies))
        save(state,output/f"modelo_{step:05}.pth")
        save(dict(**state,discriminator=disc.state_dict(),opt=opt.state_dict(),dop=dop.state_dict()),state_file)
    if start==0:
        checkpoint(0)
    # Cache synthetic drafts. The adversary compares freely generated pairs
    # with real pairs under the same type labels. Training never sees VAL.
    tiles=[];bands=[];colors=[];conditions=[]
    with torch.no_grad():
        for _ in range(48):
            ii=torch.randint(len(tags),(32,),device="cuda")
            y=tags[ii]
            tile,band,col=net.draft(torch.randn(len(y),96,device="cuda"),y)
            tiles.append(tile.byte());bands.append(band.half());colors.append(col);conditions.append(y)
    tiles,bands,colors,conditions=map(torch.cat,(tiles,bands,colors,conditions))
    for step in range(start+1,args.steps+1):
        torch.manual_seed(621000+step)
        batch=20
        sample=torch.randint(len(tiles),(batch,),device="cuda")
        fake_y=conditions[sample]
        fake_logits=net.painter(tiles[sample],bands[sample].float(),fake_y)
        probs=(fake_logits/.7).softmax(2)
        hard=F.one_hot(probs.argmax(2),COLORS).permute(0,1,4,2,3).float()
        straight=probs+(hard-probs).detach()
        fake=torch.einsum("nvkhw,nkc->nvchw",straight,colors[sample]).flatten(1,2)
        # Match the real batch to each fake condition to prevent the critic
        # from solving the task from type frequencies.
        matches=(tags[None]==fake_y[:,None]).all(2)
        ix=torch.multinomial(matches.float(),1).squeeze(1)
        target=ids[ix].long()
        real=net.colorize(target,pal[ix])
        y=tags[ix]
        disc.requires_grad_(True)
        pos,neg=disc(torch.cat((real,fake.detach())),torch.cat((y,fake_y))).chunk(2)
        dloss=F.relu(1-pos).mean()+F.relu(1+neg).mean()
        dop.zero_grad(set_to_none=True); dloss.backward(); dop.step()
        disc.requires_grad_(False)
        axis=torch.arange(SIZE,device="cuda").reshape(1,1,SIZE,1)
        mid=torch.randint(25,39,(batch,1,1,1),device="cuda")
        band=(1-(axis-mid).abs()/6).clamp_min(0).expand_as(target)
        offset=torch.randint(-3,4,(batch,2,1,1),device="cuda")
        xx=torch.arange(SIZE,device="cuda").reshape(1,1,1,SIZE).expand_as(target)
        moved=target.gather(3,(xx+offset).clamp(0,SIZE-1))
        patch=torch.rand_like(band)<(.72*band)
        impaired=torch.where(patch,moved,target)
        rec=net.painter(impaired,band,y)
        errors=F.cross_entropy(rec.flatten(0,1),target.flatten(0,1),reduction="none").reshape_as(band)
        weight=band*(1+3*(target>0))
        loss=(errors*weight).sum()/weight.sum()-.032*disc(fake,fake_y).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.painter.parameters(),3.)
        opt.step()
        if step%300==0 or step==args.steps:
            checkpoint(step)
            print(step,round(loss.item(),3),flush=True)


if __name__=="__main__":
    main()
