"""Train the paired hierarchical GAN from random initialization on the course split."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import Discriminator
from prepare import TYPES
from sprite_jerarquica import HierarchicalGenerator, PosteriorEncoder
from train_boceto import quantize
from train_estructura import dataset


class Judge(Discriminator):
    def __init__(self):
        super().__init__()
        self.types=nn.Linear(128,len(TYPES))

    def forward(self,x,y):
        h=self.blocks(x).mean((2,3))*3
        score=self.score(h).flatten()+(h*self.condition(F.normalize(y,dim=1))).sum(1)
        return score,self.types(h)


def save(obj,path):
    tmp=path.with_suffix(".tmp")
    torch.save(obj,tmp)
    tmp.replace(path)


def shape_neighbours(images,palettes,tags,species):
    feats=[]
    for chunk in images.split(128):
        mask=(chunk.amin(2)<242).float()
        feats.append(F.avg_pool2d(mask,8).flatten(1))
    feature=torch.cat((torch.cat(feats),.5*palettes.flatten(1)),1)
    result=[]
    for start in range(0,len(tags),256):
        block=slice(start,start+256)
        d=torch.cdist(feature[block],feature).square()/feature.size(1)
        shared=tags[block]@tags.T
        required=tags[block].sum(1,keepdim=True)
        d+=.15*(required-shared)
        d.masked_fill_(species[block,None]==species[None,:],float("inf"))
        result.append(d.topk(12,largest=False).indices)
    return torch.cat(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,required=True)
    args=parser.parse_args()
    out=Path("runs/jerarquica")
    out.mkdir(exist_ok=True,parents=True)
    torch.manual_seed(11071)
    data=dataset()
    ids,pal=quantize(data["images"])
    ids,pal=ids.cuda(),pal.cuda()
    real=data["images"].cuda()
    types=data["y"].cuda()
    _,sp=np.unique(data["species"],return_inverse=True)
    sp=torch.tensor(sp,device="cuda")
    weight=torch.bincount(sp)[sp].float().pow(-.65)
    freq=(weight[:,None]*types).sum(0)
    weight*=((types@freq.rsqrt())/types.sum(1)).sqrt()
    net=HierarchicalGenerator().cuda()
    ema=copy.deepcopy(net).requires_grad_(False).eval()
    enc=PosteriorEncoder().cuda()
    judge=Judge().cuda()
    params=[*net.parameters(),*enc.parameters()]
    opt=torch.optim.AdamW(params,lr=.00065,betas=(.9,.98))
    dop=torch.optim.Adam(judge.parameters(),lr=.00018,betas=(0.,.95))
    start=0
    ck=out/"checkpoint.pt"
    if ck.exists():
        state=torch.load(ck,map_location="cpu",weights_only=True)
        if state["data"]!=data["data"]: raise ValueError("Cambió la partición")
        for module,key in ((net,"net"),(ema,"generator"),(enc,"encoder"),(judge,"disc"),(opt,"opt"),(dop,"dop")):
            module.load_state_dict(state[key])
        start=state["step"]
    neighbours=shape_neighbours(real,pal,types,sp) if args.steps>9000 else None
    for step in range(start+1,args.steps+1):
        torch.manual_seed(192000+step)
        n=20
        ix=torch.multinomial(weight,n,replacement=True)
        x=real[ix].float()/127.5-1
        y=types[ix]
        target=ids[ix].long()
        palette=pal[ix]
        if step%2:
            x=x.flip(-1); target=target.flip(-1)
        def noise():
            return [torch.randn(n,k,s,s,device="cuda") for k,s in zip(net.latent,net.sizes)]
        fine=step>9000
        eps=[] if fine else None
        logits,colors,kl=net.decode(y,noise(),enc(x),encoded=eps)
        foreground=target>0
        edge=foreground&(target!=target.roll(1,-1))
        weight_fg=foreground.float()*(1+.6*edge)
        if fine:
            # Give small high-contrast clusters a voice inside the silhouette.
            light=(x.mean(2)+1)*.5
            local_range=F.max_pool2d(light,5,1,2)+F.max_pool2d(-light,5,1,2)
            weight_fg=foreground*(1.+2.4*local_range+1.1*edge)
        pigment=F.cross_entropy(logits[:,:,1:].flatten(0,1),(target-1).clamp_min(0).flatten(0,1),
                                reduction="none").reshape_as(target)
        occ=logits[:,:,0]
        soft=occ.sigmoid()
        overlap=1-(2*(soft*foreground).sum((2,3))+1)/(soft.sum((2,3))+foreground.sum((2,3))+1)
        palette_loss=F.smooth_l1_loss(colors,palette,beta=.12)
        loss=(pigment*weight_fg).sum()/weight_fg.sum()+F.binary_cross_entropy_with_logits(occ,foreground.float())
        loss+=.7*overlap.mean()+1.5*palette_loss
        beta=.20+.80*min(1.,step/1300)
        kl_weights=(.55,.68,.74) if fine else (.75,1.5,2.8)
        loss+=beta*sum(w*k.mean()/4096 for k,w in zip(kl,kl_weights))
        if step>=180 and step%2==0:
            draw=noise()
            if fine and step%4==0:
                # Match the inference distribution during adversarial training.
                # Both examples come from TRAIN and have different species IDs.
                with torch.no_grad():
                    other=neighbours[ix,torch.randint(12,(n,),device="cuda")]
                    other_x=real[other].float()/127.5-1
                    other_fields=[]
                    net.decode(types[other],[torch.zeros_like(v) for v in draw],enc(other_x),encoded=other_fields)
                    ratio=.18+.25*torch.rand(n,1,device="cuda")
                    draw=[torch.lerp(a,b,ratio).reshape_as(v) for a,b,v in zip(eps,other_fields,draw)]
            free,inks,_=net.decode(y,draw)
            sample=net.rgb(free,inks,straight=True)
            gt=F.pad(palette,(0,0,1,0),value=1)[torch.arange(n,device="cuda")[:,None,None,None],target]
            gt=gt.permute(0,1,4,2,3).flatten(1,2)
            judge.requires_grad_(True)
            scores,classes=judge(torch.cat([gt,sample.detach()]),y.repeat(2,1))
            pos,neg=scores.chunk(2)
            type_loss=lambda a:F.binary_cross_entropy_with_logits(a,y,pos_weight=y.new_full((len(TYPES),),5))
            dl=F.relu(1-pos).mean()+F.relu(1+neg).mean()+.65*type_loss(classes[:n])
            dop.zero_grad(set_to_none=True)
            dl.backward()
            dop.step()
            judge.requires_grad_(False)
            score,pred=judge(sample,y)
            loss+=-.065*score.mean()+.27*type_loss(pred)
        opt.param_groups[0]["lr"]=.00065/(1+step/8000)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),max(.008,1/step))
        if step%500==0 or step==args.steps:
            inference=dict(architecture="paired_hierarchical",generator=ema.state_dict(),step=step,data=data["data"])
            save(inference,out/f"modelo_{step:05}.pth")
            save(dict(**inference,net=net.state_dict(),encoder=enc.state_dict(),disc=judge.state_dict(),
                      opt=opt.state_dict(),dop=dop.state_dict()),ck)
            print(step,round(loss.item(),4),flush=True)


if __name__=="__main__":
    main()
