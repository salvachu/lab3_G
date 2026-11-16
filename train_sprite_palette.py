import csv
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from model import Discriminator
from prepare import ROOT
from sprite_palette import PaletteGenerator
from train import batch,train_rows
from train_silhouettes import hinge,update


def bins(views):
    rgb=((views+1)*15.5).round().long().clamp(0,31)
    return rgb[:,0]*1024+rgb[:,1]*32+rgb[:,2]


@torch.no_grad()
def learn_palette(rows):
    # RGB 5 bits por canal; estimar la paleta exclusivamente con TRAIN.
    hist=torch.zeros(32768,device="cuda")
    subset=np.random.default_rng(310).choice(len(rows),3072,replace=False)
    for offset in range(0,len(subset),32):
        pixels,_=batch(rows,subset[offset:offset+32],"cuda")
        hist+=torch.bincount(bins(pixels.reshape(-1,3,96,96)).flatten(),minlength=len(hist))
    idx=torch.arange(32768,device="cuda")
    rgb=torch.stack((idx//1024,idx//32%32,idx%32),1).float()/15.5-1
    hist[-1]=0
    w=hist.sqrt()
    pal=rgb[torch.multinomial(w,64,replacement=False)]
    pal[:2]=rgb[[32767,0]]
    for _ in range(25):
        ids=torch.cdist(rgb,pal).argmin(1)
        weights=torch.zeros(64,device="cuda").index_add_(0,ids,w)
        sums=torch.zeros_like(pal).index_add_(0,ids,w[:,None]*rgb)
        pal=torch.where(weights[:,None]>0,sums/weights[:,None].clamp_min(1),pal)
        pal[:2]=rgb[[32767,0]]
    return pal,torch.cdist(rgb,pal).argmin(1)


def main():
    out=ROOT/"runs/paleta_6000"
    out.mkdir()
    torch.manual_seed(310)
    base=ROOT/"runs/plan_piezas_5000/step_05000.pt"
    source=torch.load(base,map_location="cpu",weights_only=True)
    old=torch.load(ROOT/"runs/piezas_5000/step_03000.pt",map_location="cpu",weights_only=True)
    rows=train_rows()
    colors,lookup=learn_palette(rows)
    net=PaletteGenerator(source["groups"],len(colors),source["temperature"]).cuda()
    missing,unexpected=net.load_state_dict(source["generator"],strict=False)
    if unexpected or set(missing)!={n for n in net.state_dict() if n.startswith("tokenizer.detail.") or n=="tokenizer.palette"}:
        raise ValueError("El estado de partida no coincide con Plan")
    net.tokenizer.palette.copy_(colors)
    net.requires_grad_(False)
    for m in [net.tokenizer.decoder,net.tokenizer.detail,net.tokenizer.types]:
        m.requires_grad_(True)
    go=torch.optim.Adam((p for p in net.parameters() if p.requires_grad),lr=0.0003,betas=(0.5,0.99))
    d=Discriminator().cuda()
    d.load_state_dict(old["discriminator"])
    do=torch.optim.Adam(d.parameters(),lr=0.0002,betas=(0.,0.9))
    do.load_state_dict(old["do"])
    cfg=dict(start=str(base.relative_to(ROOT)),start_hash=hashlib.sha256(base.read_bytes()).hexdigest(),start_step=5000,
             steps=6000,colors=len(colors),palette_rows=3072,batch=24,seed=310,temperature=source["temperature"],
             data=source["data"],frozen=["prior","plan_distribution","means","occupancy","encoder","codes"])
    (out/"config.json").write_text(json.dumps(cfg,indent=2))
    begin=time.monotonic()
    with (out/"metrics.csv").open("w",newline="") as f:
        log=csv.writer(f)
        log.writerow(["step","color_ce","rgb","g_adv","d_hinge","seconds","peak_gpu_MiB"])
        for step in range(5001,6001):
            rng=np.random.default_rng([310,step])
            real,y=batch(rows,rng.integers(len(rows),size=24),"cuda")
            with torch.no_grad():
                ids,_,_=net.tokenizer.quantize(real)
                views=real.reshape(-1,3,96,96)
                target=lookup[bins(views)]
                contrast=(F.max_pool2d(views,3,1,1)+F.max_pool2d(-views,3,1,1)).amax(1)
                weights=1+4*(target!=0).float()+4*contrast
            fake,logits=net.tokenizer.palette_decode(ids,y,training=True)
            d.requires_grad_(True)
            do.zero_grad(set_to_none=True)
            dl=hinge(d(torch.cat([real,fake.detach()]),torch.cat([y,y])))
            update(dl,d,do)
            d.requires_grad_(False)
            ce=(F.cross_entropy(logits,target,reduction="none")*weights).mean()/weights.mean()
            rgb=((fake-real).abs().reshape_as(views).mean(1)*weights).mean()/weights.mean()
            adv=-d(fake,y).mean()
            go.zero_grad(set_to_none=True)
            update(ce+rgb+0.04*adv,net,go)
            log.writerow([step,ce.item(),rgb.item(),adv.item(),dl.item(),round(time.monotonic()-begin,2),
                          round(torch.cuda.max_memory_allocated()/2**20,1)])
            if step%250==0:
                f.flush()
                print(f"{step}: color CE {ce.item():.3f}",flush=True)
    temp=out/"step_06000.tmp"
    torch.save(dict(architecture="sprite_palette",groups=source["groups"],colors=len(colors),temperature=net.temperature,
                    step=6000,codec_steps=3000,generator=net.state_dict(),discriminator=d.state_dict(),go=go.state_dict(),
                    do=do.state_dict(),seed=310,batch=24,data=source["data"]),temp)
    temp.replace(out/"step_06000.pt")


if __name__=="__main__":
    main()
