"""Learn a paired outline refiner from the professor's original-resolution sprites."""
import argparse
import csv
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from model import Discriminator
from prepare import DATA, PREPARED, TYPES, read_rows
from sprite_trazo import TrazoGenerator


def images(split):
    with (PREPARED/"splits.csv").open() as f:
        ids={r["pair_id"] for r in csv.DictReader(f) if r["split"]==split}
    rows=[r for r in read_rows() if r["pair_id"] in ids]
    pixels=torch.empty(len(rows),6,96,96,dtype=torch.uint8)
    with zipfile.ZipFile(DATA/"images96.zip") as archive:
        for i,r in enumerate(rows):
            for j,key in enumerate(("front_path","back_path")):
                with Image.open(BytesIO(archive.read(r[key]))) as im:
                    pixels[i,3*j:3*j+3]=torch.from_numpy(np.array(im.convert("RGB"))).permute(2,0,1)
    tags=torch.tensor([[int(r["type_"+t]) for t in TYPES] for r in rows],dtype=torch.float32)
    return pixels,tags,rows


def degrade(real,gen):
    n=len(real)
    # Same coarse distortion in front and back, plus a small view-specific displacement.
    xy=torch.linspace(-1,1,96,device=real.device)
    grid=torch.stack(torch.meshgrid(xy,xy,indexing="xy"),-1)
    noise=torch.randn(n,2,4,4,generator=gen,device=real.device)
    warp=F.interpolate(noise,size=(96,96),mode="bicubic",align_corners=False).permute(0,2,3,1)
    warped=1+F.grid_sample(real-1,grid+.042*warp,align_corners=True,mode="nearest")
    coarse=F.interpolate(F.avg_pool2d(warped,3),scale_factor=3,mode="nearest")
    coarse=((coarse+1)*3.5).round()/3.5-1
    x=coarse.reshape(-1,3,96,96)
    flat=F.avg_pool2d(x,5,1,2,count_include_pad=False)
    x=torch.where(x.amin(1,keepdim=True)<.86,.65*x+.35*flat,1.)
    # Erase small stretches crossing the silhouette, including occasional neck/limb pixels.
    gaps=torch.rand(n*2,1,24,24,generator=gen,device=real.device)<.055
    return torch.where(F.interpolate(gaps.float(),scale_factor=4).bool(),1.,x).reshape_as(real)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=800)
    p.add_argument("--run",type=Path,default=Path("runs/trazo"))
    args=p.parse_args()
    args.run.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(8901)
    x,y,rows=images("train")
    _,sp,counts=np.unique([r["species_id"] for r in rows],return_inverse=True,return_counts=True)
    w=1/counts[sp]
    labels=y.numpy()
    frequencies=(labels.T@w).clip(1)
    weights=torch.tensor(w*((labels/np.sqrt(frequencies)).sum(1)/labels.sum(1)),dtype=torch.float64)
    net=TrazoGenerator().cuda()
    source=torch.load("pesos/ficha_todos.pth",map_location="cpu",weights_only=True)
    net.base.load_state_dict(source["generator"])
    net.base.eval().requires_grad_(False)
    disc=Discriminator().cuda()
    opt=torch.optim.AdamW(net.painter.parameters(),lr=.0007)
    od=torch.optim.Adam(disc.parameters(),lr=.00014,betas=(0.,.99))
    latest=sorted(args.run.glob("step_*.pt"))
    first=0
    if latest:
        state=torch.load(latest[-1],map_location="cpu",weights_only=True)
        if state["data"]!=source["data"]: raise ValueError("La partición cambió")
        for obj,key in ((net,"generator"),(opt,"opt"),(disc,"disc"),(od,"od")): obj.load_state_dict(state[key])
        first=state["step"]
    for step in range(first+1,args.steps+1):
        generator=torch.Generator().manual_seed(70000+step)
        idx=torch.multinomial(weights,12,replacement=True,generator=generator)
        real=x[idx].cuda().float()/127.5-1
        tags=y[idx].cuda()
        src=degrade(real,torch.Generator(device="cuda").manual_seed(80000+step))
        fake,logits=net.painter(src,tags)
        target=real.reshape(-1,3,96,96)
        mask=(target.amin(1,keepdim=True)<.94).float()
        boundary=F.max_pool2d(mask,5,1,2)+F.max_pool2d(-mask,5,1,2)
        cost=(fake-real).abs().reshape_as(target)
        loss=(cost*(mask+boundary)).sum()/(3*(mask+boundary).sum()).clamp_min(1)
        loss+=.8*F.binary_cross_entropy_with_logits(logits,mask,weight=.6+5*boundary)
        if step>60:
            disc.requires_grad_(True)
            od.zero_grad(set_to_none=True)
            pos,neg=disc(torch.cat([real,fake.detach()]),tags.repeat(2,1)).chunk(2)
            (F.relu(1-pos).mean()+F.relu(1+neg).mean()).backward()
            od.step()
            disc.requires_grad_(False)
            loss-=.009*disc(fake,tags).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.painter.parameters(),4,error_if_nonfinite=True)
        opt.step()
        if step%200==0 or step==args.steps:
            dest=args.run/f"step_{step:06}.pt"
            tmp=dest.with_suffix(".tmp")
            torch.save(dict(architecture="paired_trazo",generator=net.state_dict(),step=step,data=source["data"],
                            opt=opt.state_dict(),disc=disc.state_dict(),od=od.state_dict()),tmp)
            tmp.replace(dest)
            print(step,round(loss.item(),3),flush=True)

if __name__=="__main__":
    main()
