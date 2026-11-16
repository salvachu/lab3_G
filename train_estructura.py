"""Entrena desde cero un generador adversarial de difusión con ruido conjunto para ambas vistas."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from model import Discriminator
from noche_plan import atomic_text,digest,status
from prepare import DATA,PREPARED,TYPES
from sprite_joint import JointGenerator,representation
from train import train_rows


def canonical(im):
    arr=np.asarray(im.convert("RGB"))
    yy,xx=np.where(arr.min(2)<245)
    if len(xx)==0:
        return Image.new("RGB",(64,64),"white")
    obj=im.crop((xx.min(),yy.min(),xx.max()+1,yy.max()+1))
    factor=58/max(obj.size)
    size=tuple(max(1,round(v*factor)) for v in obj.size)
    out=Image.new("RGB",(64,64),"white")
    out.paste(obj.resize(size,Image.Resampling.NEAREST),((64-size[0])//2,(64-size[1])//2))
    return out


def dataset():
    rows=train_rows()
    path=Path("estructura/train64.pt")
    signature=digest(DATA/"pairs.csv")+digest(PREPARED/"splits.csv")
    if path.exists():
        saved=torch.load(path,weights_only=True)
        if saved["data"]!=signature:
            raise ValueError("Datos modificados")
        return saved
    pixels=[]
    for row in rows:
        pair=[]
        for key in ("front_path","back_path"):
            with Image.open(DATA/row[key]) as im:
                pair.append(torch.from_numpy(np.array(canonical(im))).permute(2,0,1))
        pixels.append(torch.stack(pair))
    saved=dict(images=torch.stack(pixels),y=torch.tensor([[int(r["type_"+t]) for t in TYPES] for r in rows],dtype=torch.float32),
               data=signature,species=[r["species_id"] for r in rows])
    tmp=path.with_suffix(".tmp")
    torch.save(saved,tmp)
    tmp.replace(path)
    return saved


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,default=Path("runs/estructura_conjunta"))
    p.add_argument("--hasta",type=int,required=True)
    p.add_argument("--batch",type=int,default=16)
    args=p.parse_args()
    out=args.run
    out.mkdir(exist_ok=True,parents=True)
    cmd=f".\\.venv\\Scripts\\python train_estructura.py --run {out.as_posix()} --hasta {args.hasta} --batch {args.batch}"
    torch.manual_seed(831)
    net=JointGenerator().cuda()
    ema=copy.deepcopy(net).eval().requires_grad_(False)
    disc=Discriminator().cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.00035,betas=(.9,.98),weight_decay=.01)
    dop=torch.optim.Adam(disc.parameters(),lr=.00015,betas=(0.0,.99))
    ck=sorted(out.glob("step_*.pt"))
    start=0
    decision="Comparar anatomía y correspondencia con Plan y Siluetas usando las columnas ocultas."
    if ck:
        state=torch.load(ck[-1],map_location="cpu",weights_only=True)
        if state["batch"]!=args.batch:
            raise ValueError("Lote distinto al checkpoint")
        net.load_state_dict(state["training"])
        ema.load_state_dict(state["generator"])
        disc.load_state_dict(state["disc"])
        opt.load_state_dict(state["opt"])
        dop.load_state_dict(state["dop"])
        start=state["step"]
    if start>=args.hasta:
        return
    status(out.name,ck[-1] if ck else Path("runs/noche_plan_10000/step_07500.pt"),cmd,decision,
           f"Inicio de {out.name} en el paso {start}. Prior y crítico inicializados localmente en el paso cero.")
    data=dataset()
    if start and state["data"]!=data["data"]:
        raise ValueError("Partición distinta")
    metrics=out/"perdidas.csv"
    if metrics.exists():
        atomic_text(metrics,"".join(l for l in metrics.open() if not l[0].isdigit() or int(l.split(",")[0])<=start))
    with metrics.open("a",newline="") as f:
        writer=csv.writer(f)
        if not f.tell():
            writer.writerow(["step","denoise","adversarial"])
        for step in range(start+1,args.hasta+1):
            rng=np.random.default_rng([831,step])
            ix=rng.integers(len(data["y"]),size=args.batch)
            gen=torch.Generator(device="cuda").manual_seed(83100000+step)
            real=data["images"][ix].cuda().float()/127.5-1
            y=data["y"][ix].cuda()
            flip=torch.rand(len(y),1,1,1,1,device="cuda",generator=gen)<.5
            real=torch.where(flip,real.flip(-1),real)
            clean=representation(real)
            t=(torch.arange(len(y),device="cuda")+torch.rand(len(y),generator=gen,device="cuda"))/len(y)
            theta=t[:,None,None,None]*math.pi/2
            noise=torch.randn(clean.shape,device="cuda",generator=gen)
            noisy=theta.cos()*clean+theta.sin()*noise
            yc=y*(torch.rand(len(y),1,device="cuda",generator=gen)>.12)
            velocity=net.net(noisy,t,yc)
            fg=(clean.reshape(-1,2,5,64,64)[:,:,3:4]+1)/2
            weights=(1+2*fg).expand(-1,-1,5,-1,-1).reshape_as(clean)
            err=velocity-(theta.cos()*noise-theta.sin()*clean)
            rec=(weights*err.square()).mean()/weights.mean()
            adv=rec.new_zeros(())
            if step>400 and step%2==0:
                n=len(y)//2
                estimate=(theta.cos()*noisy-theta.sin()*velocity)[:n].reshape(-1,2,5,64,64)
                fake=estimate[:,:,:3].clamp(-1,1).flatten(1,2)
                disc.requires_grad_(True)
                dop.zero_grad(set_to_none=True)
                a,b=disc(torch.cat([real[:n].flatten(1,2),fake.detach()]),torch.cat([y[:n],y[:n]])).chunk(2)
                (F.relu(1-a).mean()+F.relu(1+b).mean()).backward()
                dop.step()
                disc.requires_grad_(False)
                adv=-disc(fake,y[:n]).mean()
            opt.zero_grad(set_to_none=True)
            (rec+.018*adv).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2,error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),max(.002,1/(step+1)))
            if step%25==0:
                writer.writerow([step,rec.item(),adv.item()])
            if step%500==0 or step==args.hasta:
                f.flush()
                dest=out/f"step_{step:05}.pt"
                tmp=dest.with_suffix(".tmp")
                torch.save(dict(architecture="joint_diffusion",step=step,training=net.state_dict(),generator=ema.state_dict(),
                                disc=disc.state_dict(),opt=opt.state_dict(),dop=dop.state_dict(),data=data["data"],batch=args.batch),tmp)
                saved=torch.load(tmp,map_location="cpu",weights_only=True)
                assert saved["step"]==step and all(int(v["step"])==step for v in saved["opt"]["state"].values())
                ema.load_state_dict(saved["generator"])
                tmp.replace(dest)
                done=step==args.hasta
                status("ninguna" if done else out.name,dest,cmd,decision,f"Cierre de {out.name} en {step}." if done else None)
                print("Verificado",dest,flush=True)


if __name__=="__main__":
    main()
