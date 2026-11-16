"""Ajusta conjuntamente las proporciones y la apariencia de un generador articulado."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import Discriminator
from noche_plan import atomic_text,status
from sprite_parts import PartsGenerator
from sprite_families import FamilyGenerator
from estructura_familias import fit_family_statistics
from sprite_relieve import ReliefGenerator
from estructura_relieve import fit_relief
from train_estructura import dataset
from prepare import TYPES


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--hasta",type=int,required=True)
    parser.add_argument("--run",type=Path,default=Path("runs/estructura_partes"))
    parser.add_argument("--familias",action="store_true")
    parser.add_argument("--relieve",action="store_true")
    parser.add_argument("--linaje",action="store_true")
    parser.add_argument("--grabado",action="store_true")
    parser.add_argument("--elemental",action="store_true")
    parser.add_argument("--prisma",action="store_true")
    parser.add_argument("--insignias",action="store_true")
    parser.add_argument("--union",action="store_true")
    parser.add_argument("--desde",type=Path)
    args=parser.parse_args()
    args.insignias=args.insignias or args.union
    args.prisma=args.prisma or args.insignias
    args.elemental=args.elemental or args.prisma
    args.grabado=args.grabado or args.elemental
    args.linaje=args.linaje or args.grabado
    out=args.run
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(2831)
    if args.linaje or args.relieve:
        g=ReliefGenerator(lineage=args.linaje,engraved=args.grabado,elemental=args.elemental,
                          prisma=args.prisma,insignias=args.insignias,joined=args.union)
    else:
        g=FamilyGenerator() if args.familias else PartsGenerator()
    g=g.cuda()
    multiple=args.familias or args.relieve or args.linaje
    architecture="paired_engraved" if args.grabado else "paired_lineage" if args.linaje else "paired_relief" if args.relieve else "paired_families" if args.familias else "paired_parts"
    if args.elemental:
        architecture="paired_elemental"
    if args.prisma:
        architecture="paired_prisma"
    if args.insignias:
        architecture="paired_insignias"
    if args.union:
        architecture="paired_insignias_neck"
    calibrated=("topo","ink","tones","surface_mean","surface_basis") if args.grabado else ("topo","ink","tones")
    enc=nn.Sequential(nn.Conv2d(6,32,4,2,1),nn.SiLU(),nn.Conv2d(32,48,4,2,1),nn.GroupNorm(8,48),nn.SiLU(),
        nn.Conv2d(48,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(1024,64)).cuda()
    d=Discriminator().cuda()
    opt=torch.optim.AdamW([*g.parameters(),*enc.parameters()],lr=.0005,betas=(.9,.98))
    do=torch.optim.Adam(d.parameters(),lr=.00018,betas=(0.0,.99))
    paths=sorted(out.glob("step_*.pt"))
    start=0
    source=paths[-1] if paths else args.desde
    if args.linaje and not paths and not (out/"inicio.pt").exists():
        torch.save(dict(architecture=architecture,generator=g.state_dict(),seed=2831,step=0),out/"inicio.pt")
    if source:
        st=torch.load(source,map_location="cpu",weights_only=True)
        for obj,key in ((g,"generator"),(enc,"encoder"),(d,"disc"),(opt,"opt"),(do,"dop")):
            if obj is g and args.linaje and not paths:
                mismatch=g.load_state_dict(st[key],strict=False)
                assert set(mismatch.missing_keys)<=set(calibrated) and not mismatch.unexpected_keys
            else:
                obj.load_state_dict(st[key])
        start=st["step"]
    if start>=args.hasta:
        return
    cmd=f".\\.venv\\Scripts\\python train_parts.py --run {out.as_posix()} --hasta {args.hasta}"
    if args.familias:
        cmd+=" --familias"
    if args.relieve:
        cmd+=" --relieve"
    if args.union:
        cmd+=" --union"
    elif args.insignias:
        cmd+=" --insignias"
    elif args.prisma:
        cmd+=" --prisma"
    elif args.elemental:
        cmd+=" --elemental"
    elif args.grabado:
        cmd+=" --grabado"
    elif args.linaje:
        cmd+=" --linaje"
    if args.desde:
        cmd+=f" --desde {args.desde.as_posix()}"
    decision="Verificar la correspondencia y el riesgo de repetir una misma plantilla en las galerías ciegas."
    reference=source or (out/"inicio.pt" if (out/"inicio.pt").exists() else Path("runs/noche_plan_10000/step_07500.pt"))
    history=f"Continuación desde {source.as_posix()} en el paso {start}." if source else "Inicio de esta rama con pesos inicializados desde cero."
    status(out.name,reference,cmd,decision,history)
    cache=dataset()
    focus=[]
    if args.elemental:
        for case in ("water+dragon","ghost","dragon+ghost","water+ghost","bug+poison"):
            tags=torch.tensor([name in case.split("+") for name in TYPES])
            matches=(cache["y"]==tags).all(1).nonzero().flatten().numpy()
            if len(matches):
                focus.append(matches)
    if multiple:
        initial=out/"inicio.pt"
        if not initial.exists():
            torch.save(dict(architecture=architecture,generator=g.state_dict(),seed=2831,step=0),initial)
        groups_path=out/"grupos_train.pt"
        if args.linaje:
            learned=torch.load("estructura/anotacion_train/prior.pt",weights_only=True)
            if args.grabado:
                texture=torch.load("estructura/anotacion_train/superficie.pt",weights_only=True)
                assert texture["data"]==learned["data"]
                learned.update(texture)
            if args.prisma and not paths:
                from pigmentos_duales import fit_pigments
                tones,ink=fit_pigments(cache)
                learned.update(tones=tones.cpu(),ink=ink.cpu())
            if learned["data"]!=cache["data"]:
                raise ValueError("Las anotaciones pertenecen a otra partición")
            groups=learned["groups"].cuda()
            if not paths:
                for key in calibrated:
                    getattr(g,key).copy_(learned[key])
                torch.save(dict(architecture=architecture,generator=g.state_dict(),step=start,
                           origen=source.as_posix() if source else None,data=cache["data"]),out/"entrada.pt")
                zero=torch.load(out/"inicio.pt",weights_only=True,map_location="cpu")
                for key in calibrated:
                    zero["generator"][key]=learned[key]
                torch.save(zero,out/"calibrado_00000.pt")
        elif start and groups_path.exists():
            groups=torch.load(groups_path,weights_only=True).cuda()
        else:
            groups=fit_relief(g,cache) if args.relieve else fit_family_statistics(g,cache)
            torch.save(groups.cpu(),groups_path)
    if source and st["data"]!=cache["data"]:
        raise ValueError("Checkpoint de otra partición")
    log=out/"perdidas.csv"
    if log.exists():
        atomic_text(log,"".join(l for l in log.open() if not l[0].isdigit() or int(l.split(",")[0])<=start))
    with log.open("a",newline="") as stream:
        w=csv.writer(stream)
        if not stream.tell():
            w.writerow(["step","color","forma","kl","adv"])
        for step in range(start+1,args.hasta+1):
            rng=np.random.default_rng([2831,step])
            gen=torch.Generator(device="cuda").manual_seed(28310000+step)
            batch_size=4 if args.elemental else 8 if args.relieve or args.linaje else 16
            ix=rng.integers(len(cache["y"]),size=batch_size)
            if args.elemental:
                for i in range(len(ix)//2):
                    ix[i]=rng.choice(focus[rng.integers(len(focus))])
                rng.shuffle(ix)
            real=cache["images"][ix].cuda().float()/127.5-1
            y=cache["y"][ix].cuda()
            # Emparejar las orientaciones de las cabezas al compartir la geometría.
            target=(real.amin(2)<.94).float()
            if multiple:
                pm=real.new_zeros(len(y),32)
                pl=torch.zeros_like(pm)
                selected=groups[ix]
            else:
                pm,pl=g.distribution(y)
            mu,logs=enc(real.flatten(1,2)).chunk(2,1)
            logs=logs.clamp(-3,1)
            eps=torch.randn(mu.shape,generator=gen,device="cuda")
            u=mu+logs.exp()*eps
            if multiple:
                pred,mask=g.render(u,y,selected)
            else:
                pred,mask=g.render(u,y)
            overlap=(1-(2*(mask*target).sum((2,3))+1)/(mask.sum((2,3))+target.sum((2,3))+1)).mean()
            weights=1+4*target[:,:,None]
            color=((pred-real).abs()*weights).mean()/weights.mean()
            kl=(pl-logs+.5*((2*(logs-pl)).exp()+(mu-pm).square()*(-2*pl).exp()-1)).mean()
            adv=color.new_zeros(())
            diversity=color.new_zeros(())
            if step>100:
                prior_z=pm+pl.exp()*torch.randn(mu.shape,generator=gen,device="cuda")
                fake_y=y
                if args.elemental:
                    fake_y=y[:len(y)//2].repeat(2,1)
                    prior,occupancy=g.render(prior_z,fake_y,g.choose(prior_z,fake_y))
                    one,two=F.avg_pool2d(occupancy,4).chunk(2)
                    distance=(one-two).abs().mean((1,2,3))
                    z1,z2=prior_z.chunk(2)
                    latent_distance=(z1-z2).abs().mean(1)
                    diversity=F.relu(.12-distance/latent_distance.clamp_min(.1)).mean()
                elif multiple:
                    prior,_=g.render(prior_z,y,g.choose(prior_z,y))
                else:
                    prior,_=g.render(prior_z,y)
                free=prior.flatten(1,2)
                d.requires_grad_(True)
                do.zero_grad(set_to_none=True)
                a,b=d(torch.cat([real.flatten(1,2),free.detach()]),torch.cat([y,fake_y])).chunk(2)
                (F.relu(1-a).mean()+F.relu(1+b).mean()).backward()
                do.step()
                d.requires_grad_(False)
                adv=-d(free,fake_y).mean()
            opt.zero_grad(set_to_none=True)
            (color+.8*overlap+.045*kl+.025*adv+.3*diversity).backward()
            torch.nn.utils.clip_grad_norm_([*enc.parameters(),*g.parameters()],2,error_if_nonfinite=True)
            opt.step()
            if step%25==0:
                w.writerow([step,color.item(),overlap.item(),kl.item(),adv.item()])
            if step%500==0 or step==args.hasta:
                stream.flush()
                path=out/f"step_{step:05}.pt"
                tmp=path.with_suffix(".tmp")
                torch.save(dict(architecture=architecture,generator=g.state_dict(),encoder=enc.state_dict(),
                    disc=d.state_dict(),opt=opt.state_dict(),dop=do.state_dict(),step=step,data=cache["data"]),tmp)
                checked=torch.load(tmp,map_location="cpu",weights_only=True)
                g.load_state_dict(checked["generator"])
                assert all(int(s["step"])==step for s in checked["opt"]["state"].values())
                tmp.replace(path)
                done=step==args.hasta
                status("ninguna" if done else out.name,path,cmd,decision,f"Cierre de {out.name} en {step}." if done else None)
                print("Verificado",path,flush=True)


if __name__=="__main__":
    main()
