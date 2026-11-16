"""Aprende el mapa de pigmentos y un prior por mezcla de muchas especies TRAIN."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from noche_plan import status, atomic_text
from model import Discriminator
from sprite_boceto import COLORS, DIM, GROUPS, RANK, Encoder, SketchGenerator
from train_estructura import dataset


@torch.no_grad()
def quantize(images):
    targets, palettes = [], []
    for group in images.split(48):
        x = group.cuda().permute(0,1,3,4,2).reshape(len(group),-1,3).float()/127.5-1
        fg = x.amin(-1)<.92
        luma = (x*x.new_tensor([.299,.587,.114])).sum(-1)
        inds = torch.where(fg,luma,3.).argsort(1)
        loc = (fg.sum(1,keepdim=True)-1).clamp_min(0)*torch.linspace(.015,.985,COLORS,device=x.device)
        pal = x.gather(1,inds.gather(1,loc.long())[:,:,None].expand(-1,-1,3))
        for _ in range(7):
            ids = torch.cdist(x,pal).argmin(2)
            sums = torch.zeros_like(pal).scatter_add_(1,ids[:,:,None].expand_as(x),x*fg[:,:,None])
            count = torch.zeros_like(pal[:,:,:1]).scatter_add_(1,ids[:,:,None],fg[:,:,None].float())
            pal = torch.where(count>0,sums/count.clamp_min(1),pal)
        order = (pal*x.new_tensor([.299,.587,.114])).sum(-1).argsort(1)
        pal = pal.gather(1,order[:,:,None].expand_as(pal))
        ids = (1+torch.cdist(x,pal).argmin(2))*fg
        targets.append(ids.reshape(-1,2,64,64).byte().cpu())
        palettes.append(pal.cpu())
    return torch.cat(targets),torch.cat(palettes)


@torch.no_grad()
def fit_prior(net,encoder,data,pal,step):
    # Mezclas ajustadas sobre representaciones de ambas vistas, sin almacenar casos en los pesos.
    device=net.centers.device
    lat=[]
    for x in data["images"].split(64):
        x=x.to(device).float()/127.5-1
        lat.append(encoder(x,(x.amin(2)<.92).float()))
    u=torch.cat(lat)
    field=torch.cat([u,pal.cuda().flatten(1)],1)
    # El color influye menos que la forma en la asignación de grupos.
    metric=torch.cat([u,.45*pal.cuda().flatten(1)],1)
    gen=torch.Generator(device=device).manual_seed(264)
    centers=metric[torch.randperm(len(metric),device=device,generator=gen)[:GROUPS]].clone()
    species=torch.tensor(np.unique(data["species"],return_inverse=True)[1],device=device)
    one=torch.ones(len(u),device=device)
    types=data["y"].cuda()
    for _ in range(12):
        ix=torch.cdist(metric,centers).argmin(1)
        support=torch.bincount(ix*int(species.max()+1)+species,minlength=GROUPS*int(species.max()+1)).reshape(GROUPS,-1)
        w=one/support[ix,species].clamp_min(1).sqrt()
        total=one.new_zeros(GROUPS).scatter_add_(0,ix,w)
        new=torch.zeros_like(centers).index_add_(0,ix,metric*w[:,None])/total.clamp_min(1)[:,None]
        centers=torch.where((total>0)[:,None],new,centers)
    distance=torch.cdist(centers,metric)
    freq=field.new_zeros(types.shape[1]+1,GROUPS)
    for k in range(GROUPS):
        # Cada componente agrega al menos 8 especies. Las versiones de una especie
        # se reparten su peso para que ninguna sirva como único prototipo.
        neighbors=distance[k].argsort()[:192]
        take=neighbors[:48]
        for length in (64,96,128,192):
            if len(species[take].unique())>=8:
                break
            take=neighbors[:length]
        if len(species[take].unique())<8:
            take=distance[k].argsort()
            limit=192
            while len(species[take[:limit]].unique())<8:
                limit+=32
            take=take[:limit]
        count=torch.bincount(species[take])
        weights=count[species[take]].float().reciprocal()
        weights/=weights.sum()
        values=field[take]
        mean=(weights[:,None]*values).sum(0)
        delta=values-mean
        cov=(delta*weights[:,None]).T@delta
        vals,vec=torch.linalg.eigh(cov)
        vec=vec[:,-RANK:].T*vals[-RANK:].clamp_min(0).sqrt()[:,None]
        net.centers[k]=mean
        net.basis[k]=vec
        net.scatter[k]=(cov.diag()-vec.square().sum(0)).clamp_min(0).sqrt()
        freq[0,k]=1
        freq[1:,k]=(weights[:,None]*types[take]).sum(0)
    net.freq.copy_((freq+.012)/(freq+.012).sum(1,keepdim=True))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--hasta",type=int,required=True)
    p.add_argument("--run",type=Path,default=Path("runs/estructura_boceto"))
    p.add_argument("--desde",type=Path)
    p.add_argument("--adversarial",action="store_true")
    args=p.parse_args()
    out=args.run
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(3941)
    net=SketchGenerator().cuda()
    enc=Encoder().cuda()
    opt=torch.optim.AdamW([*net.parameters(),*enc.parameters()],lr=.00055,betas=(.9,.98))
    if args.adversarial:
        disc=Discriminator().cuda()
        do=torch.optim.Adam(disc.parameters(),lr=.00016,betas=(0,.99))
    paths=sorted(out.glob("step_*.pt"))
    start=0
    source=paths[-1] if paths else args.desde
    if source:
        saved=torch.load(source,map_location="cpu",weights_only=True)
        for obj,key in [(net,"generator"),(enc,"encoder"),(opt,"opt")]:
            obj.load_state_dict(saved[key])
        start=saved["step"]
        if paths and args.adversarial!=("disc" in saved):
            raise ValueError("Esta carpeta contiene otra variante")
        if args.adversarial and "disc" in saved:
            disc.load_state_dict(saved["disc"])
            do.load_state_dict(saved["dop"])
        if args.adversarial:
            for group in opt.param_groups:
                group["lr"]=.00023
    if args.adversarial and not source:
        raise ValueError("Se necesita un prior aprendido para generar las parejas adversariales")
    if start>=args.hasta:
        return
    cmd=f".\\.venv\\Scripts\\python train_boceto.py --run {out.as_posix()} --hasta {args.hasta}"
    if args.desde:
        cmd+=f" --desde {args.desde.as_posix()}"
    if args.adversarial:
        cmd+=" --adversarial"
    decision="Comparar las dos posturas, el detalle interior y la variación por tipo y z en las semillas de desarrollo."
    status(out.name,source or Path("runs/estructura_familias/step_02500.pt"),cmd,decision,
           f"Inicio de {out.name} desde el paso {start}, objetivo {args.hasta}.")
    data=dataset()
    if start and saved["data"]!=data["data"]:
        raise ValueError("La partición de entrenamiento cambió")
    target_file=(args.desde.parent if args.desde else out)/"objetivos.pt"
    if target_file.exists():
        cache=torch.load(target_file,weights_only=True)
        if cache["data"]!=data["data"]:
            raise ValueError("Objetivos de otra partición")
        labels,pal=cache["labels"],cache["pal"]
    else:
        labels,pal=quantize(data["images"])
        tmp=target_file.with_suffix(".tmp")
        torch.save(dict(labels=labels,pal=pal,data=data["data"]),tmp)
        tmp.replace(target_file)
    logfile=out/"perdidas.csv"
    if logfile.exists():
        atomic_text(logfile,"".join(l for l in logfile.open() if not l[0].isdigit() or int(l.split(",")[0])<=start))
    with logfile.open("a") as log:
        if not log.tell():
            log.write("paso,perdida\n")
        for step in range(start+1,args.hasta+1):
            rng=np.random.default_rng([3941,step])
            gen=torch.Generator(device="cuda").manual_seed(394100000+step)
            ix=rng.integers(len(labels),size=24)
            x=data["images"][ix].cuda().float()/127.5-1
            y=data["y"][ix].cuda()
            label=labels[ix].long().cuda()
            fg=(label!=0).float()
            u=enc(x,fg)
            maps=net.decode(u+.15*torch.randn(u.shape,generator=gen,device="cuda"),y)
            target=F.one_hot(label,COLORS+1).permute(0,1,4,2,3).float().flatten(0,1)
            loss=u.new_zeros(())
            for i,m in enumerate(maps):
                small=F.adaptive_avg_pool2d(target,m.shape[-2:])
                # Equilibrar fondo, contornos oscuros y reflejos pequeños.
                weights=small.new_tensor([.32,1.45,1.15,1,1,1,1,1.15,1.35])[None,:,None,None]
                cross=-(small*weights*m.flatten(0,1).log_softmax(1)).sum(1).mean()
                loss=loss+cross*(.23,.28,.48,1.)[i]
            if args.adversarial:
                z=torch.randn((len(y),96),device="cuda",generator=gen)
                uf,palf=net.sample_latent(z,y)
                logits=net.decode(uf,y)[-1]
                soft=(logits*.9).softmax(2)
                hard=F.one_hot(soft.argmax(2),COLORS+1).permute(0,1,4,2,3)
                mix=soft+(hard-soft).detach()
                fake=torch.einsum("nvkhw,nkc->nvchw",mix,palf).flatten(1,2)
                real_pal=torch.cat([torch.ones_like(palf[:,:1]),pal[ix].cuda()],1)
                real=torch.einsum("nvkhw,nkc->nvchw",target.unflatten(0,(-1,2)),real_pal).flatten(1,2)
                disc.requires_grad_(True)
                do.zero_grad(set_to_none=True)
                pos,neg=disc(torch.cat([real,fake.detach()]),torch.cat([y,y])).chunk(2)
                (F.relu(1-pos).mean()+F.relu(1+neg).mean()).backward()
                do.step()
                disc.requires_grad_(False)
                # Rasgos de la pareja a tres escalas compensan el dominio de los píxeles de fondo.
                r,f=real,fake
                feature=loss.new_zeros(())
                for layer in disc.blocks:
                    r=layer(r).detach()
                    f=layer(f)
                    if isinstance(layer,torch.nn.LeakyReLU):
                        feature+=(f.mean((0,2,3))-r.mean((0,2,3))).abs().mean()
                loss=loss+.11*(-disc(fake,y).mean())+.8*feature
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*net.parameters(),*enc.parameters()],4,error_if_nonfinite=True)
            opt.step()
            if step%50==0:
                log.write(f"{step},{loss.item():.6f}\n")
            if step%500==0 or step==args.hasta:
                log.flush()
                fit_prior(net,enc,data,pal,step)
                state=dict(architecture="paired_sketch",generator=net.state_dict(),encoder=enc.state_dict(),
                           opt=opt.state_dict(),step=step,data=data["data"],seed=3941)
                if args.adversarial:
                    state.update(disc=disc.state_dict(),dop=do.state_dict())
                path=out/f"step_{step:05}.pt"
                tmp=path.with_suffix(".tmp")
                torch.save(state,tmp)
                checked=torch.load(tmp,map_location="cpu",weights_only=True)
                assert checked["step"]==step and all(int(v["step"])==step for v in checked["opt"]["state"].values())
                tmp.replace(path)
                end=step==args.hasta
                status("ninguna" if end else out.name,path,cmd,decision,f"Cierre de {out.name} en {step}." if end else None)
                print("Verificado",path,flush=True)


if __name__=="__main__":
    main()
