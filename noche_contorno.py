"""Continúa una rama de Plan con un modelo de contornos pareados ajustado en TRAIN."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from noche_plan import atomic_text, digest, status, training_examples
from prepare import DATA, PREPARED, ROOT
from sprite_contours import ContourGenerator
from sprite_ink import InkGenerator,TONES,roles,luminance
from sprite_edges import EdgeGenerator
from sprite_tokens import GRID
from train import batch, train_rows
import hashlib
import json


@torch.no_grad()
def masks_train(data):
    path=ROOT/"noche/mascaras48.pt"
    if path.exists():
        saved=torch.load(path,weights_only=True)
        if saved["data"]==data:
            return saved["masks"].cuda()
    rows=train_rows()
    masks=[]
    for i in range(0,len(rows),48):
        real,_=batch(rows,range(i,min(i+48,len(rows))),"cuda")
        foreground=real.reshape(-1,2,3,96,96).amin(2)<0.92
        masks.append((4*F.avg_pool2d(foreground.float(),2)).byte().cpu())
    masks=torch.cat(masks)
    tmp=path.with_suffix(".tmp")
    torch.save(dict(data=data,masks=masks),tmp)
    tmp.replace(path)
    return masks.cuda()


@torch.no_grad()
def ink_targets(net,tokens,labels,data):
    path=ROOT/"noche/tonos_train.pt"
    if path.exists():
        cache=torch.load(path,weights_only=True)
        if cache["data"]==data:
            return cache["roles"].cuda(),cache["shade"].cuda()
    rows=train_rows()
    tones,shade=[],[]
    for first in range(0,len(rows),32):
        ix=slice(first,min(first+32,len(rows)))
        real,_=batch(rows,range(ix.start,ix.stop),"cuda")
        ids=tokens[ix].long().reshape(-1,GRID*GRID,2).transpose(1,2).reshape(-1,2,GRID,GRID)
        coarse=net.tokenizer.decode(ids,labels[ix])[0].reshape(-1,2,3,96,96)
        tones.append(roles((real.reshape_as(coarse)+1)/2)[0].byte().cpu())
        shade.append((255*F.avg_pool2d(luminance((coarse+1)/2),4)).round().byte().cpu())
    tones,shade=torch.cat(tones),torch.cat(shade)
    tmp=path.with_suffix(".tmp")
    torch.save(dict(data=data,roles=tones,shade=shade),tmp)
    tmp.replace(path)
    return tones.cuda(),shade.cuda()


def contour_loss(head,target,mean,cond,rng):
    mu,logvar=head.encode(target,mean,cond)
    z=mu+torch.randn(mu.shape,device=mu.device,generator=rng)*(0.5*logvar).exp()
    logits=head.decode(z,mean,cond)
    pred=logits.sigmoid()
    bce=F.binary_cross_entropy_with_logits(logits,target)
    dice=(1-(2*(pred*target).sum((2,3))+1)/(pred.sum((2,3))+target.sum((2,3))+1)).mean()
    kl=(mu.square()+logvar.exp()-logvar-1).mean()/2
    return bce+0.65*dice+0.035*kl,[bce,dice,kl]


def ink_loss(net,target,mean,cond,tones,shade,rng):
    with torch.no_grad():
        mu,logvar=net.contour.encode(target,mean,cond)
        latent=mu+torch.randn(mu.shape,device=mu.device,generator=rng)*(0.5*logvar).exp()
        mask=net.contour.decode(latent,mean,cond)
        mask=(2*F.max_pool2d(F.interpolate(mask,scale_factor=2,mode="bilinear",align_corners=False),3,1,1)).sigmoid()
        # Alternar contornos reconstruidos y reales ayuda a tolerar pequeñas desviaciones.
        exact=F.interpolate(target,scale_factor=2,mode="bilinear",align_corners=False)
        use_real=torch.rand(len(mask),1,1,1,device=mask.device,generator=rng)<0.5
        mask=torch.where(use_real,exact,mask)
    logits=net.ink(mask,shade.float()/255)
    target_tones=tones.long().flatten(0,1)
    # Dar peso a los trazos interiores de un píxel junto a zonas más claras.
    local=F.max_pool2d(target_tones[:,None].float().masked_fill(target_tones[:,None]==TONES,0),3,1,1)[:,0]
    weights=1+1.5*((target_tones<=1)&(local-target_tones>=2))
    ce=F.cross_entropy(logits.flatten(0,1),target_tones,ignore_index=TONES,reduction="none")
    loss=(ce*weights).sum()/weights[target_tones!=TONES].sum().clamp_min(1)
    return loss,[loss]


def edge_loss(net,target,mean,cond,tones,tokens,y,rng):
    with torch.no_grad():
        ids=tokens.long().reshape(-1,GRID*GRID,2).transpose(1,2).reshape(-1,2,GRID,GRID)
        pair=net.tokenizer.decode(ids,y)[0].reshape(-1,2,3,96,96)
        evidence=(1-pair.amin(2))/2
        mu,logvar=net.contour.encode(target,mean,cond)
        z=mu+torch.randn(mu.shape,device=mu.device,generator=rng)*(0.5*logvar).exp()
        coarse=net.contour.decode(z,mean,cond)
        coarse=(2*F.max_pool2d(F.interpolate(coarse,scale_factor=2,mode="bilinear",align_corners=False),3,1,1)).sigmoid()
        clean=(tones!=TONES).float()*2-1
        noise=torch.randn(clean.shape,device=clean.device,generator=rng)
        t=torch.rand(len(clean),device=clean.device,generator=rng)
        theta=t[:,None,None,None]*math.pi/2
        noisy=theta.cos()*clean+theta.sin()*noise
        velocity=theta.cos()*noise-theta.sin()*clean
        edge=(F.max_pool2d(clean,5,1,2)+F.max_pool2d(-clean,5,1,2))/2
        weights=1+3*edge+(clean>0)
    pred=net.edge(noisy,t,coarse,evidence)
    loss=(weights*(pred-velocity).square()).mean()/weights.mean()
    return loss,[loss]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--from-ck",type=Path,required=True)
    p.add_argument("--until",type=int,required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--tinta",action="store_true")
    mode.add_argument("--borde",action="store_true")
    args=p.parse_args()
    out,base,end=args.run,args.from_ck,args.until
    out.mkdir(parents=True,exist_ok=True)
    cfg=out/"continuacion.json"
    config=dict(start=base.as_posix(),start_hash=digest(base),end=end,latent=32,mask=48,batch=24,seed=310)
    if args.tinta:
        config["tinta"]=True
    if args.borde:
        config["borde"]=True
    if cfg.exists() and json.loads(cfg.read_text())!=config:
        raise ValueError("Esta carpeta ya contiene otra corrida")
    atomic_text(cfg,json.dumps(config,indent=2))
    checkpoints=list(out.glob("step_*.pt"))
    ck=max(checkpoints,key=lambda f:int(f.stem.split("_")[-1])) if checkpoints else base
    state=torch.load(ck,map_location="cpu",weights_only=True)
    data=hashlib.sha256((PREPARED/"splits.csv").read_bytes()+(DATA/"pairs.csv").read_bytes()).hexdigest()
    if args.borde:
        allowed,model,head_name=("sprite_ink","sprite_edges"),EdgeGenerator,"edge"
    elif args.tinta:
        allowed,model,head_name=("sprite_contours","sprite_ink"),InkGenerator,"ink"
    else:
        allowed,model,head_name=("sprite_plans","sprite_contours"),ContourGenerator,"contour"
    architecture=allowed[1]
    optimizer_key=head_name+"_opt"
    start_key=head_name+"_start"
    if data!=state["data"] or state["architecture"] not in allowed:
        raise ValueError("Origen incompatible")
    torch.manual_seed(310)
    net=model(state["groups"],state["temperature"]).cuda()
    missing,extra=net.load_state_dict(state["generator"],strict=False)
    if extra or any(not key.startswith(head_name+".") for key in missing) or (state["architecture"]==architecture and missing):
        raise ValueError("Pesos incompletos")
    net.requires_grad_(False)
    head=getattr(net,head_name).requires_grad_(True)
    opt=torch.optim.AdamW(head.parameters(),lr=0.0006,betas=(0.9,0.95),weight_decay=0.01)
    if optimizer_key in state:
        opt.load_state_dict(state[optimizer_key])
    start=state["step"]
    if start==end:
        print(f"Verificado {ck}")
        return
    if start>end:
        raise ValueError("Destino anterior al origen")
    command=f".\\.venv\\Scripts\\python noche_contorno.py --run {out.as_posix()} --from-ck {base.as_posix()} --until {end}"
    if args.tinta:
        command+=" --tinta"
    if args.borde:
        command+=" --borde"
    decision="Comparar cuerpo y rasgo compartido con Plan en las mismas 32 parejas de desarrollo."
    status(out.name,ck,command,decision,f"Inicio de {out.name} desde {ck.as_posix()} hasta {end}.")
    tokens,labels,groups=training_examples(net,data)
    masks=masks_train(data)
    if args.tinta or args.borde:
        roles_train,shades_train=ink_targets(net,tokens,labels,data)
    metrics=out/"metricas.csv"
    if metrics.exists():
        with metrics.open() as f:
            kept=[line for line in f if not line[0].isdigit() or int(line.split(",")[0])<=start]
        atomic_text(metrics,"".join(kept))
    with metrics.open("a",newline="") as stream:
        log=csv.writer(stream)
        if stream.tell()==0:
            columns={"edge":["MSE_velocidad"],"ink":["CE_tonos"],"contour":["BCE","Dice_loss","KL_por_dimension"]}
            log.writerow(["paso",*columns[head_name]])
        for step in range(start+1,end+1):
            ix=torch.from_numpy(np.random.default_rng([310,step]).integers(len(groups),size=24)).cuda()
            target=masks[ix].float()/4
            group=groups[ix]
            mean=net.means[group]
            cond=net.contour.condition(group,labels[ix])
            rng=torch.Generator(device="cuda").manual_seed(3100000+step)
            if args.borde:
                loss,values=edge_loss(net,target,mean,cond,roles_train[ix],tokens[ix],labels[ix],rng)
            elif args.tinta:
                loss,values=ink_loss(net,target,mean,cond,roles_train[ix],shades_train[ix],rng)
            else:
                loss,values=contour_loss(head,target,mean,cond,rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1,error_if_nonfinite=True)
            opt.step()
            log.writerow([step,*[v.item() for v in values]])
            if step%500==0 or step==end:
                stream.flush()
                dest=out/f"step_{step:05}.pt"
                temp=dest.with_suffix(".tmp")
                saved=dict(state,architecture=architecture,step=step,generator=net.state_dict())
                saved[start_key]=state.get(start_key,start)
                saved[optimizer_key]=opt.state_dict()
                torch.save(saved,temp)
                verified=torch.load(temp,map_location="cpu",weights_only=True)
                net.load_state_dict(verified["generator"])
                if any(int(s["step"])!=step-verified[start_key] for s in verified[optimizer_key]["state"].values()):
                    raise ValueError("Estado de AdamW incompleto")
                temp.replace(dest)
                done=step==end
                status("ninguna" if done else out.name,dest,command,decision,
                       f"Cierre de {out.name} en {dest.as_posix()}." if done else None)
                print(f"Verificado {dest}",flush=True)


if __name__=="__main__":
    main()
