"""Entrena desde cero el vocabulario adversarial y después el prior de la pareja."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import Discriminator
from prepare import DATA, PREPARED, ROOT
from sprite_tokens import TokenGenerator, VOCAB, sequence
from train import batch, train_rows
from train_silhouettes import hinge, update


def image_loss(x, prediction, alpha_logits):
    target=x.reshape(-1,3,96,96)
    fg=(target.amin(1,keepdim=True)<0.97).float()
    contrast=(F.max_pool2d(target,3,1,1)+F.max_pool2d(-target,3,1,1)).amax(1,keepdim=True)
    weights=1+4*fg+3*contrast
    color=((target-prediction.reshape_as(target)).abs()*weights).mean()/weights.mean()
    opacity=F.binary_cross_entropy_with_logits(alpha_logits,fg)
    return color+0.15*opacity


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--codec",type=int,default=3000)
    p.add_argument("--steps",type=int,default=5000)
    p.add_argument("--resume",type=Path,help="Continúa exclusivamente el prior con el codec congelado")
    p.add_argument("--temperature",type=float,default=0.9)
    args=p.parse_args()
    if args.run.exists() or not 400<args.codec<args.steps:
        p.error("Seleccione carpeta nueva y presupuestos válidos")
    torch.manual_seed(310)
    net=TokenGenerator(args.temperature).cuda()
    d=Discriminator().cuda()
    go=torch.optim.Adam(net.tokenizer.parameters(),lr=0.0005,betas=(0.5,0.99))
    do=torch.optim.Adam(d.parameters(),lr=0.0002,betas=(0.0,0.9))
    po=torch.optim.AdamW(net.prior.parameters(),lr=0.0006,betas=(0.9,0.95),weight_decay=0.02)
    data=hashlib.sha256((PREPARED/"splits.csv").read_bytes()+(DATA/"pairs.csv").read_bytes()).hexdigest()
    rows=train_rows()
    start=0
    if args.resume:
        state=torch.load(args.resume,map_location="cpu",weights_only=True)
        if state["data"]!=data or state["codec_steps"]!=args.codec or not args.codec<=state["step"]<args.steps:
            raise ValueError("Estado del prior incompatible")
        start=state["step"]
        net.load_state_dict(state["generator"])
        d.load_state_dict(state["discriminator"])
        go.load_state_dict(state["go"])
        do.load_state_dict(state["do"])
        po.load_state_dict(state["po"])
    args.run.mkdir(parents=True)
    config=dict(vars(args),data=data,seed=310,batch=24,initial_step=start,train_pairs=len(rows),
                gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
                source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                        for name in ("sprite_tokens.py","train_sprite_tokens.py","train.py","model.py")})
    (args.run/"config.json").write_text(json.dumps(config,default=str,indent=2))
    def save(step):
        dest=args.run/f"step_{step:05}.pt"
        tmp=dest.with_suffix(".tmp")
        torch.save(dict(architecture="sprite_tokens",step=step,codec_steps=args.codec,generator=net.state_dict(),
                        discriminator=d.state_dict(),go=go.state_dict(),do=do.state_dict(),po=po.state_dict(),
                        data=data,seed=310,batch=24,temperature=args.temperature),tmp)
        tmp.replace(dest)
    begin=time.monotonic()
    with (args.run/"metrics.csv").open("w",newline="") as f:
        log=csv.writer(f)
        log.writerow(["step","phase","recon","commit","g_adv","d_hinge","prior_nll","seconds","peak_gpu_MiB"])
        for step in range(start+1,args.codec+1):
            rng=np.random.default_rng([310,step])
            x,y=batch(rows,rng.integers(len(rows),size=24),"cuda")
            ids,straight,commit=net.tokenizer.quantize(x,update=True)
            fake,alpha=net.tokenizer.decode(ids,y,straight)
            dl=adv=x.new_zeros(())
            if step>400:
                d.requires_grad_(True)
                do.zero_grad(set_to_none=True)
                dl=hinge(d(torch.cat([x,fake.detach()]),torch.cat([y,y])))
                update(dl,d,do)
                d.requires_grad_(False)
                adv=-d(fake,y).mean()
            rec=image_loss(x,fake,alpha)
            go.zero_grad(set_to_none=True)
            update(rec+0.25*commit+0.035*adv,net.tokenizer,go)
            log.writerow([step,"VQ",rec.item(),commit.item(),adv.item(),dl.item(),"",round(time.monotonic()-begin,2),
                          round(torch.cuda.max_memory_allocated()/2**20,1)])
            if step%500==0 or step==args.codec:
                f.flush()
                print(f"VQ {step}: recon {rec.item():.4f}",flush=True)
        if start<args.codec:
            save(args.codec)
        net.tokenizer.eval().requires_grad_(False)
        tokens,labels=[],[]
        with torch.inference_mode():
            for i in range(0,len(rows),32):
                x,y=batch(rows,range(i,min(i+32,len(rows))),"cuda")
                ids,_,_=net.tokenizer.quantize(x)
                tokens.append(sequence(ids).to(torch.int16).cpu())
                labels.append(y.cpu())
        tokens=torch.cat(tokens).cuda()
        labels=torch.cat(labels).cuda()
        for step in range(max(args.codec,start)+1,args.steps+1):
            rng=np.random.default_rng([310,step])
            ix=torch.from_numpy(rng.integers(len(rows),size=24)).cuda()
            target=tokens[ix].long()
            prev=F.pad(target[:,:-1],(1,0),value=VOCAB)
            logits=net.prior(prev,labels[ix])
            loss=F.cross_entropy(logits.transpose(1,2),target)
            po.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.prior.parameters(),1,error_if_nonfinite=True)
            po.step()
            log.writerow([step,"prior","","","","",loss.item(),round(time.monotonic()-begin,2),
                          round(torch.cuda.max_memory_allocated()/2**20,1)])
            if step%500==0 or step==args.steps:
                f.flush()
                print(f"Prior {step}: NLL {loss.item():.3f}",flush=True)
        save(args.steps)


if __name__=="__main__":
    main()
