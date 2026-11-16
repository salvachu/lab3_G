"""GAN de pares con reconstrucción variacional auxiliar solo durante entrenamiento."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import Discriminator, Generator, LATENT
from prepare import ROOT, DATA, PREPARED
from train import batch, train_rows
from train_silhouettes import hinge, update


class PairEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for src, dst in [(6, 32), (32, 48), (48, 64), (64, 96)]:
            blocks.extend([nn.Conv2d(src, dst, 4, 2, 1), nn.GroupNorm(8, dst), nn.LeakyReLU(0.2)])
        self.body = nn.Sequential(*blocks, nn.Flatten())
        self.output = nn.Linear(96*6*6, LATENT*2)

    def forward(self, x):
        mu, log_var = self.output(self.body(x)).chunk(2, dim=1)
        return mu, log_var.clamp(-7, 3)


def reconstruction(real, predicted):
    original = real.reshape(-1, 3, 96, 96)
    recon = predicted.reshape_as(original)
    fg = (original.amin(1, keepdim=True) < 0.76).float()
    errors = (original - recon).abs().mean(1, keepdim=True)
    def region(mask):
        return ((errors * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)).mean()
    coverage = ((0.76 - recon.amin(1, keepdim=True))/0.07).sigmoid()
    dice = (2*(fg*coverage).sum((2, 3))+1) / (fg.sum((2, 3))+coverage.sum((2, 3))+1)
    return 0.5*(region(fg)+region(1-fg)) + 0.5*(1-dice.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--recon-weight", type=float, default=8.0)
    args=p.parse_args()
    state=torch.load(args.resume,map_location="cpu",weights_only=True)
    seed, size, start = state["seed"], state["batch"], state["step"]
    signature=hashlib.sha256((PREPARED/"splits.csv").read_bytes()+(DATA/"pairs.csv").read_bytes()).hexdigest()
    if state["data"]!=signature or args.steps<=start or args.run.exists():
        p.error("Revisar firma de datos, destino nuevo y paso total")
    torch.manual_seed(seed)
    rows=train_rows()
    g,d,e=Generator().cuda(), Discriminator().cuda(), PairEncoder().cuda()
    opts=[torch.optim.Adam(m.parameters(),lr=0.0002,betas=(0.0,0.9)) for m in (g,d,e)]
    go,do,eo=opts
    for m,opt,model_key,opt_key in [(g,go,"generator","go"),(d,do,"discriminator","do"),(e,eo,"encoder","eo")]:
        if model_key in state:
            m.load_state_dict(state[model_key])
            opt.load_state_dict(state[opt_key])
    args.run.mkdir(parents=True)
    config=dict(vars(args), initial_step=start, data=signature, batch=size, seed=seed, kl_weight=0.3,
                initial_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name(),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.run/"config.json").write_text(json.dumps(config,default=str,indent=2))
    begin=time.monotonic()
    with (args.run/"metrics.csv").open("w",newline="") as f:
        log=csv.writer(f)
        log.writerow(["step","d","g_adversarial","reconstruction","kl","seconds","peak_gpu_MiB"])
        for step in range(start+1,args.steps+1):
            rng=np.random.default_rng([seed,step])
            real,y=batch(rows,rng.integers(len(rows),size=size),"cuda")
            noise_rng=torch.Generator(device="cuda").manual_seed(int(rng.integers(2**62)))
            z=torch.randn(size,LATENT,device="cuda",generator=noise_rng)
            fake=g(z,y)
            d.requires_grad_(True)
            do.zero_grad(set_to_none=True)
            dl=hinge(d(torch.cat([real,fake.detach()]),torch.cat([y,y])))
            update(dl,d,do)
            d.requires_grad_(False)
            go.zero_grad(set_to_none=True)
            eo.zero_grad(set_to_none=True)
            adversarial=-d(fake,y).mean()
            mu,lv=e(real)
            eps=torch.randn(size,LATENT,device="cuda",generator=noise_rng)
            rec=reconstruction(real,g(mu+(0.5*lv).exp()*eps,y))
            kl=0.5*(mu.square()+lv.exp()-1-lv).mean()
            loss=adversarial+args.recon_weight*rec+0.3*kl
            loss.backward()
            for m,opt in ((g,go),(e,eo)):
                nn.utils.clip_grad_norm_(m.parameters(),float("inf"),error_if_nonfinite=True)
                opt.step()
            log.writerow([step,dl.item(),adversarial.item(),rec.item(),kl.item(),
                          round(time.monotonic()-begin,2),round(torch.cuda.max_memory_allocated()/2**20,1)])
            if step%500==0 or step==args.steps:
                f.flush()
                print(f"{step}: recon {rec.item():.3f}, KL {kl.item():.3f}",flush=True)
            if step==args.steps:
                ck=args.run/f"step_{step:05}.pt"
                tmp=ck.with_suffix(".tmp")
                torch.save(dict(step=step,data=signature,seed=seed,batch=size,generator=g.state_dict(),
                                discriminator=d.state_dict(),encoder=e.state_dict(),go=go.state_dict(),
                                do=do.state_dict(),eo=eo.state_dict()),tmp)
                tmp.replace(ck)


if __name__=="__main__":
    main()
