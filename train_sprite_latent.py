"""Train and resume a paired sprite codec followed by its conditional adversarial latent prior."""
import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from generate import render
from model import Discriminator
from sprite_latent import LatentGenerator
from train_estructura import dataset

CASES = ["water","dragon","fighting","fire","electric","ghost","bug+poison","water+dragon"]
SEEDS = [96553,121211,121212]


def encoded_dataset(net, images):
    with torch.no_grad():
        return torch.cat([net.codec.encode(x.float()/127.5-1) for x in images.split(64)])


def codec_loss(real, predicted, logits):
    target = real.reshape(-1,3,64,64)
    mask = (target.amin(1,keepdim=True)<.93).float()
    rgb = predicted.reshape_as(target)
    foreground = ((rgb-target).abs()*mask).sum()/(3*mask.sum()).clamp_min(1)
    strokes = []
    for axis in (2,3):
        diff = target.diff(dim=axis)
        strong = (diff.abs().amax(1,keepdim=True)>.28).float()
        strokes.append(((rgb.diff(dim=axis)-diff).abs()*strong).sum()/(3*strong.sum()).clamp_min(1))
    return (foreground + .45*sum(strokes)/2
            + .55*F.binary_cross_entropy_with_logits(logits.reshape_as(mask),mask))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/latente_pareado"))
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--codec-steps", type=int, default=1800)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--desde", type=Path, help="Checkpoint inicial para una nueva rama")
    parser.add_argument("--contornos", action="store_true", help="Entrenar con contraste de tipos sobre la forma")
    args=parser.parse_args()
    if min(args.steps,args.batch,args.codec_steps)<1:
        parser.error("Indica cantidades positivas")
    args.run.mkdir(exist_ok=True,parents=True)
    torch.manual_seed(1147)
    data=dataset()
    images=data["images"].cuda()
    labels=data["y"].cuda()
    _, sp, ns=np.unique(data["species"],return_inverse=True,return_counts=True)
    _, tc, nt=np.unique(data["y"].numpy(),axis=0,return_inverse=True,return_counts=True)
    weights=ns[sp]**-.5 * nt[tc]**-.30
    weights/=weights.sum()
    net=LatentGenerator().cuda()
    ema=copy.deepcopy(net).eval().requires_grad_(False)
    disc=Discriminator().cuda()
    op=torch.optim.AdamW(net.parameters(),lr=.00085,betas=(.9,.98))
    od=torch.optim.Adam(disc.parameters(),lr=.00016,betas=(0.,.9))
    paths=sorted(args.run.glob("step_*.pt"))
    start=0
    if paths or args.desde:
        state=torch.load(paths[-1] if paths else args.desde,map_location="cpu",weights_only=True)
        if (state["data"],state["batch"],state["codec_steps"])!=(data["data"],args.batch,args.codec_steps):
            raise ValueError("La partición y la configuración deben coincidir con el checkpoint")
        for obj,key in ((net,"training"),(ema,"generator"),(disc,"disc"),(op,"opt"),(od,"dop")):
            obj.load_state_dict(state[key])
        start=state["step"]
        if paths:
            if args.contornos and not state.get("contornos",False):
                raise ValueError("Usa --desde con una carpeta nueva para cambiar el objetivo.")
            args.contornos=state.get("contornos",False)
    if start>=args.steps:
        return
    codes=None
    for step in range(start+1,args.steps+1):
        rng=np.random.default_rng([1147,step])
        idx=rng.choice(len(weights),args.batch,p=weights)
        gen=torch.Generator(device="cuda").manual_seed(5000000+step)
        real=images[idx].float()/127.5-1
        y=labels[idx]
        in_codec=step<=args.codec_steps
        if in_codec:
            u=net.codec.encode(real)
            jitter=torch.randn(u.shape,device="cuda",generator=gen)*.055
            fake,logits=net.codec.decode(u+jitter)
            loss=codec_loss(real,fake,logits)
        else:
            if codes is None:
                net.codec.eval().requires_grad_(False)
                codes=encoded_dataset(net,images)
                if step==args.codec_steps+1:
                    net.mean.copy_(codes.mean((0,2,3),keepdim=True))
                    net.std.copy_(codes.std((0,2,3),keepdim=True).clamp_min(.08))
                ema.codec.load_state_dict(net.codec.state_dict())
                ema.mean.copy_(net.mean)
                ema.std.copy_(net.std)
                op.param_groups[0]["lr"]=.0005
            clean=(codes[idx]-net.mean)/net.std
            noise=torch.randn(clean.shape,device="cuda",generator=gen)
            t=(torch.arange(len(y),device="cuda")+torch.rand(len(y),device="cuda",generator=gen))/len(y)
            theta=t[:,None,None,None]*math.pi/2
            noisy=theta.cos()*clean+theta.sin()*noise
            keep=torch.rand(len(y),1,device="cuda",generator=gen)>.12
            velocity=net.velocity(noisy,t,y*keep)
            target=theta.cos()*noise-theta.sin()*clean
            loss=F.mse_loss(velocity,target)
            if args.contornos:
                # The four occupancy channels in each view encode the real silhouette.
                def shape_error(v):
                    difference=(v-target).reshape(len(y),2,12,8,8)[:,:,:4].reshape(-1,4,8,8)
                    shape=F.pixel_shuffle(difference,2)
                    return sum(F.avg_pool2d(shape,k).square().reshape(len(y),-1).mean(1)
                               for k in (1,2,4))/3
                good=shape_error(velocity)
                loss=loss+.35*good.mean()
                if step%2==0:
                    other=y.roll(1,0)
                    valid=(y*other).sum(1).eq(0)&keep[:,0]&(t>.35)
                    wrong=net.velocity(noisy,t,other)
                    margin=F.relu(.025+good-shape_error(wrong))
                    loss=loss+.20*(margin*valid).sum()/valid.sum().clamp_min(1)
            if step%4==0:
                estimate=(theta.cos()*noisy-theta.sin()*velocity)*net.std+net.mean
                fake,_=net.codec.decode(estimate.clamp(-1,1))
        if step>=250 and (in_codec or step%4==0):
            disc.requires_grad_(True)
            od.zero_grad(set_to_none=True)
            pos,neg=disc(torch.cat([real.flatten(1,2),fake.detach()]),y.repeat(2,1)).chunk(2)
            (F.relu(1-pos).mean()+F.relu(1+neg).mean()).backward()
            od.step()
            disc.requires_grad_(False)
            loss=loss-.018*disc(fake,y).mean()
        op.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),4,error_if_nonfinite=True)
        op.step()
        if not in_codec:
            with torch.no_grad():
                params=[(a,b) for a,b in zip(ema.parameters(),net.parameters()) if b.requires_grad]
                torch._foreach_lerp_([a for a,_ in params],[b for _,b in params],max(.004,1/(step-args.codec_steps)))
        if step%500==0 or step in (args.codec_steps,args.steps):
            if in_codec:
                ema.codec.load_state_dict(net.codec.state_dict())
            dest=args.run/f"step_{step:06}.pt"
            tmp=dest.with_suffix(".tmp")
            torch.save(dict(architecture="paired_spatial_latent",step=step,data=data["data"],batch=args.batch,contornos=args.contornos,
                            codec_steps=args.codec_steps,training=net.state_dict(),generator=ema.state_dict(),
                            disc=disc.state_dict(),opt=op.state_dict(),dop=od.state_dict()),tmp)
            tmp.replace(dest)
            if not in_codec:
                render(ema,CASES,SEEDS,dest.with_suffix(".png"))
            print(step,round(loss.item(),4),flush=True)


if __name__=="__main__":
    main()
