"""Estima el prior de morfologías y los dos pigmentos usando parejas de TRAIN."""
import torch
from torch.nn import functional as F


def silhouette(x):
    h=F.avg_pool2d(x,4)
    return torch.cat([h.flatten(1),2*h.mean(2).flatten(1),2*h.mean(3).flatten(1)],1)


def centered(m):
    out=[]
    for pair in m:
        views=[]
        for view in pair:
            iy,ix=torch.where(view>.5)
            crop=view[iy.min():iy.max()+1,ix.min():ix.max()+1]
            size=[max(1,round(58*t/max(crop.shape))) for t in crop.shape]
            scaled=F.interpolate(crop[None,None],size=size,mode="bilinear",align_corners=False)[0,0]
            a,b=[(64-x)//2 for x in size]
            views.append(F.pad(scaled,(b,64-b-size[1],a,64-a-size[0])))
        out.append(torch.stack(views))
    return torch.stack(out)


@torch.no_grad()
def fit_relief(net,cache):
    y=cache["y"].cuda()
    geom=[]
    pigments=[]
    for rgb in cache["images"].split(64):
        rgb=rgb.cuda().float()/255
        mask=rgb.amin(2)<.96
        geom.append(silhouette(mask.float()))
        px=rgb.transpose(1,2).flatten(2)
        fg=mask.flatten(1)
        chroma=px.amax(1)-px.amin(1)
        w=fg*(.15+chroma)*(px.mean(1)>.12)
        main=(px*w[:,None]).sum(2)/w.sum(1,keepdim=True).clamp_min(1)
        diff=(px-main[:,:,None]).square().mean(1)
        w=w*diff.square()*(px.amin(1)<.88)
        alt=(px*w[:,None]).sum(2)/w.sum(1,keepdim=True).clamp_min(1e-6)
        pigments.append(torch.stack([main,alt],1))
    geom=torch.cat(geom)
    pigs=torch.cat(pigments)
    ref=[]
    rng=torch.Generator(device="cuda").manual_seed(572)
    for _ in range(12):
        z=torch.randn(8,32,device="cuda",generator=rng)
        _,mask=net.render(z,y[:8],torch.arange(8,device="cuda"),True)
        ref.append(silhouette(centered(mask)))
    ref=torch.stack(ref).mean(0)
    centers=ref.clone()
    for _ in range(6):
        cost=torch.cdist(geom,centers).square()/geom.shape[1]
        logq=-cost/.019
        for _ in range(30):
            logq-=logq.logsumexp(0,keepdim=True)
            logq-=logq.logsumexp(1,keepdim=True)
        q=logq.exp()
        centers=.48*ref+.52*(q.T@geom)/q.sum(0)[:,None]
    labels=torch.cat([torch.ones_like(y[:,:1]),y],1)
    # La tabla de frecuencias incorpora la asignación blanda, evitando saltos por empates.
    tab=labels.T@q+1.
    net.freq.copy_(tab/tab.sum(1,keepdim=True))
    for k in range(8):
        w=labels*q[:,k:k+1]
        prior=(pigs*q[:,k,None,None]).sum(0)/q[:,k].sum()
        net.colors[:,k]=((w.T@pigs.flatten(1)).reshape(-1,2,3)+4*prior)/(w.sum(0)[:,None,None]+4)
    return q.argmax(1)
