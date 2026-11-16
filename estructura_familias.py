"""Asigna familias por forma conjunta y estima sus frecuencias y color en TRAIN."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def fit_family_statistics(net,cache):
    # Cuatro perfiles de referencia del renderizador sin entrenar establecen
    # los nombres de grupo. Se ajusta una mezcla de contornos reales por distancia.
    dev=net.axis.device
    y=cache["y"].to(dev)
    samples=[]
    palettes=[]
    for x in cache["images"].split(64):
        pix=x.to(dev).float()/255
        mask=pix.amin(2)<.96
        f=F.avg_pool2d(mask.float(),4)
        samples.append(f.flatten(1))
        fg=mask.flatten(1)
        axes=pix.transpose(1,2).reshape(len(x),3,-1)
        mean=(axes*fg[:,None]).sum(2)/fg.sum(1).clamp_min(1)[:,None]
        contrast=(axes-mean[:,:,None]).square().mean(1)
        different=fg&(contrast>.055)
        accent=(axes*different[:,None]).sum(2)/different.sum(1).clamp_min(1)[:,None]
        accent=torch.where((different.sum(1)>8)[:,None],accent,mean*.75)
        palettes.append(torch.stack([mean,accent],1))
    samples=torch.cat(samples)
    palettes=torch.cat(palettes)
    z=torch.zeros(4,96,device=dev)
    _,m=net.render(z,y[:4],torch.arange(4,device=dev),True)
    centers=F.avg_pool2d(m,4).flatten(1)
    centers0=centers.clone()
    # Regularizar las medias hacia los perfiles iniciales mantiene los nombres
    # de familia. Son etiquetas geométricas aproximadas, no etiquetas de especie.
    for _ in range(12):
        groups=torch.cdist(samples,centers).argmin(1)
        for k in range(4):
            if (groups==k).any():
                centers[k]=.65*samples[groups==k].mean(0)+.35*centers0[k]
    groups=torch.cdist(samples,centers).argmin(1)
    labels=torch.cat([torch.ones(len(y),1,device=dev),y],1)
    freq=labels.T@F.one_hot(groups,4).float()+4
    net.freq.copy_(freq/freq.sum(1,keepdim=True))
    for j in range(labels.shape[1]):
        for k in range(4):
            take=(labels[:,j]>0)&(groups==k)
            prior=palettes[groups==k].mean(0) if (groups==k).any() else palettes.mean(0)
            net.pigments[j,k]=(palettes[take].sum(0)+8*prior)/(int(take.sum())+8)
    return groups
