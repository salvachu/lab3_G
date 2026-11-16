"""Aprende distribuciones de topología y pigmentos con supervisión parcial de TRAIN."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from prepare import TYPES
from sprite_boceto import Encoder
from train import train_rows
from train_estructura import dataset

CLASSES="aqsfubmh"
PALETTES=24


def main():
    out=Path("estructura/anotacion_train")
    items=json.loads((out/"filas.json").read_text())
    labels=(out/"formas.txt").read_text().split()
    assert len(items)==len(labels)==320
    data=dataset()
    rows=train_rows()
    for row in items:
        assert row["pair_id"]==rows[row["indice_train"]]["pair_id"]
    chosen=torch.tensor([r["indice_train"] for r,c in zip(items,labels) if c!="x"],device="cuda")
    label=torch.tensor([CLASSES.index(c) for c in labels if c!="x"],device="cuda")
    annotation=F.one_hot(label,8).float()
    encoder=Encoder().cuda().eval()
    encoder.load_state_dict(torch.load("runs/estructura_boceto/step_05000.pt",map_location="cpu",weights_only=True)["encoder"])
    palette=torch.load("runs/estructura_boceto/objetivos.pt",mmap=True,weights_only=True)
    y=data["y"].cuda()
    features=[]; pigments=[]
    with torch.no_grad():
        for i in range(0,len(y),64):
            x=data["images"][i:i+64].cuda().float()/127.5-1
            mask=(x.amin(2)<.92).float()
            code=F.normalize(encoder(x,mask))
            form=F.avg_pool2d(mask,8).flatten(1)
            features.append(torch.cat([code,F.normalize(form-form.mean(1,keepdim=True))*.7],1))
            pal=palette["pal"][i:i+64].cuda()/2+.5
            ids=palette["labels"][i:i+64].cuda().flatten(1).long()
            freq=x.new_zeros(len(x),9).scatter_add_(1,ids,torch.ones_like(ids,dtype=x.dtype))[:,1:]
            freq=freq/freq.sum(1,keepdim=True).clamp_min(1)
            chroma=pal.amax(2)-pal.amin(2)
            lum=pal.mean(2)
            usable=((lum>.18)&(lum<.91)).float()
            salience=(freq+.005).sqrt()*(.18+chroma)*(.10+usable)
            idx=salience.argmax(1)
            main=pal[torch.arange(len(x)),idx]
            difference=(pal-main[:,None]).abs().mean(2)
            idx=(salience.sqrt()*difference*(.25+usable)).argmax(1)
            accent=pal[torch.arange(len(x)),idx]
            pigments.append(torch.cat([main,accent],1))
    features=torch.cat(features)
    colors=torch.cat(pigments)
    species=torch.tensor(np.unique(data["species"],return_inverse=True)[1],device="cuda")
    dist=torch.cdist(features,features[chosen]).square()
    post=(-dist/.14).softmax(1)@annotation
    lookup=torch.full((int(species.max())+1,),-1,device="cuda",dtype=torch.long)
    lookup[species[chosen]]=label
    assigned=lookup[species]
    known=assigned>=0
    post[known]=F.one_hot(assigned[known],8).float()
    groups=post.argmax(1)
    species_w=1./torch.bincount(species)[species]
    weights=species_w*torch.where(known,1.,.16)
    prior=(post*weights[:,None]).sum(0)
    prior=(prior+1)/(prior.sum()+8)
    # Distribución de pigmentos aprendida conjuntamente para las dos vistas.
    rng=torch.Generator(device="cuda").manual_seed(4729)
    centers=colors[torch.randperm(len(y),generator=rng,device="cuda")[:PALETTES]].clone()
    for _ in range(30):
        cid=torch.cdist(colors,centers).argmin(1)
        mass=colors.new_zeros(PALETTES).scatter_add_(0,cid,species_w)
        means=torch.zeros_like(centers).index_add_(0,cid,colors*species_w[:,None])/mass.clamp_min(.001)[:,None]
        centers=torch.where(mass[:,None]>0,means,centers)
    col=F.one_hot(cid,PALETTES).float()
    color_prior=(col*species_w[:,None]).sum(0)+1
    color_prior/=color_prior.sum()
    types=len(TYPES)
    morph=colors.new_empty(types,types,8)
    pigments=colors.new_empty(types,types,8,PALETTES)
    for i in range(types):
        for j in [i,*range(i)]:
            both=y[:,i]*y[:,j]
            def tissue(w):
                ph=(post*w[:,None]).sum(0)+1.8*prior
                return ph/ph.sum()
            def ink(w):
                tab=(post*w[:,None]).T@col+1.0*color_prior
                return tab/tab.sum(1,keepdim=True)
            if i==j:
                m=tissue(weights*both)
                ink_prob=ink(species_w*both)
            else:
                support=(weights*both).sum()
                mix=support/(support+5)
                m=(morph[i,i]*morph[j,j]/prior).clamp_min(.0001)
                m=m/m.sum()
                m=m*(1-mix)+tissue(weights*both)*mix
                cmix=(species_w*both).sum()/((species_w*both).sum()+12)
                back=(pigments[i,i]*pigments[j,j]/color_prior).clamp_min(.00001).sqrt()
                ink_prob=back/back.sum(1,keepdim=True)*(1-cmix)+ink(species_w*both)*cmix
            morph[i,j]=morph[j,i]=m
            pigments[i,j]=pigments[j,i]=ink_prob
    # Evaluación interna leave-one-out: inferir solo la morfología de una especie TRAIN anotada por sus tipos.
    ya=y[chosen]; correct=0; naive=0; nll=0.; nll0=0.
    for i in range(len(chosen)):
        keep=torch.arange(len(chosen),device="cuda")!=i
        tags=ya[i].nonzero().flatten()
        base=annotation[keep].sum(0)+1
        base/=base.sum()
        pp=base.clone()
        for k in tags:
            tab=(ya[keep,k:k+1]*annotation[keep]).sum(0)+3*base
            pp*=tab/tab.sum()/base
        pp/=pp.sum()
        correct+=int(pp.argmax()==label[i]); naive+=int(base.argmax()==label[i])
        nll-=float(pp[label[i]].log()); nll0-=float(base[label[i]].log())
    assert torch.isfinite(morph).all() and torch.isfinite(pigments).all()
    assert torch.allclose(morph.sum(-1),torch.ones_like(morph[:,:,0]),atol=.0001)
    assert torch.allclose(pigments.sum(-1),torch.ones_like(pigments[:,:,:,0]),atol=.0001)
    torch.save(dict(data=data["data"],groups=groups.cpu(),topo=morph.cpu(),ink=pigments.cpu(),tones=centers.reshape(PALETTES,2,3).cpu()),out/"prior.pt")
    metadata=dict(anotadas=len(chosen),ambiguas=len(items)-len(chosen),clases=CLASSES,
                  soporte=annotation.sum(0).long().tolist(),leave_one_out_top1=correct/len(chosen),global_top1=naive/len(chosen),
                  entropia_cruzada=nll/len(chosen),entropia_global=nll0/len(chosen))
    (out/"resumen.json").write_text(json.dumps(metadata,indent=2),encoding="utf8")
    print(metadata)


if __name__=="__main__":
    main()
