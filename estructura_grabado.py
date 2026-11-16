"""Distribución conjunta de dibujo interior y proporciones, ajustada solo a TRAIN."""
from pathlib import Path

import torch
from torch.nn import functional as F


@torch.no_grad()
def main():
    root=Path("estructura/anotacion_train")
    data=torch.load("estructura/train64.pt",mmap=True,weights_only=True)
    prior=torch.load(root/"prior.pt",weights_only=True)
    features=[]
    for chunk in data["images"].split(64):
        x=chunk.cuda().float()/255
        mask=(x.amin(2)<.96).float()
        lum=(x*x.new_tensor([.30,.59,.11])[None,None,:,None,None]).sum(2)
        inner=-F.max_pool2d(-mask,5,1,2)
        mean=(lum*inner).sum((2,3))/inner.sum((2,3)).clamp_min(1)
        diff=(lum-mean[:,:,None,None])*inner
        # Ondas suaves, sin copiar el borde ni los ojos de un caso individual.
        density=F.avg_pool2d(inner,4)
        gray=F.avg_pool2d(diff,4)/density.clamp_min(.16)
        gray=F.avg_pool2d(gray,3,1,1)
        gray=F.interpolate(gray,(10,10),mode="bilinear",align_corners=False)
        width=mask.amax(2).sum(2)/58
        height=mask.amax(3).sum(2)/58
        top=mask[:,:,:32].sum((2,3))/mask.sum((2,3)).clamp_min(1)
        wide=mask[:,:,:,16:48].sum((2,3))/mask.sum((2,3)).clamp_min(1)
        geom=torch.cat([width,height,top,wide],1)
        features.append(torch.cat([gray.flatten(1)*4,(geom-.5)*2],1))
    feat=torch.cat(features)
    group=prior["groups"].cuda()
    import numpy as np
    ids=torch.tensor(np.unique(data["species"],return_inverse=True)[1],device=feat.device)
    w=1./torch.bincount(ids)[ids]
    means=[]; basis=[]
    for k in range(8):
        sub=feat[group==k]
        weights=w[group==k]
        weights=weights/weights.sum()
        mean=(sub*weights[:,None]).sum(0)
        delta=sub-mean
        cov=(delta*weights[:,None]).T@delta
        values,vectors=torch.linalg.eigh(cov)
        means.append(mean)
        basis.append(vectors[:,-16:].T*values[-16:].clamp_min(0).sqrt()[:,None])
    torch.save(dict(data=data["data"],surface_mean=torch.stack(means).cpu(),
                    surface_basis=torch.stack(basis).cpu()),root/"superficie.pt")


if __name__=="__main__":
    main()
