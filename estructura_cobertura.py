"""Cobertura condicional en 102 especies VAL con un presupuesto fijo de 24 muestras."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import LATENT
from prepare import DATA, PREPARED, TYPES, read_rows
from sample_cycles import load_generator
from train_estructura import canonical


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("--modelo",nargs=2,action="append",required=True)
    a=p.parse_args()
    split={r["pair_id"]:r["split"] for r in csv.DictReader((PREPARED/"splits.csv").open())}
    species={}
    for r in sorted(read_rows(),key=lambda r:r["pair_id"]):
        if split[r["pair_id"]]=="val":
            species.setdefault(r["species_id"],r)
    ys=[]; images=[]
    for r in species.values():
        views=[]
        for key in ("front_path","back_path"):
            with Image.open(DATA/r[key]) as im:
                views.append(np.array(canonical(im)))
        images.append(views)
        ys.append(tuple(int(r["type_"+t]) for t in TYPES))
    real=torch.from_numpy(np.array(images)).cuda().permute(0,1,4,2,3).float()/255
    mask=real.amin(2)<.96
    seeds=list(range(630011,630035))
    z=torch.stack([torch.randn(LATENT,generator=torch.Generator().manual_seed(s)) for s in seeds]).cuda()
    rows=[]
    for name,path in a.modelo:
        net=load_generator(path,"cuda")
        for types in sorted(set(ys)):
            y=z.new_tensor(types)[None].expand(len(z),-1)
            fake=torch.cat([net(u,c) for u,c in zip(z.split(6),y.split(6))]).unflatten(1,(2,3))[:,:,:,16:80,16:80]/2+.5
            fg=fake.amin(2)<.96
            for i,actual in enumerate(ys):
                if actual!=types:
                    continue
                iou=(fg&mask[i]).sum((2,3))/(fg|mask[i]).sum((2,3)).clamp_min(1)
                err=((fake-real[i]).abs()*mask[i,:,None]).sum((2,3,4))/(3*mask[i].sum((1,2))).clamp_min(1)
                distance=(1-iou+.6*err).mean(1)
                j=int(distance.argmin())
                rows.append([name,list(species)[i],seeds[j],*iou[j].tolist(),*err[j].tolist(),distance[j].item()])
    a.out.mkdir(exist_ok=True,parents=True)
    with (a.out/"cobertura.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["modelo","especie_val","semilla","IoU_frente","IoU_espalda","MAE_frente","MAE_espalda","distancia"])
        w.writerows(rows)
    stats={name:np.mean([r[3:] for r in rows if r[0]==name],0).tolist() for name,_ in a.modelo}
    (a.out/"cobertura.json").write_text(json.dumps(dict(especies=len(species),semillas=seeds,medias=stats),ensure_ascii=False,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
