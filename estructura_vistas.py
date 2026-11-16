"""Forma de las dos vistas y elecciones del prior, sin usar ningún sprite para generar."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from compare_checkpoints import shape_mask
from estructura_eval import CFG
from model import LATENT
from prepare import CASES, TYPES
from sample_cycles import load_generator


def main():
    p=argparse.ArgumentParser()
    p.add_argument("folder",type=Path)
    p.add_argument("--reserva",action="store_true")
    a=p.parse_args()
    models=json.loads((a.folder/"modelos.json").read_text(encoding="utf8"))
    seeds=CFG["reserva" if a.reserva else "desarrollo"]
    morph={}
    for name,path in models.items():
        if "familias" not in path and "relieve" not in path:
            continue
        net=load_generator(path,"cpu")
        z=torch.stack([torch.randn(LATENT,generator=torch.Generator().manual_seed(s)) for s in seeds])
        morph[name]={}
        for case in [*CASES,"electric","bug+poison","dragon+ghost","water+ice"]:
            y=torch.tensor([[float(t in case.split("+")) for t in TYPES]]).expand(len(z),-1)
            morph[name][case]=net.choose(z,y).tolist()
    (a.folder/"eleccion_morfologia.json").write_text(json.dumps({"semillas":seeds,"modelos":morph},indent=2,ensure_ascii=False),encoding="utf8")
    rows=[]
    for case in CASES:
        with Image.open(a.folder/f"{case}.png") as im:
            for i,s in enumerate(seeds):
                for j,name in enumerate(["Real · mismo tipo",*models]):
                    x,y=2*(135+j*200),2*(74+i*104)
                    pair=im.crop((x,y,x+384,y+192)).resize((192,96),Image.Resampling.NEAREST)
                    fr,bk=[shape_mask(pair.crop((d,0,d+96,96))) for d in (0,96)]
                    if not fr.any() or not bk.any():
                        continue
                    flip=bk[:,::-1]
                    iou=(fr&flip).sum()/(fr|flip).sum()
                    direct=(fr&bk).sum()/(fr|bk).sum()
                    rows.append((name,case,s,float(iou),float(direct)))
    with (a.folder/"espejo.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["modelo","tipo","semilla","IoU_frente_espalda_reflejada","IoU_frente_espalda"])
        w.writerows(rows)
    summary={n:[round(float(np.mean([r[j] for r in rows if r[0]==n])),3) for j in (3,4)] for n in dict.fromkeys(r[0] for r in rows)}
    (a.folder/"espejo.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")


if __name__=="__main__":
    main()
