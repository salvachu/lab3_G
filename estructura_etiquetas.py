"""Muestra de TRAIN por especie para anotar topología sin ver la especie ni los tipos."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from train import train_rows


def main():
    root=Path("estructura/anotacion_train")
    root.mkdir(exist_ok=True)
    data=torch.load("estructura/train64.pt",mmap=True,weights_only=True)
    rows=train_rows()
    species=np.array(data["species"])
    rng=np.random.default_rng(2271)
    chosen=rng.permutation(np.unique(species))[:320]
    ids=[]
    for count,s in enumerate(chosen):
        if count%64==0:
            sheet=Image.new("RGB",(8*134,8*86),"white")
            pen=ImageDraw.Draw(sheet)
        i=int(rng.choice(np.flatnonzero(species==s)))
        ids.append({"indice_train":i,"pair_id":rows[i]["pair_id"]})
        row,col=divmod(count%64,8)
        x,y=col*134,row*86
        pen.text((x+3,y),str(count+1),fill="black")
        for v,img in enumerate(data["images"][i]):
            sheet.paste(Image.fromarray(img.permute(1,2,0).numpy()),(x+64*v,y+16))
        if count%64==63:
            sheet.save(root/f"hoja_{count//64+1}.png")
    (root/"filas.json").write_text(json.dumps(ids,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
