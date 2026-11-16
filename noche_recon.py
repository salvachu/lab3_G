"""Diagnóstico en VAL: separa el dibujo interior del contorno que lo guía."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw

from compare_checkpoints import font
from diagnose_pairs import rgb_pairs
from prepare import CASES,PREPARED,read_rows
from sample_cycles import load_generator
from train import batch
from model import LATENT


@torch.inference_mode()
def diagnose(ck):
    net=load_generator(ck,"cuda")
    manifest=json.loads((PREPARED/"benchmark.json").read_text())
    by_id={r["pair_id"]:r for r in read_rows()}
    rows=[by_id[key] for case in CASES for key in manifest["real_pairs"][case][:2]]
    real,y=batch(rows,range(len(rows)),"cuda")
    ids,_,_=net.tokenizer.quantize(real)
    coarse=net.tokenizer.decode(ids,y)[0]
    pair=coarse.reshape(-1,2,3,96,96)
    mask=(real.reshape_as(pair).amin(2)<0.92).float()
    groups=torch.cdist(F.avg_pool2d(mask,8).flatten(1),net.means.flatten(1)).argmin(1)
    mean=net.means[groups]
    cond=net.contour.condition(groups,y)
    mu,_=net.contour.encode(F.avg_pool2d(mask,2),mean,cond)
    logits=net.contour.decode(mu,mean,cond)
    estimated=(2*F.max_pool2d(F.interpolate(logits,scale_factor=2,mode="bilinear",align_corners=False),3,1,1)).sigmoid()
    columns=[("Objetivo",real),("Decodificación VQ",coarse),("Tinta · máscara real",net.paint(pair,mask)),
             ("Tinta · máscara reconstruida",net.paint(pair,estimated))]
    masks={"contorno":estimated}
    if hasattr(net,"fine_mask"):
        z=torch.randn(len(rows),LATENT,generator=torch.Generator().manual_seed(6201)).cuda()
        fine=net.fine_mask(pair,estimated,z)
        columns.append(("Borde · máscara refinada",net.paint(pair,fine)))
        masks["borde"]=fine
    ious={}
    for name,prob in masks.items():
        binary=prob>0.5
        ious[name]=((binary&mask.bool()).sum((2,3))/(binary|mask.bool()).sum((2,3)).clamp_min(1)).tolist()
    (ck.parent/"mascaras_val.json").write_text(json.dumps(dict(checkpoint=ck.name,pairs=[r["pair_id"] for r in rows],iou=ious),indent=2))
    sheet=Image.new("RGB",(117+211*len(columns),904),"white")
    d=ImageDraw.Draw(sheet)
    d.text((8,4),"Diagnóstico de reconstrucción en validación",font=font(19,True),fill="#253d4c")
    d.text((8,31),"Códigos extraídos del objetivo. El relleno y la paleta se mantienen al cambiar la máscara.",font=font(14),fill="#253d4c")
    for j,(name,data) in enumerate(columns):
        x=117+j*211
        d.text((x,62),name,font=font(13,True),fill="#253d4c")
        for i,img in enumerate(rgb_pairs(data)):
            top=86+101*i
            if j==0:
                d.text((8,top+34),rows[i]["species_name"].capitalize(),font=font(15),fill="#253d4c")
            sheet.paste(img,(x,top))
    sheet.save(ck.parent/"diagnostico_val.png")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint",type=Path)
    diagnose(p.parse_args().checkpoint)
