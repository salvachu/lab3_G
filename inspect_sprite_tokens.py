"""Control del vocabulario y comprobación de vecinos de las muestras libres."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image,ImageDraw

from anatomy_gallery import OUT, PLAN
from compare_checkpoints import NAMES,font
from diagnose_pairs import rgb_pairs
from prepare import CASES, PREPARED, ROOT, read_rows
from sample_cycles import load_generator
from train import batch,train_rows


@torch.inference_mode()
def codec_check(path):
    net=load_generator(path,"cuda")
    by_id={r["pair_id"]:r for r in read_rows()}
    manifest=json.loads((PREPARED/"benchmark.json").read_text())
    rows=[by_id[k] for case in CASES for k in manifest["real_pairs"][case][:2]]
    real,y=batch(rows,range(8),"cuda")
    ids,_,_=net.tokenizer.quantize(real)
    if hasattr(net,"refine"):
        from model import LATENT
        z=torch.randn(8,LATENT,generator=torch.Generator().manual_seed(6201)).to(y.device)
        pred=net.refine(ids,z,y)
    else:
        pred=net.tokenizer.decode(ids,y)[0]
    sheet=Image.new("RGB",(570,890),"white")
    draw=ImageDraw.Draw(sheet)
    draw.text((9,5),"Diagnóstico del vocabulario en validación",font=font(19,True),fill="#213647")
    draw.text((140,34),"Objetivo real",font=font(14,True),fill="#213647")
    draw.text((353,34),"Reconstrucción",font=font(14,True),fill="#213647")
    for i,(a,b) in enumerate(zip(rgb_pairs(real),rgb_pairs(pred))):
        y0=59+103*i
        draw.text((8,y0+39),rows[i]["species_name"].capitalize(),font=font(15),fill="#213647")
        sheet.paste(a,(140,y0))
        sheet.paste(b,(353,y0))
    sheet.save(path.parent/"codec_val.png")


def nearest(phase,output=OUT,protocol=PLAN):
    folder=output/phase
    registry=json.loads((folder/"modelos.json").read_text(encoding="utf-8"))
    hashes=json.loads((folder/"huellas.json").read_text(encoding="utf-8"))
    known={(Path(r["front_path"]).stem,Path(r["back_path"]).stem) for r in read_rows()}
    names=list(registry)
    crops=[]
    for case in CASES:
        with Image.open(folder/f"{case}.png") as page:
            for j in range(8):
                for k in range(len(names)):
                    x,y=2*(135+200*(1+k)),2*(74+104*j)
                    im=page.crop((x,y,x+384,y+192)).resize((192,96),Image.Resampling.NEAREST)
                    crops.append(np.asarray(im.resize((48,24),Image.Resampling.BOX),dtype=np.float32)/255)
    query=torch.from_numpy(np.stack(crops)).cuda().flatten(1)
    rows=train_rows()
    best=torch.full((len(crops),),float("inf"),device="cuda")
    idx=torch.zeros(len(crops),dtype=torch.long,device="cuda")
    for i in range(0,len(rows),64):
        images,_=batch(rows,range(i,min(i+64,len(rows))),"cuda")
        # La misma reducción de área con redondeo de Pillow para candidatos y consultas.
        small=np.stack([np.asarray(im.resize((48,24),Image.Resampling.BOX),dtype=np.float32)/255 for im in rgb_pairs(images)])
        candidates=torch.from_numpy(small).cuda().flatten(1)
        distance,arg=torch.cdist(query,candidates).min(1)
        idx=torch.where(distance<best,arg+i,idx)
        best=torch.minimum(best,distance)
    finger={(r["model"],r["case"],r["seed"]):(r["front"],r["back"]) for r in hashes}
    seedlist=protocol["reserva"]
    with (folder/"vecinos_train.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["modelo","tipo","semilla","pareja_exacta_dataset","vecino_train","especie","RMSE_RGB_24"])
        k=0
        for case in CASES:
            for s in seedlist:
                for name in names:
                    row=rows[idx[k].item()]
                    w.writerow([name,case,s,int(finger[name,case,s] in known),row["pair_id"],row["species_name"],
                                best[k].item()/query.size(1)**0.5])
                    k+=1


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--codec",type=Path)
    p.add_argument("--vecinos")
    p.add_argument("--noche",action="store_true")
    a=p.parse_args()
    if a.codec:
        codec_check(a.codec)
    if a.vecinos:
        output=ROOT/"noche" if a.noche else OUT
        protocol=json.loads((output/"protocolo.json").read_text(encoding="utf-8"))
        nearest(a.vecinos,output,protocol)
