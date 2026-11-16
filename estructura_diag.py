"""Contrasta reconstrucción de especies apartadas y generación con tipos y ruido."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from anatomy_gallery import sprites
from compare_checkpoints import font
from diagnose_pairs import rgb_pairs
from prepare import ROOT, PREPARED, read_rows
from sample_cycles import load_generator
from train import batch

OUT=ROOT/"estructura"

@torch.inference_mode()
def main():
    cfg=json.loads((OUT/"protocolo.json").read_text())
    plan=load_generator(ROOT/cfg["plan"],"cuda")
    split={r["pair_id"]:r["split"] for r in csv.DictReader((PREPARED/"splits.csv").open())}
    rows=read_rows()
    train={r["species_id"] for r in rows if split[r["pair_id"]]=="train"}
    val=[r for r in rows if split[r["pair_id"]]=="val"]
    assert not train.intersection(r["species_id"] for r in val)
    bench=json.loads((PREPARED/"benchmark.json").read_text())
    sample=[]
    for case,ids in bench["real_pairs"].items():
        chosen=set()
        for row in sorted((r for r in val if r["pair_id"] in ids),key=lambda r:ids.index(r["pair_id"])):
            if row["species_id"] not in chosen and len(chosen)<2:
                sample.append((row,case))
                chosen.add(row["species_id"])
    sheet=Image.new("RGB",(790,69+104*len(sample)),"white")
    pen=ImageDraw.Draw(sheet)
    pen.text((8,2),"Diagnóstico de Plan · 7.500 en especies apartadas",font=font(19,True),fill="#263e50")
    for j,name in enumerate(["Objetivo real","Sus propios códigos","Generación libre"]):
        pen.text((148+210*j,35),name,font=font(16,True),fill="#263e50")
    measures=[]
    for i in range(0,len(val),32):
        real,y=batch(val,range(i,min(i+32,len(val))),"cuda")
        ids,_,_=plan.tokenizer.quantize(real)
        pred=plan.tokenizer.decode(ids,y)[0]
        real,pred=[x.reshape(-1,2,3,96,96) for x in (real,pred)]
        a,b=real.amin(2)<0.92,pred.amin(2)<0.92
        iou=(a&b).sum((2,3))/(a|b).sum((2,3)).clamp_min(1)
        mae=(((real-pred).abs()/2)*a[:,:,None]).sum((2,3,4))/(3*a.sum((2,3))).clamp_min(1)
        measures.extend(torch.cat([iou,mae],1).cpu().tolist())
    for i,(row,case) in enumerate(sample):
        real,y=batch([row],[0],"cuda")
        ids,_,_=plan.tokenizer.quantize(real)
        recon=plan.tokenizer.decode(ids,y)[0]
        seed=cfg["desarrollo"][i%8]
        cells=[rgb_pairs(real)[0],rgb_pairs(recon)[0],sprites(plan,case,[seed])[0]]
        top=63+104*i
        pen.text((7,top+27),row["species_name"].capitalize(),font=font(16),fill="#263e50")
        pen.text((7,top+49),f"z = {seed}",font=font(13),fill="#263e50")
        for j,img in enumerate(cells):
            sheet.paste(img,(148+210*j,top))
    sheet.save(OUT/"diagnostico_plan.png")
    (OUT/"diagnostico_plan.json").write_text(json.dumps(dict(validacion=len(val),especies=len({r["species_id"] for r in val}),
                medidas=["IoU frente","IoU espalda","MAE figura frente [0,1]","MAE figura espalda [0,1]"],
                promedios=np.mean(measures,0).tolist(),filas=[r["pair_id"] for r,_ in sample]),indent=2))

if __name__=="__main__":
    main()
