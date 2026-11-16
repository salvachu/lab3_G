"""Contrasta a 0 y a N pasos la capacidad del generador para explicar VAL, a z optimizado."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
from compare_checkpoints import font
from diagnose_pairs import rgb_pairs
from prepare import DATA,PREPARED,TYPES,read_rows
from sample_cycles import load_generator
from train_estructura import canonical


def main():
    p=argparse.ArgumentParser()
    p.add_argument("dest",type=Path)
    p.add_argument("--modelo",nargs=2,action="append",required=True)
    a=p.parse_args()
    rows=read_rows()
    split={r["pair_id"]:r["split"] for r in csv.DictReader((PREPARED/"splits.csv").open())}
    val=sorted((r for r in rows if split[r["pair_id"]]=="val"),key=lambda r:r["pair_id"])
    # Una pareja por especie. El orden de las siete filas mostradas coincide con el diagnóstico de Plan.
    ids=json.loads(Path("estructura/diagnostico_plan.json").read_text())["filas"]
    selected=[next(r for r in val if r["pair_id"]==k) for k in ids]
    used={r["species_id"] for r in selected}
    for r in val:
        if r["species_id"] not in used:
            selected.append(r)
            used.add(r["species_id"])
    sheet=Image.new("RGB",(142+200*(1+len(a.modelo)),56+98*len(ids)),"white")
    pen=ImageDraw.Draw(sheet)
    pen.text((6,3),"Capacidad de ajuste a especies apartadas",font=font(18,True),fill="#253d4c")
    pen.text((145,33),"Objetivo normalizado",font=font(14),fill="#253d4c")
    for j,(name,_) in enumerate(a.modelo):
        pen.text((345+200*j,33),name,font=font(14),fill="#253d4c")
    report={}
    for j,(name,ck) in enumerate(a.modelo):
        net=load_generator(Path(ck),"cuda").requires_grad_(False)
        measures=[]
        for i in range(0,len(selected),8):
            group=selected[i:i+8]
            imgs=[]
            for r in group:
                pair=[]
                for key in ("front_path","back_path"):
                    with Image.open(DATA/r[key]) as im:
                        pair.append(np.array(canonical(im)))
                imgs.append(pair)
            real=torch.tensor(np.array(imgs),device="cuda").permute(0,1,4,2,3)/127.5-1
            mask=(real.amin(2)<.94).float()
            y=torch.tensor([[int(r["type_"+t]) for t in TYPES] for r in group],device="cuda",dtype=torch.float32)
            # Misma búsqueda para los controles y las redes entrenadas. La familia se
            # elige aquí contra el objetivo para medir capacidad; en generación sale solo de z y tipos.
            u=torch.zeros(len(y),32,device="cuda",requires_grad=True)
            if hasattr(net,"choose"):
                with torch.no_grad():
                    fits=[]
                    for k in range(net.freq.shape[1]):
                        _,m=net.render(u,y,torch.full((len(y),),k,device="cuda"))
                        fits.append((m-mask).abs().mean((1,2,3)))
                    family=torch.stack(fits,1).argmin(1)
                def render(h): return net.render(h,y,family)
            else:
                def render(h): return net.render(h,y)
            opt=torch.optim.Adam([u],lr=.12)
            for _ in range(45):
                pred,m=render(u)
                loss=((pred-real).abs()*(1+3*mask[:,:,None])).mean()+2*(m-mask).abs().mean()+.009*u.square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                pred,m=render(u)
                h=m>.5
                rgb_mae=((pred-real).abs()/2*mask[:,:,None]).sum((2,3,4))/(3*mask.sum((2,3))).clamp_min(1)
                iou=(h&mask.bool()).sum((2,3))/(h|mask.bool()).sum((2,3)).clamp_min(1)
                measures.extend(torch.cat([iou,rgb_mae],1).cpu().tolist())
                src,rec=[rgb_pairs(F.pad(t.flatten(1,2),(16,16,16,16),value=1)) for t in (real,pred)]
                for k in range(min(len(ids)-i,len(group))):
                    top=53+98*(i+k)
                    pen.text((8,top+35),group[k]["species_name"].capitalize(),font=font(15),fill="#253d4c")
                    sheet.paste(src[k],(145,top))
                    sheet.paste(rec[k],(345+200*j,top))
        report[name]=np.mean(measures,axis=0).tolist()
    sheet.save(a.dest.with_suffix(".png"))
    a.dest.with_suffix(".json").write_text(json.dumps(dict(especies=len(selected),orden=["IoU frente","IoU espalda","MAE frente","MAE espalda"],
                pasos_ajuste_z=45,medias=report,ids=[r["pair_id"] for r in selected]),indent=2,ensure_ascii=False),encoding="utf8")


if __name__=="__main__":
    main()
