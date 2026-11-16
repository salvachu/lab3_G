"""Vecinos TRAIN por pareja tras igualar escala. Contrasta con especies apartadas."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from torch.nn import functional as F
from compare_checkpoints import font
from prepare import DATA,CASES,PREPARED,read_rows
from train import train_rows
from train_estructura import canonical
from estructura_eval import CFG,ROOT


def features(x):
    x=x.to("cuda",dtype=torch.float32)/255
    n=len(x)
    x=x.flatten(0,1)
    mask=(x.amin(1,keepdim=True)<.96).float()
    def pool(t):
        return F.avg_pool2d(t,2).reshape(n,-1)
    return torch.cat([pool(mask),pool((1-x))*0.7],1)


def normalized(pair):
    return np.stack([np.array(canonical(pair.crop((j,0,j+96,96)))).transpose(2,0,1) for j in (0,96)])


def pair_image(row):
    pair=Image.new("RGB",(192,96),"white")
    for j,key in enumerate(("front_path","back_path")):
        with Image.open(DATA/row[key]) as im:
            pair.paste(im,(96*j,0))
    return pair


def check(phase):
    folder=ROOT/phase
    if (folder/"protocolo.json").exists():
        protocol=json.loads((folder/"protocolo.json").read_text(encoding="utf8"))
        names=list(protocol["modelos"])
        cases=list(protocol["referencias"])
        seeds=protocol["semillas"]
    else:
        names=list(json.loads((folder/"modelos.json").read_text(encoding="utf8")))
        cases, seeds=CASES, CFG["reserva"]
    allrows=read_rows()
    rowhash={(Path(r["front_path"]).stem,Path(r["back_path"]).stem) for r in allrows}
    rows=train_rows()
    split={r["pair_id"]:r["split"] for r in csv.DictReader((PREPARED/"splits.csv").open())}
    train_hash={r["pair_id"] for r in rows}
    assert all(split[r["pair_id"]]=="train" for r in rows)
    ids=[]; images=[]
    for case in cases:
        with Image.open(folder/f"{case}.png") as sheet:
            for i,s in enumerate(seeds):
                for j,name in enumerate(names):
                    x,y=2*(335+200*j),2*(74+104*i)
                    images.append(sheet.crop((x,y,x+384,y+192)).resize((192,96),Image.Resampling.NEAREST))
                    ids.append((name,case,s))
    n=len(ids)
    # Un sprite real por especie de validación, elegido por ID fijo.
    species=set()
    for r in sorted(allrows,key=lambda row:row["pair_id"]):
        if split[r["pair_id"]]=="val" and r["species_id"] not in species:
            species.add(r["species_id"])
            images.append(pair_image(r))
            ids.append(("VAL",r["species_name"],r["pair_id"]))
    q=features(torch.from_numpy(np.stack([normalized(im) for im in images])))
    state=torch.load(ROOT/"train64.pt",weights_only=True,mmap=True)
    best=q.new_full((len(q),),float("inf"))
    index=torch.zeros(len(q),dtype=torch.long,device=q.device)
    for j in range(0,len(rows),64):
        t=state["images"][j:j+64]
        for mirrored in (False,True):
            f=features(t.flip(-1) if mirrored else t)
            d,ind=torch.cdist(q,f).min(1)
            index=torch.where(d<best,ind+j,index)
            best=torch.minimum(best,d)
    dis=(best/q.shape[1]**.5).cpu().numpy()
    hits=index.tolist()
    threshold=float(np.percentile(dis[n:],5))
    report={}
    with (folder/"vecinos.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["modelo","tipo_o_especie","semilla_o_id","exacta_dataset","distancia","TRAIN_id","TRAIN_especie"])
        for k,(meta,im) in enumerate(zip(ids,images)):
            hashes=tuple(hashlib.sha256(im.crop((j,0,j+96,96)).tobytes()).hexdigest() for j in (0,96))
            r=rows[hits[k]]
            assert r["pair_id"] in train_hash
            w.writerow([*meta,int(hashes in rowhash),float(dis[k]),r["pair_id"],r["species_name"]])
    board=Image.new("RGB",(596,55+103*2*len(names)),"white")
    d=ImageDraw.Draw(board)
    d.text((12,7),"Dos parejas más cercanas a TRAIN por modelo",font=font(18,True),fill="#253d4c")
    d.text((163,33),"Generada",font=font(14),fill="#253d4c")
    d.text((380,33),"Vecino real de TRAIN",font=font(14),fill="#253d4c")
    for j,name in enumerate(names):
        sel=[k for k in range(n) if ids[k][0]==name]
        nearest=sorted(sel,key=lambda k:dis[k])
        report[name]=dict(minimo=float(dis[nearest[0]]),mediana=float(np.median(dis[sel])),bajo_umbral=int((dis[sel]<threshold).sum()))
        for i,k in enumerate(nearest[:2]):
            top=55+103*(j*2+i)
            d.text((6,top+23),name,font=font(13,True),fill="#253d4c")
            d.text((6,top+42),f"{ids[k][1]} {ids[k][2]}",font=font(12),fill="#253d4c")
            d.text((6,top+61),rows[hits[k]]["species_name"],font=font(12),fill="#253d4c")
            board.paste(images[k],(163,top))
            board.paste(pair_image(rows[hits[k]]),(380,top))
    board.save(folder/"parecidos_train.png")
    (folder/"vecinos_resumen.json").write_text(json.dumps(dict(val_especies=len(species),p05_val=threshold,
        mediana_val=float(np.median(dis[n:])),modelos=report),ensure_ascii=False,indent=2),encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase")
    check(p.parse_args().phase)
