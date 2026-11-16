"""Cambia solo los tipos: el mismo z para las dos vistas y todas las columnas."""
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from anatomy_gallery import sprites
from compare_checkpoints import NAMES
from estructura_eval import CFG
from gallery_cycles import draw_board
from prepare import DATA, PREPARED, condition, read_rows
from sample_cycles import load_generator


def main():
    p=argparse.ArgumentParser()
    p.add_argument("folder",type=Path)
    p.add_argument("--modelo",nargs=2,action="append",required=True)
    p.add_argument("--reserva",action="store_true")
    a=p.parse_args()
    a.folder.mkdir(exist_ok=True,parents=True)
    cases={"electric":"Eléctrico","bug+poison":"Bicho/Veneno","dragon+ghost":"Dragón/Fantasma","water+ice":"Agua/Hielo"}
    NAMES.update(cases)
    split={r["pair_id"]:r["split"] for r in csv.DictReader((PREPARED/"splits.csv").open())}
    allrows=read_rows()
    refs={}
    tiles={}
    for c in cases:
        available=sorted((r for r in allrows if condition(r)==set(c.split("+"))),
                         key=lambda r:(split[r["pair_id"]]!="val",r["pair_id"]))
        row=available[0]
        refs[c]={k:row[k] for k in ("pair_id","species_name")}
        refs[c]["split"]=split[row["pair_id"]]
        tiles[c]=Image.new("RGB",(192,96),"white")
        for j,k in enumerate(("front_path","back_path")):
            with Image.open(DATA/row[k]) as im:
                tiles[c].paste(im,(96*j,0))
    seeds=CFG["portada_final" if a.reserva else "portada_desarrollo"]
    models=dict(a.modelo)
    samples={}
    for name,path in models.items():
        net=load_generator(path,"cuda")
        samples[name]={c:sprites(net,c,seeds) for c in cases}
    records=[(c,s,[tiles[c]]+[samples[name][c][i] for name in models]) for i,s in enumerate(seeds) for c in cases]
    draw_board(a.folder/"comparacion_visual.png",["Real · mismo tipo",*models],records,"Semilla fija · tipos adicionales")
    (a.folder/"referencias.json").write_text(json.dumps(refs,ensure_ascii=False,indent=2),encoding="utf8")
    (a.folder/"modelos.json").write_text(json.dumps(models,ensure_ascii=False,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
