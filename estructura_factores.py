"""Compara variar exclusivamente z o exclusivamente tipos. Lee las mismas muestras de la reserva."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image
from compare_checkpoints import shape_mask
from gallery_cycles import draw_board
from estructura_eval import CFG,ROOT
from prepare import CASES


def main():
    final=ROOT/"final"
    pages=[final,*[p.parent for p in sorted(final.glob("*/modelos.json"))]]
    sources={}
    for folder in pages:
        names=json.loads((folder/"modelos.json").read_text(encoding="utf8"))
        for k,name in enumerate(names):
            sources[name]=(folder,k+1)
    seeds=CFG["reserva"]
    rows=[]
    all_pairs={}
    for name,(folder,col) in sources.items():
        masks={}
        grid={}
        for case in CASES:
            with Image.open(folder/f"{case}.png") as page:
                for i,s in enumerate(seeds):
                    x,y=2*(135+200*col),2*(74+104*i)
                    im=page.crop((x,y,x+384,y+192)).resize((192,96),Image.Resampling.NEAREST)
                    real=page.crop((270,y,654,y+192)).resize((192,96),Image.Resampling.NEAREST)
                    grid[case,s]=(real,im)
                    masks[case,s]=[shape_mask(im.crop((j,0,j+96,96))) for j in (0,96)]
        for a,b in itertools.combinations(grid,2):
            if a[0]!=b[0] and a[1]!=b[1]:
                continue
            factor="z" if a[0]==b[0] else "tipos"
            for v,(m,k) in enumerate(zip(masks[a],masks[b])):
                if m.any() and k.any():
                    rows.append([name,factor,a[0],a[1],b[0],b[1],v,float((m&k).sum()/(m|k).sum())])
        all_pairs[name]=grid
    with (final/"variacion.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["modelo","variable","tipo1","semilla1","tipo2","semilla2","vista","IoU_centrada"])
        w.writerows(rows)
    summary={name:{k:round(float(np.mean([r[-1] for r in rows if r[0]==name and r[1]==k])),3) for k in ("z","tipos")} for name in sources}
    (final/"variacion.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
    names=list(json.loads((final/"modelos.json").read_text(encoding="utf8")))
    for label,keys in (("solo_z",[("water",s) for s in seeds]),("solo_tipos",[(c,seeds[0]) for c in CASES]+[(c,seeds[1]) for c in CASES])):
        records=[(c,s,[all_pairs[names[0]][c,s][0]]+[all_pairs[n][c,s][1] for n in names]) for c,s in keys]
        draw_board(final/f"{label}.png",["Real · mismo tipo",*names],records,
                   "Tipo fijo · ocho semillas" if label=="solo_z" else "Semilla fija · cuatro condiciones")


if __name__=="__main__":
    main()
