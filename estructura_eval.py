"""Genera galerías y oculta la identidad de cada columna por tipo antes del recuento."""
import argparse
import csv
import json
from pathlib import Path
import random
from PIL import Image
from anatomy_gallery import build
from compare_checkpoints import NAMES
from gallery_cycles import draw_board
from prepare import CASES

ROOT=Path("estructura")
CFG=json.loads((ROOT/"protocolo.json").read_text(encoding="utf8"))
BASE={"Siluetas · 5.000":CFG["referencia_modelo"],"Plan · 7.500":CFG["plan"]}


def blind(phase,models,final):
    folder=ROOT/phase
    build(phase,models,final,CFG,ROOT)
    seeds=CFG["reserva" if final else "desarrollo"]
    key={}
    rng=random.SystemRandom()
    for case in CASES:
        ordinals=list(range(1,1+len(models)))
        rng.shuffle(ordinals)
        key[case]=[list(models)[i-1] for i in ordinals]
        with Image.open(folder/f"{case}.png") as original:
            records=[]
            for j,seed in enumerate(seeds):
                cells=[]
                y=2*(74+104*j)
                for i in [0,*ordinals]:
                    x=2*(135+200*i)
                    cells.append(original.crop((x,y,x+384,y+192)).resize((192,96),Image.Resampling.NEAREST))
                records.append((case,seed,cells))
        draw_board(folder/f"ciego_{case}.png",["Real · mismo tipo",*[chr(65+i) for i in range(len(models))]],records,NAMES[case],scale=1)
    (folder/"clave.json").write_text(json.dumps(key,ensure_ascii=False,indent=2),encoding="utf8")


def score(phase):
    folder=ROOT/phase
    reading=json.loads((folder/"lectura_ciega.json").read_text(encoding="utf8"))
    key=json.loads((folder/"clave.json").read_text(encoding="utf8"))
    seeds=CFG["reserva" if phase.startswith("final") else "desarrollo"]
    rows=[]
    for case in CASES:
        for j,name in enumerate(key[case]):
            code=chr(65+j)
            bodies,features=reading[case][code]
            assert set(bodies)<=set(seeds) and set(map(int,features))<=set(seeds)
            for s in seeds:
                rows.append([name,case,s,code,int(s in bodies),features.get(str(s),"")])
    with (folder/"recuento.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["modelo","tipo","semilla","columna","cuerpo_en_ambas","rasgo_compartido"])
        w.writerows(rows)
    scores={}
    for name in dict.fromkeys(r[0] for r in rows):
        group=[r for r in rows if r[0]==name]
        scores[name]=[sum(r[4] for r in group),sum(bool(r[5]) for r in group),len(group)]
    (folder/"resumen.json").write_text(json.dumps(scores,ensure_ascii=False,indent=2),encoding="utf8")
    if (folder/"giro_ciego.json").exists():
        turn=json.loads((folder/"giro_ciego.json").read_text(encoding="utf8"))
        counts={}
        for case,names in key.items():
            for j,name in enumerate(names):
                counts[name]=counts.get(name,0)+len(turn[case][chr(65+j)])
        (folder/"giro_resumen.json").write_text(json.dumps(counts,ensure_ascii=False,indent=2),encoding="utf8")
    print(scores)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase")
    p.add_argument("--modelo",nargs=2,action="append",default=[])
    p.add_argument("--reserva",action="store_true")
    p.add_argument("--plan5000",action="store_true")
    p.add_argument("--contar",action="store_true")
    a=p.parse_args()
    if a.contar:
        score(a.phase)
    else:
        base=BASE.copy()
        if a.plan5000:
            base={next(iter(BASE)):next(iter(BASE.values())),
                  "Plan · 5.000":"runs/plan_piezas_5000/step_05000.pt",**BASE}
        blind(a.phase,{**base,**dict(a.modelo)},a.reserva)
