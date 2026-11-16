"""Benchmarks de generación con las referencias y el diseño de la plantilla original."""
import argparse
import hashlib
import json

import torch
from PIL import Image

from compare_checkpoints import NAMES, pairs, silhouette_stats
from diagnose_pairs import rgb_pairs
from gallery_cycles import CASES, draw_board
from model import LATENT
from prepare import PREPARED, ROOT, SEEDS, TYPES
from sample_cycles import load_generator

OUT=ROOT/"anatomia"
PLAN=json.loads((OUT/"protocolo.json").read_text(encoding="utf-8"))
BASE="Siluetas · 5.000"


@torch.inference_mode()
def sprites(model,case,seeds):
    z=torch.stack([torch.randn(LATENT,generator=torch.Generator().manual_seed(seed)) for seed in seeds]).cuda()
    y=torch.tensor([t in case.split("+") for t in TYPES],device=z.device,dtype=torch.float32)
    return rgb_pairs(model(z,y.expand(len(z),-1)))


def build(phase,models,final,protocol=PLAN,output=OUT):
    seeds=protocol["reserva" if final else "desarrollo"]
    cover=protocol["portada_final" if final else "portada_desarrollo"]
    folder=output/phase
    folder.mkdir()
    real=pairs(PREPARED/"real_val_roundtrip.png",CASES,SEEDS)
    grids={}
    fingerprints=[]
    for name,ck in models.items():
        net=load_generator(ROOT/ck,"cuda")
        grids[name]=[sprites(net,case,seeds) for case in CASES]
        for i,case in enumerate(CASES):
            for j,seed in enumerate(seeds):
                pair=grids[name][i][j]
                ids=[hashlib.sha256(pair.crop((k,0,k+96,96)).tobytes()).hexdigest() for k in (0,96)]
                fingerprints.append(dict(model=name,case=case,seed=seed,front=ids[0],back=ids[1]))
    def records(types,chosen):
        return [(CASES[i],s,[real[i][seeds.index(s)%len(SEEDS)]]+[grid[i][seeds.index(s)] for grid in grids.values()])
                for i in types for s in chosen]
    headings=["Real · mismo tipo",*models]
    def save_and_check(path,rows,title):
        draw_board(path,headings,rows,title)
        with Image.open(path) as board:
            for j,(_,_,cells) in enumerate(rows):
                for k,im in enumerate(cells):
                    x,y=2*(135+200*k),2*(74+104*j)
                    assert board.crop((x,y,x+384,y+192)).tobytes()==im.resize((384,192),Image.Resampling.NEAREST).tobytes()
    for i,case in enumerate(CASES):
        save_and_check(folder/f"{case}.png",records([i],seeds),NAMES[case])
    save_and_check(folder/"comparacion_visual.png",records(range(4),cover),"Evolución de los sprites")
    silhouette_stats(grids,CASES,folder/"siluetas.csv")
    (folder/"modelos.json").write_text(json.dumps(models,ensure_ascii=False,indent=2),encoding="utf-8")
    (folder/"huellas.json").write_text(json.dumps(fingerprints,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase")
    p.add_argument("--modelo",nargs=2,action="append",required=True,metavar=("NOMBRE","CHECKPOINT"))
    p.add_argument("--reserva",action="store_true")
    p.add_argument("--noche",action="store_true")
    a=p.parse_args()
    output=ROOT/"noche" if a.noche else OUT
    protocol=json.loads((output/"protocolo.json").read_text(encoding="utf-8"))
    build(a.phase,{BASE:protocol["referencia_modelo"],**dict(a.modelo)},a.reserva,protocol,output)
