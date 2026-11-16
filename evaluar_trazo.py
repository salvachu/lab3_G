"""Compare held-out reconstruction and free generations for every type separately."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diagnose_pairs import rgb_pairs
from evaluar_perfiles import NAMES, SPANISH, draw_comparison, labels, pixel_reference
from evaluar_tipos_duales import reference
from gallery_cycles import draw_board
from model import LATENT
from prepare import TYPES, read_rows
from sample_cycles import load_generator
from train_sprite_latent import CASES
from train_trazo import degrade, images


@torch.inference_mode()
def main():
    out=Path("estructura/trazo")
    out.mkdir(exist_ok=True,parents=True)
    checkpoint=Path("runs/trazo/step_000800.pt")
    state=torch.load(checkpoint,weights_only=True,map_location="cpu")
    weights=Path("pesos/trazo.pth")
    torch.save({k:state[k] for k in ("architecture","generator","step","data")},weights)
    net=load_generator(weights,"cuda")
    x,y,rows=images("val")
    groups={}
    for start in range(0,len(x),16):
        real=x[start:start+16].cuda().float()/127.5-1
        tags=y[start:start+16].cuda()
        source=degrade(real,torch.Generator(device="cuda").manual_seed(50000+start))
        output=net.painter(source,tags,hard=True)[0]
        target=(real.reshape(-1,2,3,96,96).amin(2)<.94)
        def iou(data):
            mask=data.reshape_as(real).reshape(-1,2,3,96,96).amin(2)<.94
            return ((mask&target).sum((-1,-2))/(mask|target).sum((-1,-2)).clamp_min(1)).mean(1)
        stats=torch.stack([iou(source),iou(output)],1).cpu().numpy()
        for row,pair in zip(rows[start:start+16],stats):
            for typ in {row["type_1"],row["type_2"]}-{""}:
                groups.setdefault(typ,{}).setdefault(row["species_id"],[]).append(pair)
    with (out/"validacion.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow(["tipo","especies_val","IoU_boceto","IoU_trazo"])
        for typ in TYPES:
            species=groups.get(typ,{})
            means=np.mean([np.mean(values,axis=0) for values in species.values()],axis=0) if species else [None,None]
            writer.writerow([typ,len(species),*means])
    protocol=json.loads(Path("estructura/ficha/protocolo.json").read_text(encoding="utf8"))
    seeds=protocol["semillas"]
    z=torch.stack([torch.randn(LATENT,generator=torch.Generator().manual_seed(s)) for s in seeds]).cuda()
    old=load_generator(Path("pesos/ficha.pth"),"cuda")
    records={}
    names=dict(zip(TYPES,SPANISH))
    real={r["pair_id"]:r for r in read_rows()}
    heads=["Real · mismo tipo","Ficha · 1.200","Cuerpos · 2.000","Trazo · +800"]
    for case,refs in protocol["referencias"].items():
        NAMES[case]=" / ".join(names[t] for t in case.split("+"))
        tags=z.new_tensor(labels(case)).expand(len(z),-1)
        base=net.base(z,tags)
        outputs=net.refine(base,tags)
        previous=rgb_pairs(old(z,tags))
        records[case]=[(case,s,[reference(real[refs[i%len(refs)][0]]),previous[i],a,b])
                       for i,(s,a,b) in enumerate(zip(seeds,rgb_pairs(base),rgb_pairs(outputs)))]
        draw_board(out/f"{case}.png",heads,records[case],NAMES[case],scale=1)
    draw_comparison(heads,[records[c][3] for c in CASES],out/"comparacion.png",
                    "Generación libre · Misma semilla y dos vistas de 96 px")
    for page in range(3):
        draw_board(out/f"tipos_{page+1}.png",heads,[records[t][3] for t in TYPES[page*6:page*6+6]],
                   "Comparación por tipo · Semilla 318071",scale=1)
    with Image.open("estructura/referencias_visuales/bicho_veneno_imagegen.png") as ref:
        w,h=ref.size
        leaf=pixel_reference([ref.crop((0,0,w//2,h)),ref.crop((w//2,0,w,h))])
    draw_board(out/"insecto_hoja.png",[*heads,"GPT · 96 px"],[(c,s,[*imgs,leaf]) for c,s,imgs in records["bug+poison"][:3]],
               "Insecto hoja · Tres semillas y una referencia",scale=2)
    conditions=TYPES+["+".join(p) for p in itertools.combinations(TYPES,2)]
    for i in range(0,len(conditions),6):
        tags=z.new_tensor([labels(c) for c in conditions[i:i+6]])
        pair=net(z[:1].expand(len(tags),-1),tags).reshape(-1,2,3,96,96)
        assert torch.isfinite(pair).all() and ((pair.amin(2)<.94).sum((-1,-2))>50).all()
    print("18 tipos y 153 combinaciones verificados.",flush=True)

if __name__=="__main__":
    main()
