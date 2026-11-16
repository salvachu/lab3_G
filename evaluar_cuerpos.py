"""Export the selected weights and compare all 18 types and their 153 pairs."""
import itertools
from pathlib import Path
import json

import torch
from diagnose_pairs import rgb_pairs
from evaluar_perfiles import NAMES, SPANISH, draw_comparison, labels, pixel_reference
from PIL import Image
from evaluar_tipos_duales import reference
from gallery_cycles import draw_board
from model import LATENT
from prepare import TYPES, read_rows
from sample_cycles import load_generator
from train_sprite_latent import CASES


@torch.inference_mode()
def main():
    out=Path("estructura/cuerpos")
    out.mkdir(exist_ok=True,parents=True)
    checkpoint=Path("runs/ficha_todos/step_002000.pt")
    state=torch.load(checkpoint,weights_only=True,map_location="cpu")
    weights=Path("pesos/ficha_todos.pth")
    weights.parent.mkdir(exist_ok=True)
    keys=("architecture","generator","data","step","campos","pares","especies","detailed","all_bodies")
    torch.save({k:state[k] for k in keys},weights)
    before=torch.load("runs/ficha_descriptiva/step_001200.pt",weights_only=True,map_location="cpu")
    torch.save({k:before[k] for k in keys if k in before},weights.parent/"ficha.pth")
    device="cuda" if torch.cuda.is_available() else "cpu"
    net=load_generator(weights,device)
    previous=load_generator(Path("runs/ficha_descriptiva/step_001200.pt"),device)
    protocol=json.loads(Path("estructura/ficha/protocolo.json").read_text(encoding="utf8"))
    seeds=protocol["semillas"]
    z=torch.stack([torch.randn(LATENT,generator=torch.Generator().manual_seed(s)) for s in seeds]).to(device)
    names=dict(zip(TYPES,SPANISH))
    real={r["pair_id"]:r for r in read_rows()}
    grids={}
    headings=["Real · mismo tipo","Ficha · 1.200","Cuerpos · 2.000"]
    for case,refs in protocol["referencias"].items():
        NAMES[case]=" / ".join(names[t] for t in case.split("+"))
        tags=z.new_tensor(labels(case)).expand(len(z),-1)
        outputs=net(z,tags)
        assert torch.isfinite(outputs).all()
        sprites=rgb_pairs(outputs)
        assert len({sprite.tobytes() for sprite in sprites})==len(seeds)
        grids[case]=[(case,seed,[reference(real[refs[j%len(refs)][0]]),old,new])
                     for j,(seed,old,new) in enumerate(zip(seeds,rgb_pairs(previous(z,tags)),sprites))]
    draw_comparison(headings,[grids[c][3] for c in CASES],out/"comparacion.png","Nuevos cuerpos · Ocho tipos con la misma semilla")
    with Image.open("estructura/referencias_visuales/bicho_veneno_imagegen.png") as ref:
        half=ref.width//2
        pair=pixel_reference([ref.crop((0,0,half,ref.height)),ref.crop((half,0,ref.width,ref.height))])
    leaf=[(c,s,[*imgs,pair]) for c,s,imgs in grids["bug+poison"][:3]]
    draw_board(out/"insecto_hoja.png",[*headings,"GPT · 96 px"],leaf,"Bicho / Veneno · Referencia de insecto hoja",scale=2)
    for part in range(3):
        draw_board(out/f"tipos_{part+1}.png",headings,[grids[t][3] for t in TYPES[part*6:part*6+6]],
                   "Cuerpos por tipo · Semilla 318071",scale=1)
    seen=TYPES+["+".join(pair) for pair in itertools.combinations(TYPES,2)]
    for n in range(0,len(seen),6):
        y=z.new_tensor([labels(case) for case in seen[n:n+6]])
        pairs=net(z[:1].expand(len(y),-1),y).reshape(-1,2,3,96,96)
        assert torch.isfinite(pairs).all()
        assert ((pairs.amin(2)<.9).sum((-1,-2))>50).all()
        assert (pairs[:,0]!=pairs[:,1]).flatten(1).any(1).all()
    (out/"protocolo.json").write_text(json.dumps({"checkpoint":str(checkpoint),"semillas":seeds,
        "tipos_simples":len(TYPES),"combinaciones_verificadas":len(seen),"condiciones_con_seis_semillas":len(grids)},
        ensure_ascii=False,indent=2),encoding="utf8")

if __name__=="__main__":
    main()
