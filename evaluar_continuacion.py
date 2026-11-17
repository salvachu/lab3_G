"""Compare the two continued models on fixed, paired, single-image CPU samples."""
import argparse
import itertools
import json
from pathlib import Path
from statistics import mean

import torch

from diagnose_pairs import rgb_pairs
from evaluar_diversa import CASES
from evaluar_empalme import nearest_training_pairs
from evaluar_perfiles import SPANISH, draw_comparison, labels
from evaluar_tipos_duales import reference, silhouette_distance
from gallery_cycles import NAMES, draw_board
from model import LATENT
from prepare import TYPES, read_rows
from sample_cycles import load_generator
from train import train_rows
from train_estructura import dataset


def union_errors(model, z, tags):
    ids, band, _ = model.draft(z, tags)
    ink = model.painter(ids, band, tags).argmax(2)
    seam = band.argmax(2, keepdim=True)
    above = ink.gather(2, seam)
    below = ink.gather(2, seam+1)
    occupied = (above>0) | (below>0)
    both = (above>0) & (below>0)
    edge = ((above>0) ^ (below>0)).sum() / occupied.sum().clamp_min(1)
    pigment = ((above-below).abs()*both).sum() / (8*both.sum().clamp_min(1))
    return edge.item(), pigment.item()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gan",type=Path,required=True)
    parser.add_argument("--empalme",type=Path,required=True)
    parser.add_argument("--progresion",type=Path,nargs="*",default=[])
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path("estructura/continuacion")
    out.mkdir(parents=True,exist_ok=True)
    files = [Path("pesos/jerarquica.pth"),args.gan,Path("pesos/empalme.pth"),args.empalme]
    models = [load_generator(p,"cpu") for p in files]
    steps = [torch.load(p,map_location="cpu",weights_only=True)["step"] for p in files]
    headings = [f"{name} · {steps[i]:,}".replace(",",".") for i,name in enumerate(["GAN","GAN","Empalme","Empalme"])]
    names = dict(zip(TYPES,SPANISH))
    history = json.loads(Path("estructura/bench_general/protocolo.json").read_text(encoding="utf8"))
    rows = {r["pair_id"]:r for r in read_rows()}
    seeds = [42003,43001,43002,43003,43004]
    report = {"semillas":seeds,"pasos":steps,"uniones":{},"distancia_siluetas":{}}
    errors = [[],[]]
    seam_cases=[]
    galleries = [[] for _ in models]
    distances = [[] for _ in models]
    comparison = []
    queries = []
    for case in CASES:
        NAMES[case] = " / ".join(names[t] for t in case.split("+"))
        tags = torch.tensor([labels(case)])
        images = [[] for _ in models]
        for seed in seeds:
            z = torch.randn(1,LATENT,generator=torch.Generator().manual_seed(seed))
            for i,model in enumerate(models):
                images[i].append(rgb_pairs(model(z,tags))[0])
            if seed!=42003:
                for i in (0,1):
                    queries.append((headings[i],case,seed,images[i][-1]))
                previous=models[2].select(z,tags)
                current=models[3].select(z,tags)
                assert all(torch.equal(a,b) for a,b in zip(previous,current))
                for i,model in enumerate(models[2:]):
                    errors[i].append(union_errors(model,z,tags))
                seam_cases.append((case,seed,[images[2][-1],images[3][-1]]))
        real = reference(rows[history["referencias"][case][0][0]])
        comparison.append((case,42003,[real]+[group[0] for group in images]))
        for i,group in enumerate(images):
            independent=group[1:]
            galleries[i].append((case,None,independent))
            distances[i].extend(silhouette_distance(a,b) for a,b in itertools.combinations(independent,2))
    draw_comparison(["Real · mismo tipo",*headings],comparison,out/"comparacion.png","Misma semilla 42003 · generación libre")
    for i,name in enumerate(["gan_antes","gan_despues","empalme_antes","empalme_despues"]):
        draw_board(out/(name+".png"),[f"Semilla {s}" for s in seeds[1:]],galleries[i],headings[i]+" · semillas de reserva",scale=1)
        report["distancia_siluetas"][headings[i]]=mean(distances[i])
    for name, measurements in zip(headings[2:],errors):
        report["uniones"][name]={"pares":len(measurements),"salto_contorno":mean(a for a,b in measurements),
                                 "salto_tono":mean(b for a,b in measurements)}
    changes=[before[0]-after[0] for before,after in zip(*errors)]
    order=sorted(range(len(changes)),key=changes.__getitem__)
    review=[seam_cases[i] for i in [*reversed(order[-3:]),*order[:3]]]
    draw_board(out/"union.png",headings[2:],review,
               "Borde de la unión · mejores y peores cambios",scale=2)
    report["uniones"]["pares_con_menor_salto_de_contorno"]=sum(change>0 for change in changes)
    report["uniones"]["pares_con_mayor_salto_de_contorno"]=sum(change<0 for change in changes)
    if args.progresion:
        interim=[load_generator(path,"cpu") for path in args.progresion]
        checkpoints=[torch.load(path,map_location="cpu",weights_only=True)["step"] for path in args.progresion]
        records=[]
        for case,seed,cells in comparison:
            tags=torch.tensor([labels(case)])
            z=torch.randn(1,LATENT,generator=torch.Generator().manual_seed(seed))
            records.append((case,seed,[cells[0],cells[1],*[rgb_pairs(m(z,tags))[0] for m in interim],cells[2]]))
        cols=["Real · mismo tipo",headings[0],*[f"GAN · {step:,}".replace(",",".") for step in checkpoints],headings[1]]
        draw_board(out/"progresion.png",cols,records,"GAN jerárquica · evolución con la misma semilla",scale=1)
    report["similitud_train"]=nearest_training_pairs(queries,rows,train_rows(),dataset()["images"],out)
    cover=[]
    for kind in TYPES:
        NAMES[kind]=names[kind]
        tags=torch.tensor([labels(kind)])
        z=torch.randn(1,LATENT,generator=torch.Generator().manual_seed(43002))
        source=reference(rows[history["referencias"][kind][0][0]])
        cover.append((kind,43002,[source]+[rgb_pairs(m(z,tags))[0] for m in models]))
    for start in (0,9):
        draw_board(out/f"tipos_{1+start//9}.png",["Real · mismo tipo",*headings],cover[start:start+9],
                   "Los 18 tipos · misma semilla",scale=1)
    (out/"resultados.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    print(report,flush=True)


if __name__=="__main__":
    main()
