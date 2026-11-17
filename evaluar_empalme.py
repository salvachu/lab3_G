"""Free paired generation on fixed seeds, component provenance, and overlap checks."""
import hashlib
import itertools
import json
import csv
from pathlib import Path

import numpy as np
import torch

from diagnose_pairs import rgb_pairs
from estructura_vecinos import features, normalized, pair_image
from evaluar_diversa import CASES
from evaluar_perfiles import SPANISH, draw_comparison, labels
from evaluar_tipos_duales import reference
from gallery_cycles import NAMES, draw_board
from model import LATENT
from prepare import PREPARED, TYPES, read_rows
from sample_cycles import load_generator
from train import train_rows
from train_empalme import representatives
from train_estructura import dataset


def nearest_training_pairs(queries, rows, train, images, out):
    """Compare whole sprite pairs with TRAIN in both orientations at the same scale."""
    with (PREPARED/"splits.csv").open(encoding="utf8") as file:
        held_out={r["pair_id"] for r in csv.DictReader(file) if r["split"]=="val"}
    validation={}
    for key in sorted(held_out):
        row=rows[key]
        validation.setdefault(row["species_id"],row)
    controls=[pair_image(r) for r in validation.values()]
    pictures=[pair for _,_,_,pair in queries]+controls
    query=features(torch.from_numpy(np.stack([normalized(im) for im in pictures])))
    best=query.new_full((len(pictures),),float("inf"))
    hit=torch.zeros_like(best,dtype=torch.long)
    for start in range(0,len(train),64):
        chunk=images[start:start+64]
        for batch in (chunk,chunk.flip(-1)):
            distance,index=torch.cdist(query,features(batch)).min(1)
            hit=torch.where(distance<best,index+start,hit)
            best=torch.minimum(best,distance)
    distance=(best/query.shape[1]**.5).tolist()
    hits=hit.tolist()
    threshold=float(np.percentile(distance[len(queries):],5))
    result={"especies_validacion":len(controls),"p05_validacion":threshold,"modelos":{}}
    records=[]
    for model in dict.fromkeys(record[0] for record in queries):
        indices=sorted((i for i,rec in enumerate(queries) if rec[0]==model),key=distance.__getitem__)
        values=[distance[i] for i in indices]
        result["modelos"][model]={"pares":len(values),"por_debajo_p05_validacion":sum(v<threshold for v in values),
                                  "distancia_mediana":float(np.median(values))}
        for i in indices[:3]:
            _,case,seed,im=queries[i]
            nearest=train[hits[i]]
            records.append((case,f"{model} · {seed}",[im,pair_image(nearest)]))
    draw_board(out/"parecidos_train.png",["Generada","Real más cercano"],records,
               "Tres pares más cercanos por modelo",scale=1)
    return result


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    out=Path("estructura/empalme")
    out.mkdir(exist_ok=True,parents=True)
    # CPU single-image sampling matches the normal interface on Linux and Windows.
    models={name:load_generator(Path("pesos")/(name+".pth"),"cpu") for name in
            ("aegan_6000","mosaico","empalme","jerarquica","ficha","perfiles","ficha_todos")}
    chosen={"water":"ficha","dragon":"perfiles","fighting":"perfiles","fire":"ficha",
            "electric":"ficha","ghost":"ficha","bug+poison":"ficha_todos","water+dragon":"ficha_todos"}
    history=json.loads(Path("estructura/bench_general/protocolo.json").read_text(encoding="utf8"))
    rows={r["pair_id"]:r for r in read_rows()}
    train=train_rows()
    train_images=dataset()["images"]
    original_hashes={Path(r[k]).stem for r in train for k in ("front_path","back_path")}
    centred_hashes={hashlib.sha256(view.permute(1,2,0).contiguous().numpy().tobytes()).hexdigest()
                    for pair in train_images for view in pair}
    heads,bodies=representatives(train)
    assert {train[i]["species_id"] for i in heads}.isdisjoint({train[i]["species_id"] for i in bodies})
    names=dict(zip(TYPES,SPANISH))
    seeds=[42001,42002,42003,42004,318071]
    comparison=[]; progression=[]; provenance=[]; sources=[]; grid=[]; mosaic_grid=[]; neural_grid=[]
    counts={k:{"raw_exact_views":0,"canonical_exact_views":0,"checks":0} for k in ("empalme","jerarquica")}
    queries=[]
    neural_step=torch.load("pesos/jerarquica.pth",weights_only=True,map_location="cpu")["step"]
    neural_label=f"GAN · {neural_step:,}".replace(",",".")
    changed=0
    for case in CASES:
        NAMES[case]=" / ".join(names[t] for t in case.split("+"))
        tags=torch.tensor([labels(case)])
        variants=[]; mosaics=[]; neural=[]
        for seed in seeds:
            z=torch.randn(1,LATENT,generator=torch.Generator().manual_seed(seed))
            net=models["empalme"]
            pair=net(z,tags)
            sprite=rgb_pairs(pair)[0]
            learned=models["jerarquica"](z,tags)
            neural_sprite=rgb_pairs(learned)[0]
            queries.extend([("jerarquica",case,seed,neural_sprite),("empalme",case,seed,sprite)])
            mosaic=rgb_pairs(models["mosaico"](z,tags))[0]
            ids,band,_=net.draft(z,tags)
            edited=net.painter(ids,band,tags).argmax(2)
            changed+=int((ids!=edited).sum())
            for name,output in (("empalme",pair),("jerarquica",learned)):
                for view in ((output[0].reshape(2,3,96,96)+1)*127.5).round().byte():
                    pixels=view.permute(1,2,0).contiguous().numpy()
                    counts[name]["checks"]+=1
                    counts[name]["raw_exact_views"]+=hashlib.sha256(pixels.tobytes()).hexdigest() in original_hashes
                    counts[name]["canonical_exact_views"]+=hashlib.sha256(pixels[16:80,16:80].tobytes()).hexdigest() in centred_hashes
            h,b=net.select(z,tags)
            head=train[heads[int(h)]]
            body=train[bodies[int(b)]]
            assert head["species_id"]!=body["species_id"]
            sources.append(dict(case=case,seed=seed,head=head["pair_id"],body=body["pair_id"]))
            if seed==318071:
                rid=history["referencias"][case][0][0]
                previous=rgb_pairs(models[chosen[case]](z,tags))[0]
                aegan=rgb_pairs(models["aegan_6000"](z,tags))[0]
                prev_name={"ficha":"Ficha","ficha_todos":"Cuerpos","perfiles":"Perfiles"}[chosen[case]]
                comparison.append((case,prev_name,[reference(rows[rid]),previous,neural_sprite,sprite]))
                progression.append((case,seed,[reference(rows[rid]),aegan,mosaic,neural_sprite,sprite]))
                provenance.append((case,seed,[reference(head),reference(body),sprite]))
            else:
                variants.append(sprite)
                mosaics.append(mosaic)
                neural.append(neural_sprite)
        grid.append((case,None,variants))
        mosaic_grid.append((case,None,mosaics))
        neural_grid.append((case,None,neural))
    draw_comparison(["Real · mismo tipo","Anterior de la fila",neural_label,"Empalme · 1.500"],
                    comparison,out/"comparacion.png","Misma semilla 318071 · Ocho tipos y dos vistas")
    draw_board(out/"modelos.png",["Real · mismo tipo","AEGAN · 6.000","Mosaico · +1.000",neural_label,"Empalme · 1.500"],
               progression,"Generación desde tipos y semilla",scale=1)
    headings=[f"Semilla {n}" for n in seeds[:-1]]
    draw_board(out/"semillas.png",headings,grid,"Empalme · Cuatro semillas consecutivas",scale=1)
    draw_board(out/"mosaico.png",headings,mosaic_grid,"Mosaico · Cuatro semillas consecutivas",scale=1)
    draw_board(out/"jerarquica.png",headings,neural_grid,"GAN jerárquica · Cuatro semillas consecutivas",scale=1)
    draw_board(out/"origen.png",["Origen superior","Origen inferior","Empalme · 1.500"],provenance,
               "Empalme · Especies de entrenamiento que aportan cada región",scale=1)
    with (PREPARED/"splits.csv").open(encoding="utf8") as file:
        validation=[rows[row["pair_id"]] for row in csv.DictReader(file) if row["split"]=="val"]
    coverage=[]
    for kind in TYPES:
        NAMES[kind]=names[kind]
        candidates=[r for r in validation if r["type_"+kind]=="1"]
        target=min(candidates,key=lambda r:(bool(r["type_2"]),r["pair_id"]))
        sprites=[reference(target)]
        for model in ("jerarquica","empalme"):
            for seed in seeds[:2]:
                noise=torch.randn(1,LATENT,generator=torch.Generator().manual_seed(seed))
                sprites.append(rgb_pairs(models[model](noise,torch.tensor([labels(kind)])))[0])
        coverage.append((kind,None,sprites))
    for start in (0,9):
        draw_board(out/f"tipos_{start//9+1}.png",
                   ["Real · validación","GAN · 42001","GAN · 42002","Empalme · 42001","Empalme · 42002"],
                   coverage[start:start+9],"Los 18 tipos · dos semillas por modelo",scale=1)
    conditions=[[t] for t in TYPES]+list(itertools.combinations(TYPES,2))
    for i in range(0,len(conditions),4):
        tags=torch.tensor([labels("+".join(c)) for c in conditions[i:i+4]])
        for name in ("empalme","mosaico","jerarquica"):
            rgb=models[name](z.expand(len(tags),-1),tags).reshape(-1,2,3,96,96)
            assert torch.isfinite(rgb).all() and (rgb.amin(2)<.9).flatten(2).any(2).all()
    metrics=dict(modelos_anteriores=chosen,semillas=seeds,condiciones_verificadas=len(conditions),
                 controles=counts,pixeles_modificados_en_union=changed,origenes=sources,paso_gan=neural_step,
                 vecinos=nearest_training_pairs(queries,rows,train,train_images,out))
    (out/"evaluacion.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf8")
    print(counts,flush=True)


if __name__=="__main__":
    main()
