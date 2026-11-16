"""Compare six unfiltered seeds and export measured TRAIN cards for inspection."""
import colorsys
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diagnose_pairs import rgb_pairs
from estructura_vecinos import check
from evaluar_perfiles import MODELS, NAMES, OUTPUT, SPANISH, draw_comparison, labels
from evaluar_tipos_duales import reference
from gallery_cycles import draw_board
from model import LATENT
from prepare import TYPES, read_rows
from sample_cycles import load_generator
from sprite_ficha import FIELDS, measure
from train import train_rows
from train_estructura import dataset
from train_sprite_latent import CASES

OUT=Path("estructura/ficha")
CHECKPOINT=Path("runs/ficha_descriptiva/step_001200.pt")


def caption(values):
    ratio,lower,fill,r,g,b=values
    hue,sat,value=colorsys.rgb_to_hsv(r,g,b)
    hues=["rojo","naranja","amarillo","verde","verde","verde azulado","cian","azul","azul","violeta","magenta","rosa"]
    color=hues[round(12*hue)%12] if sat>.19 else ("gris oscuro" if value<.44 else "gris claro")
    shape="alta" if ratio<.85 else "ancha" if ratio>1.18 else "equilibrada"
    mass="abajo" if lower>.12 else "arriba" if lower<-.12 else "repartida"
    return f"Silueta {shape}, masa {mass}, ocupación {fill:.0%}, tono {color}."


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    net=load_generator(CHECKPOINT,"cuda")
    baseline=load_generator(Path(list(MODELS.values())[-1]),"cuda")
    data=dataset()
    cards=torch.cat([measure(a.cuda().float()/127.5-1).cpu() for a in data["images"].split(64)])
    with (OUT/"fichas_train.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["pair_id","tipo_1","tipo_2",*FIELDS,"descripcion"])
        for row,values in zip(train_rows(),cards.tolist(),strict=True):
            writer.writerow([row["pair_id"],row["type_1"],row["type_2"],*map(lambda v:round(v,4),values),caption(values)])
    old=json.loads((OUTPUT/"protocolo.json").read_text(encoding="utf8"))
    refs=old["referencias"]
    seeds=old["semillas"]
    z=torch.stack([torch.randn(LATENT,generator=torch.Generator().manual_seed(s)) for s in seeds]).cuda()
    real={r["pair_id"]:r for r in read_rows()}
    translations=dict(zip(TYPES,SPANISH))
    metrics={}
    rows_main=[]
    models={"Pigmentos · 24 paletas":list(MODELS.values())[-1],"Ficha · +1.200":str(CHECKPOINT)}
    with (OUT/"fichas_generadas.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["tipo","semilla",*FIELDS,"descripcion"])
        for case in refs:
            NAMES[case]=" / ".join(translations[t] for t in case.split("+"))
            y=z.new_tensor(labels(case)).expand(len(z),-1)
            traits=net.describe(z,y)
            previous=baseline(z,y)
            new=net(z,y)
            assert torch.isfinite(new).all()
            actual=measure(new)
            metrics[case]={"error_de_proporcion_previo":(measure(previous)[:,0]-traits[:,0]).abs().mean().item(),
                           "error_de_proporcion_ficha":(actual[:,0]-traits[:,0]).abs().mean().item()}
            samples=rgb_pairs(new)
            entries=[]
            old_samples=rgb_pairs(previous)
            for i,(seed,sprite,values) in enumerate(zip(seeds,samples,traits.tolist())):
                ref=reference(real[refs[case][i%len(refs[case])][0]])
                entries.append((case,seed,[ref,old_samples[i],sprite]))
                writer.writerow([case,seed,*[round(v,4) for v in values],caption(values)])
                if i==3 and case in CASES:
                    # Reuse the four rendered original columns for the exact historical comparison.
                    with Image.open(OUTPUT/f"{case}.png") as sheet:
                        top=2*(74+104*i)
                        prior=[sheet.crop((2*x,top,2*x+384,top+192)).resize((192,96),Image.Resampling.NEAREST)
                               for x in (135,335,535,735)]
                    rows_main.append((case,seed,[*prior,sprite]))
            draw_board(OUT/f"{case}.png",["Real · mismo tipo",*models],entries,NAMES[case])
            if case=="ghost":
                draw_comparison(["Real · mismo tipo",*models],entries[:3],OUT/"fantasma_con_gpt.png",
                                "Fantasma · Tres semillas frente a la misma referencia GPT")
    rows_main.sort(key=lambda row:CASES.index(row[0]))
    draw_comparison(["Real · mismo tipo",*MODELS,"Ficha · +1.200"],rows_main,OUT/"comparacion.png")
    conditions=TYPES+["+".join(pair) for pair in itertools.combinations(TYPES,2)]
    for start in range(0,len(conditions),4):
        labels_batch=z.new_tensor([labels(c) for c in conditions[start:start+4]])
        output=net(z[:1].expand(len(labels_batch),-1),labels_batch)
        assert torch.isfinite(output).all() and (output.reshape(-1,2,3,96,96).amin(2)<.94).flatten(2).any(2).all()
    # Intervention: keep type and seed fixed, vary only the specified proportions and pigment.
    y=z.new_tensor([labels("ghost")])
    controls=[{"ancho_sobre_alto":.62,"rojo":.40,"verde":.25,"azul":.75},
              {"ancho_sobre_alto":1.65,"rojo":.40,"verde":.25,"azul":.75},
              {"ancho_sobre_alto":1.65,"rojo":.83,"verde":.30,"azul":.15}]
    generated=[]
    widths=[]
    for custom in controls:
        net.overrides=custom
        sprite=net(z[3:4],y)
        generated+=rgb_pairs(sprite)
        widths.append(measure(sprite)[0,0].item())
    assert widths[1]>widths[0] and not np.array_equal(np.asarray(generated[1]),np.asarray(generated[2]))
    (OUT/"ficha_ejemplo.json").write_text(json.dumps(controls[0],indent=2),encoding="utf8")
    draw_board(OUT/"control.png",["Estrecha · violeta","Ancha · violeta","Ancha · rojiza"],
               [("ghost",seeds[3],generated)],"Una semilla, tres fichas")
    protocol=dict(modelos=models,semillas=seeds,referencias=refs,prueba_condiciones=len(conditions),
                  medidas=metrics,ancho_altura_control=widths)
    (OUT/"protocolo.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf8")
    check("ficha")


if __name__=="__main__":
    main()
