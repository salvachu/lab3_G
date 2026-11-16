"""Expande la lectura visual por código anónimo a un CSV verificable por celda."""
import argparse
import csv
import json
from statistics import mean

from gallery_cycles import OUT, PROTOCOL


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score(phase):
    folder = OUT/phase
    key, reading = (load_json(folder/name) for name in ("orden_revision.json", "lectura.json"))
    seeds = PROTOCOL["reserva" if phase=="final" else "desarrollo"]
    rows = []
    for case, names in key.items():
        for idx, name in enumerate(names):
            code = chr(65+idx)
            evaluation = reading[case][code]
            assert set(evaluation["cuerpos"]) <= set(seeds)
            assert set(map(int,evaluation["rasgos"])) <= set(seeds)
            for seed in seeds:
                feature = evaluation["rasgos"].get(str(seed),"")
                rows.append(dict(modelo=name,tipo=case,semilla=seed,columna_oculta=code,
                                 cuerpo_en_ambas=int(seed in evaluation["cuerpos"]),
                                 rasgo_compartido=int(bool(feature)),observacion=feature))
    def export(filename, records):
        with (folder/filename).open("w",newline="",encoding="utf-8-sig") as stream:
            w=csv.DictWriter(stream,fieldnames=records[0])
            w.writeheader()
            w.writerows(records)
    export("valoracion_por_pareja.csv", rows)
    with (folder/"siluetas.csv").open() as stream:
        ious=list(csv.DictReader(stream))
    summary=[]
    registry=folder/"checkpoints.json"
    order=load_json(registry) if registry.exists() else dict.fromkeys(r["modelo"] for r in rows)
    for name in order:
        group=[r for r in rows if r["modelo"]==name]
        def avg_iou(view):
            return round(mean(float(r["mean_iou"]) for r in ious if r["stage"]==name and r["view"]==view),3)
        summary.append(dict(modelo=name,parejas=len(group),cuerpos=sum(r["cuerpo_en_ambas"] for r in group),
                            rasgos=sum(r["rasgo_compartido"] for r in group),IoU_frente=avg_iou("frente"),
                            IoU_espalda=avg_iou("espalda")))
    export("resultados.csv",summary)
    print(summary)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase")
    score(p.parse_args().phase)
