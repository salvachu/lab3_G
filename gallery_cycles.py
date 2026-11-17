"""Muestreo fijo, galerías por tipo y láminas A/B para lectura visual."""
import argparse
import csv
import itertools
import json
import random
from statistics import mean
from pathlib import Path

from PIL import Image, ImageDraw

from compare_checkpoints import NAMES, font, pairs, shape_mask, silhouette_stats
from generate import render
from sample_cycles import load_generator
from prepare import ROOT, PREPARED, SEEDS

OUT = ROOT / "benchmark_ciclos"
PROTOCOL = json.loads((OUT / "protocolo.json").read_text(encoding="utf-8"))
CASES = PROTOCOL["tipos"]
BASES = {
    "Original · 2.500": "runs/baseline/step_02500.pt",
    "Original · 5.000": "runs/baseline_5000/step_05000.pt",
    "Siluetas · 5.000": "runs/siluetas_5000/step_05000.pt",
}


def draw_board(path, headings, records, title, scale=2):
    margin, col, row = 135, 200, 104
    image = Image.new("RGB", ((margin+col*len(headings))*scale, (76+row*len(records))*scale), "white")
    pen = ImageDraw.Draw(image)
    def label(x, y, txt, size, bold=False, color="#283e4d"):
        pen.text((x*scale, y*scale), txt, font=font(size*scale, bold), fill=color)
    label(8, 4, title, 19, True)
    for j, name in enumerate(headings):
        x = margin+col*j
        label(x+4, 33, name, 14, True)
        label(x+4, 54, "Frente", 11)
        label(x+100, 54, "Espalda", 11)
    for i, (case, seed, sprites) in enumerate(records):
        y = 74+row*i
        label(8, y+29, NAMES[case], 13, True)
        if seed is not None:
            label(8, y+48, seed if isinstance(seed,str) else f"z = {seed}", 12)
        for j, sprite in enumerate(sprites):
            image.paste(sprite.resize((192*scale, 96*scale), Image.Resampling.NEAREST), ((margin+col*j)*scale, y*scale))
        pen.line((6*scale,(y+99)*scale,image.width-6*scale,(y+99)*scale),fill="#dce4e7",width=scale)
    image.save(path)


def build(phase, models, cover=None):
    seeds = PROTOCOL["reserva" if phase=="final" else "desarrollo"]
    folder = OUT / phase
    folder.mkdir(exist_ok=True)
    (folder/"checkpoints.json").write_text(json.dumps(models,ensure_ascii=False,indent=2),encoding="utf-8")
    grids = {}
    for name, path in models.items():
        image_path = folder / (Path(path).parent.name + "_"+Path(path).stem+".png")
        net = load_generator(ROOT/path,"cuda")
        render(net, CASES, seeds, image_path)
        grids[name] = pairs(image_path, CASES, seeds)
    silhouette_stats(grids, CASES, folder / "siluetas.csv")
    with (folder/"misma_semilla_entre_tipos.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow(["modelo","semilla","vista","IoU_media_entre_tipos","comparaciones_validas"])
        for name,grid in grids.items():
            for j,seed in enumerate(seeds):
                for view,offset in (("frente",0),("espalda",96)):
                    masks=[shape_mask(group[j].crop((offset,0,offset+96,96))) for group in grid]
                    overlaps=[(a&b).sum()/(a|b).sum() for a,b in itertools.combinations(masks,2) if a.any() and b.any()]
                    writer.writerow([name,seed,view,mean(overlaps) if overlaps else None,len(overlaps)])
    key = {}
    rng = random.Random(930+sum(map(ord,phase)))
    for i, case in enumerate(CASES):
        names = list(models)
        def records(order):
            return [(case, seed, [grids[n][i][j] for n in order]) for j, seed in enumerate(seeds)]
        draw_board(folder / f"{case}.png", names, records(names), NAMES[case])
        rng.shuffle(names)
        key[case] = names
        draw_board(folder / f"revision_{case}.png", [chr(65+j) for j in range(len(names))],
                   records(names), NAMES[case], scale=1)
    (folder/"orden_revision.json").write_text(json.dumps(key, ensure_ascii=False, indent=2),encoding="utf-8")
    if cover:
        real = pairs(PREPARED/"real_val_roundtrip.png", CASES, SEEDS)
        records = [(case, seed, [real[i][k]]+[grids[n][i][seeds.index(seed)] for n in cover])
                   for i,case in enumerate(CASES) for k,seed in enumerate(PROTOCOL["portada"])]
        draw_board(OUT/"comparacion_visual.png", ["Real · mismo tipo"]+cover, records, "Evolución de los sprites")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase")
    p.add_argument("--registry",type=Path)
    args=p.parse_args()
    if args.registry:
        cfg=json.loads(args.registry.read_text(encoding="utf-8"))
        build(args.phase,cfg["models"],cfg.get("cover"))
    else:
        build(args.phase,dict(list(BASES.items())[1:]))
