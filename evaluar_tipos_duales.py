"""Benchmark reproducible de forma, condición y correspondencia para los tipos solicitados."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from compare_checkpoints import NAMES, font
from diagnose_pairs import rgb_pairs
from gallery_cycles import draw_board
from model import LATENT
from prepare import DATA, PREPARED, TYPES, condition, read_rows
from sample_cycles import load_generator


CASES = {"water+dragon":"Agua / Dragón", "ghost":"Fantasma", "dragon+ghost":"Dragón / Fantasma",
         "bug+poison":"Bicho / Veneno", "water+ghost":"Agua / Fantasma"}
SEEDS = {"desarrollo":[137, 211, 613, 997],
         "reserva":[83009, 83417, 83663, 84089, 84377, 84811, 85369, 85909],
         "confirmacion":[96137, 96553, 97001, 97367, 97931, 98317, 98809, 99277]}


def reference(row):
    result = Image.new("RGB", (192, 96), "white")
    for i, key in enumerate(("front_path", "back_path")):
        with Image.open(DATA / row[key]) as im:
            result.paste(im.convert("RGB"), (96 * i, 0))
    return result


def silhouette_distance(left, right):
    a = np.asarray(left).min(2) < 238
    b = np.asarray(right).min(2) < 238
    return 1 - np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--modelo", nargs=2, action="append", required=True)
    p.add_argument("--fase", choices=SEEDS, required=True)
    p.add_argument("--control",choices=("acuatico","veneno"))
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    cases=CASES
    if args.control=="acuatico":
        cases={"water":"Agua","dragon":"Dragón","water+dragon":"Agua / Dragón"}
    if args.control=="veneno":
        cases={"bug":"Bicho","poison":"Veneno","bug+poison":"Bicho / Veneno"}
    NAMES.update(cases)
    seeds = SEEDS[args.fase]
    if args.control:
        seeds=[91577,91813,92219,92863] if args.fase!="confirmacion" else [102113,102811,103333,103841]
    device = "cpu"
    torch.set_num_threads(3)
    models = dict(args.modelo)
    z = torch.stack([torch.randn(LATENT, generator=torch.Generator().manual_seed(s)) for s in seeds])
    with (PREPARED / "splits.csv").open() as stream:
        splits = {r["pair_id"]:r["split"] for r in csv.DictReader(stream)}
    rows = read_rows()
    config = {"semillas": seeds, "modelos": models, "referencias": {}, "soporte": {}}
    grids = {}
    refs = {}
    for case in cases:
        available = [r for r in rows if condition(r) == set(case.split("+"))]
        available.sort(key=lambda r: (splits[r["pair_id"]] != "val", r["pair_id"]))
        unique = {}
        for row in available:
            unique.setdefault(row["species_id"],row)
        by_species = list(unique.values())
        refs[case] = [reference(by_species[i % len(by_species)]) for i in range(len(seeds))]
        config["referencias"][case] = [(r["pair_id"],splits[r["pair_id"]]) for r in by_species]
        config["soporte"][case] = {}
        for split in ("train","val"):
            group = [r for r in available if splits[r["pair_id"]] == split]
            config["soporte"][case][split] = [len(group),len({r["species_id"] for r in group})]
    metrics = {}
    for name, ck in models.items():
        net = load_generator(Path(ck),device)
        grid = {}
        for case in cases:
            y = torch.tensor([[float(t in case.split("+")) for t in TYPES]]).expand(len(seeds),-1)
            grid[case] = rgb_pairs(net(z,y))
        grids[name] = grid
        def distance(case1,i,case2,j):
            return silhouette_distance(grid[case1][i],grid[case2][j])
        metrics[name] = {
            "diferencia_forma_cambio_z": np.mean([distance(c,i,c,j) for c in cases for i in range(len(seeds)) for j in range(i)]),
            "diferencia_forma_cambio_tipo": np.mean([distance(c,i,d,i) for k,c in enumerate(cases) for d in list(cases)[:k] for i in range(len(seeds))]),
        }
    def records(cases, indices):
        return [(c,seeds[i],[refs[c][i]] + [g[c][i] for g in grids.values()]) for c in cases for i in indices]
    for case in cases:
        draw_board(args.out / (case+".png"), ["Real · mismo tipo", *models], records([case],range(len(seeds))), cases[case])
    draw_board(args.out/"comparacion.png", ["Real · mismo tipo", *models], records(cases,[0,1] if args.control else [0]), "Comparación con los tipos individuales" if args.control else "Una misma semilla en cinco condiciones")
    if args.fase in ("reserva","confirmacion"):
        # One anonymous sheet per candidate. An independent reader can label it before opening its key.
        key = {}
        for n, (name,grid) in enumerate(grids.items()):
            shuffled = [(c,i) for c in cases for i in range(2)]
            salt=113 if args.control=="acuatico" else 331 if args.control=="veneno" else 0
            shuffle_seed=8417+n if args.fase=="reserva" else 6511+n+salt
            random.Random(shuffle_seed).shuffle(shuffled)
            board = Image.new("RGB",(820,210*len(cases)),"white")
            pen = ImageDraw.Draw(board)
            for k, (c,i) in enumerate(shuffled):
                x, y = 410*(k%2),210*(k//2)
                pen.text((x+8,y),str(k+1),fill="black",font=font(17,True))
                board.paste(grid[c][i].resize((384,192),Image.Resampling.NEAREST),(x+15,y+17))
            board.save(args.out/f"ciego_{n+1}.png")
            key[str(n+1)]={"modelo":name,"orden":[(c,seeds[i]) for c,i in shuffled]}
        (args.out/"clave.json").write_text(json.dumps(key,indent=2),encoding="utf8")
    (args.out/"protocolo.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf8")
    (args.out/"diversidad.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf8")


if __name__ == "__main__":
    main()
