"""Verifica los sprites y genera una partición reproducible por especie."""
import csv
import hashlib
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
PREPARED = ROOT / "prepared"
TYPES = "normal fighting flying poison ground rock bug ghost steel fire water grass electric psychic ice dragon dark fairy".split()
CASES = ["water", "fire", "grass+poison", "normal+flying"]
SEEDS = [11, 29, 47, 83]


def condition(row):
    return {row[key] for key in ("type_1", "type_2") if row[key]}


def read_rows():
    with (DATA / "pairs.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def split_species(rows):
    # Agrupa también las especies que comparten alguna vista exacta.
    parent = {r["species_id"]: r["species_id"] for r in rows}

    def root(item):
        while item != parent[item]:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    owners = {}
    for row in rows:
        species = row["species_id"]
        for key in ("front_path", "back_path"):
            other = owners.setdefault(row[key], species)
            parent[root(other)] = root(species)
    groups = sorted({root(s) for s in parent}, key=int)
    random.Random(310).shuffle(groups)
    held_out = set(groups[:round(len(groups) * 0.1)])
    return {s: "val" if root(s) in held_out else "train" for s in parent}


def main():
    PREPARED.mkdir(exist_ok=True)
    rows = read_rows()
    ids, pairs, referenced = set(), set(), set()
    for r in rows:
        pair = (r["front_path"], r["back_path"])
        active = {t for t in TYPES if r["type_" + t] == "1"}
        bits = {r["type_" + t] for t in TYPES}
        if (r["pair_id"] in ids or pair in pairs or bits - {"0", "1"}
                or active != condition(r) or len(active) not in (1, 2)
                or len(active) != int(r["type_count"])):
            raise ValueError(f"Pareja o etiqueta inválida: {r['pair_id']}")
        ids.add(r["pair_id"])
        pairs.add(pair)
        referenced.update(pair)
    with zipfile.ZipFile(DATA / "images96.zip") as archive:
        files = {n for n in archive.namelist() if not n.endswith("/")}
        if referenced - files:
            raise ValueError("Hay vistas en el CSV ausentes del ZIP")
        for name in sorted(files):
            if not re.fullmatch(r"images96/[0-9a-f]{2}/[0-9a-f]{64}\.png", name):
                raise ValueError(name)
            path = DATA / name
            if not path.exists():
                archive.extract(name, DATA)
            with Image.open(path) as im:
                digest = hashlib.sha256(im.tobytes()).hexdigest()
                if im.size != (96, 96) or im.mode != "RGB" or digest != path.stem:
                    raise ValueError(f"Sprite inválido: {name}")
    partition = split_species(rows)
    for row in rows:
        row["split"] = partition[row["species_id"]]
    with (PREPARED / "splits.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "species_id", "split"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for part in ("train", "val"):
        subset = [r for r in rows if r["split"] == part]
        counts[part] = dict(pairs=len(subset), species=len({r["species_id"] for r in subset}),
                            conditions=Counter("+".join(sorted(condition(r))) for r in subset))
    summary = dict(pairs=len(rows), species=len(partition), png=len(files), referenced=len(referenced),
                   unused=sorted(files - referenced), type_counts=Counter(r["type_count"] for r in rows),
                   splits=counts, csv_sha256=hashlib.sha256((DATA / "pairs.csv").read_bytes()).hexdigest())
    (PREPARED / "audit.json").write_text(json.dumps(summary, indent=2))
    cases = {}
    board = Image.new("RGB", (4 * 198, len(CASES) * 118), "white")
    draw = ImageDraw.Draw(board)
    for i, case in enumerate(CASES):
        candidates = [r for r in rows if r["split"] == "val" and condition(r) == set(case.split("+"))]
        if not candidates:
            raise ValueError(f"Sin ejemplos de validación para {case}")
        chosen = random.Random(310).sample(candidates, min(4, len(candidates)))
        cases[case] = [r["pair_id"] for r in chosen]
        for j, r in enumerate(chosen):
            draw.text((j * 198, i * 118), f"{case} | {r['species_name']}", fill="black")
            for k, view in enumerate(("front_path", "back_path")):
                with Image.open(DATA / r[view]) as im:
                    board.paste(im, (j * 198 + k * 96, i * 118 + 18))
    board.save(PREPARED / "real_val.png")
    (PREPARED / "benchmark.json").write_text(json.dumps(dict(seeds=SEEDS, real_pairs=cases), indent=2))
    print({p: (c["pairs"], c["species"]) for p, c in counts.items()})


if __name__ == "__main__":
    main()
