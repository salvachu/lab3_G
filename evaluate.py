"""Diagnósticos para la lámina fija de frente/espalda. Valores RGB en [0, 1]."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prepare import PREPARED


def read_grid(path, rows, cols):
    with Image.open(path) as grid:
        if grid.size != (198 * cols, 118 * rows):
            raise ValueError(f"Dimensiones inesperadas en {path}")
        array = np.asarray(grid.convert("RGB"), dtype=np.float32) / 255
    return np.array([[array[118*i+18:118*i+114, 198*j:198*j+192].reshape(96, 2, 96, 3)
                      .transpose(1, 0, 2, 3) for j in range(cols)] for i in range(rows)])


def describe(pairs):
    mask = pairs.min(axis=-1) < 0.88
    palettes = (pairs * mask[..., None]).sum(axis=(2, 3)) / np.maximum(mask.sum(axis=(2, 3))[..., None], 1)
    diffs = [np.abs(a - b).mean() for a, b in itertools.combinations(pairs, 2)]
    return dict(front_foreground=float(mask[:, 0].mean()), back_foreground=float(mask[:, 1].mean()),
                between_creatures_l1=float(np.mean(diffs)),
                front_back_l1=float(np.abs(pairs[:, 0] - pairs[:, 1]).mean()),
                front_back_mean_color_gap=float(np.abs(palettes[:, 0] - palettes[:, 1]).mean()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("grid", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    bench = json.loads((PREPARED / "benchmark.json").read_text())
    cases = list(bench["real_pairs"])
    seeds = bench["seeds"]
    data = []
    for name, path in [("val", PREPARED / "real_val.png"), ("generated", args.grid)]:
        grid = read_grid(path, len(cases), len(seeds))
        for idx, (case, pairs) in enumerate(zip(cases, grid)):
            n = len(bench["real_pairs"][case]) if name == "val" else len(seeds)
            across_types = [np.abs(pairs - other).mean() for j, other in enumerate(grid) if idx != j]
            data.append(dict(source=name, condition=case, **describe(pairs[:n]),
                             same_seed_across_types_l1=float(np.mean(across_types)) if name == "generated" else None))
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0])
        writer.writeheader()
        writer.writerows(data)
