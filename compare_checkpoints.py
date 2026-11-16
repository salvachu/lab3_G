"""Compara las 16 parejas fijas sin reescalar ni filtrar los sprites mostrados."""
import csv
import itertools
import json
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prepare import PREPARED, ROOT

RUN = ROOT / "runs/baseline_5000"
STAGES = {"500": ROOT / "runs/pilot/step_00500.png",
          "2.500": ROOT / "runs/baseline/step_02500.png",
          "5.000": RUN / "step_05000.png"}
NAMES = {"water": "Agua", "fire": "Fuego", "grass+poison": "Planta / Veneno",
         "normal+flying": "Normal / Volador"}


def pairs(path, cases, seeds):
    with Image.open(path) as im:
        if im.size != (198*len(seeds), 118*len(cases)):
            raise ValueError(f"Dimensiones inesperadas en {path}")
        return [[im.crop((198*j, 118*i+18, 198*j+192, 118*i+114))
                 for j in range(len(seeds))] for i in range(len(cases))]


def font(size, bold=False):
    try:
        return ImageFont.truetype("segoeuib.ttf" if bold else "segoeui.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def contact_sheet(grids, cases, seeds):
    board = Image.new("RGB", (1328, 1034), "#e6e9ed")
    draw = ImageDraw.Draw(board)
    locations = []
    for i, case in enumerate(cases):
        x, y = 8 + (i % 2)*660, 8 + (i//2)*512
        draw.rectangle((x, y, x+651, y+503), fill="white")
        draw.text((x+12, y+3), NAMES[case], font=font(21, True), fill="#172535")
        for s, (label, grid) in enumerate(grids.items()):
            col = x+53 + s*197
            draw.text((col+96, y+30), f"{label} pasos", font=font(17, True), anchor="mt", fill="#172535")
            for k, view in enumerate(("frente", "espalda")):
                draw.text((col+48+k*96, y+54), view, font=font(13), anchor="mt", fill="#465461")
            for j, seed in enumerate(seeds):
                top = y+74+j*105
                if s == 0:
                    draw.text((x+5, top+40), f"z={seed}", font=font(13), fill="#263b50")
                img = grid[i][j]
                board.paste(img, (col, top))
                locations.append(((col, top, col+192, top+96), img))
                if j < len(seeds)-1:
                    draw.line((col, top+99, col+191, top+99), fill="#e7e9eb")
    path = RUN / "comparacion.png"
    board.save(path)
    with Image.open(path) as exported:
        for bounds, source in locations:
            if exported.crop(bounds).tobytes() != source.tobytes():
                raise ValueError("La lámina compuesta modificó un sprite")


def shape_mask(image):
    # Reduce el impacto de píxeles coloreados sueltos. Solo se usa para la métrica.
    foreground = np.asarray(image).min(-1) < 0.88*255
    cells = foreground.reshape(24, 4, 24, 4).sum(axis=(1, 3)) >= 4
    remaining = set(map(tuple, np.argwhere(cells)))
    largest = set()
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        group, queue = {first}, deque([first])
        while queue:
            y, x = queue.popleft()
            for neighbor in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    group.add(neighbor)
                    queue.append(neighbor)
        if len(group) > len(largest):
            largest = group
    normalized = Image.new("L", (24, 24))
    if largest:
        mask = Image.new("L", (24, 24))
        for y, x in largest:
            mask.putpixel((int(x), int(y)), 255)
        crop = mask.crop(mask.getbbox())
        factor = 24 / max(crop.size)
        w, h = (max(1, round(n*factor)) for n in crop.size)
        normalized.paste(crop.resize((w, h), Image.Resampling.NEAREST), ((24-w)//2, (24-h)//2))
    return np.asarray(normalized) > 0


def silhouette_stats(grids, cases, output=RUN / "siluetas.csv"):
    records = []
    for stage, grid in grids.items():
        for case, sprites in zip(cases, grid):
            for view, offset in (("frente", 0), ("espalda", 96)):
                masks = [shape_mask(im.crop((offset, 0, offset+96, 96))) for im in sprites]
                overlaps = []
                for a, b in itertools.combinations(masks, 2):
                    if a.any() and b.any():
                        overlaps.append((a & b).sum() / (a | b).sum())
                records.append(dict(stage=stage, condition=case, view=view, comparisons=len(overlaps),
                                    mean_iou=float(np.mean(overlaps)) if overlaps else None))
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    bench = json.loads((PREPARED / "benchmark.json").read_text())
    cases, seeds = list(bench["real_pairs"]), bench["seeds"]
    grids = {step: pairs(path, cases, seeds) for step, path in STAGES.items()}
    contact_sheet(grids, cases, seeds)
    grids["VAL"] = pairs(PREPARED / "real_val_roundtrip.png", cases, seeds)
    silhouette_stats(grids, cases)
