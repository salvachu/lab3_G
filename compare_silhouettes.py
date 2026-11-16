"""Monta las dos corridas a 5.000, con ampliación entera y sin filtrar los píxeles."""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from compare_checkpoints import NAMES, font, pairs, silhouette_stats
from prepare import PREPARED, ROOT

RUN = ROOT / "runs/siluetas_5000"
CONTROL = ROOT / "runs/baseline_5000"


def board(grids, cases, seeds, output, blind=False):
    width, height = 895, 985
    board = Image.new("RGB", (2*width+24, 2*height+110), "#e7ebef")
    draw = ImageDraw.Draw(board)
    title = "Revisión visual" if blind else "Original y auxiliar de siluetas"
    draw.text((22, 12), f"{title}   ·   5.000 pasos", font=font(34, True), fill="#182c40")
    rng = random.Random(8601)
    checks = []
    key = {}
    for i, case in enumerate(cases):
        x, y = 8+(i%2)*(width+8), 78+(i//2)*(height+8)
        draw.rectangle((x, y, x+width-1, y+height-1), fill="white")
        draw.text((x+19, y+6), NAMES[case], font=font(32, True), fill="#193348")
        for j, seed in enumerate(seeds):
            labels = list(grids)
            if blind:
                rng.shuffle(labels)
            key[f"{case}/{seed}"] = labels
            top = y+134+j*208
            draw.text((x+8, top+82), f"z={seed}", font=font(20, True), fill="#45545e")
            for s, label in enumerate(labels):
                left = x+73+s*403
                if j == 0:
                    heading = ["A", "B"][s] if blind else label
                    draw.text((left+192, y+56), heading, font=font(25, True), anchor="mt", fill="#213a4a")
                    for k, view in enumerate(("frente", "espalda")):
                        draw.text((left+96+k*192, y+96), view, font=font(21), anchor="mt", fill="#526778")
                pair = grids[label][i][j]
                scaled = pair.resize((384, 192), Image.Resampling.NEAREST)
                board.paste(scaled, (left, top))
                checks.append(((left, top, left+384, top+192), pair))
            if j+1 < len(seeds):
                draw.line((x+14, top+198, x+width-16, top+198), fill="#e2e6e9", width=2)
    board.save(output)
    with Image.open(output) as result:
        for box, original in checks:
            expected = original.resize((384, 192), Image.Resampling.NEAREST)
            if result.crop(box).tobytes() != expected.tobytes():
                raise ValueError("La imagen comparativa no conserva los sprites")
    return key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-dir", type=Path, help="Carpeta para revisión con A/B aleatorizados por pareja")
    args = parser.parse_args()
    manifest = json.loads((PREPARED / "benchmark.json").read_text())
    cases, seeds = list(manifest["real_pairs"]), manifest["seeds"]
    grids = {name: pairs(folder / "step_05000.png", cases, seeds)
             for name, folder in (("Original", CONTROL), ("Con siluetas", RUN))}
    board(grids, cases, seeds, RUN / "comparacion_5000.png")
    silhouette_stats(grids, cases, RUN / "solapamiento.csv")
    if args.blind_dir:
        args.blind_dir.mkdir(parents=True, exist_ok=True)
        key = board(grids, cases, seeds, args.blind_dir / "ab.png", blind=True)
        (args.blind_dir / "clave.json").write_text(json.dumps(key, indent=2))
