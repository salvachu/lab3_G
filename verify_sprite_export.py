"""Pasa los pares VAL por train.batch y generate.render y compara el PNG con los originales."""
import hashlib
import json

import numpy as np
import torch
from PIL import Image, ImageDraw

from generate import render
from prepare import CASES, DATA, PREPARED, ROOT, SEEDS, TYPES, read_rows
from train import batch


class ValidationPairs(torch.nn.Module):
    """Adaptador para que el exportador reciba el lote real ya normalizado."""

    def __init__(self, rows, device):
        super().__init__()
        self.device_marker = torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False)
        self.rows = rows

    def forward(self, noise, types):
        case = frozenset(t for t, active in zip(TYPES, types[0].tolist()) if active)
        rows = self.rows[case]
        pixels, labels = batch(rows, range(len(rows)), types.device)
        if not torch.equal(labels, types):
            raise ValueError("El orden de tipos del benchmark no coincide")
        return pixels


def main():
    manifest = json.loads((PREPARED / "benchmark.json").read_text())
    rows = {r["pair_id"]: r for r in read_rows()}
    chosen = {case: [rows[i] for i in ids] for case, ids in manifest["real_pairs"].items()}
    if list(chosen) != CASES or manifest["seeds"] != SEEDS:
        raise ValueError("El benchmark no coincide con la configuración de las láminas generadas")
    out = PREPARED / "real_val_roundtrip.png"
    adapter = ValidationPairs({frozenset(c.split("+")): rs for c, rs in chosen.items()}, "cuda")
    render(adapter, CASES, SEEDS, out)
    # Las cuatro columnas contienen ejemplos reales. Sustituir solo sus rótulos de semilla.
    with Image.open(out) as exported:
        board = exported.copy()
    draw = ImageDraw.Draw(board)
    for i, (case, records) in enumerate(chosen.items()):
        draw.rectangle((0, 118 * i, board.width - 1, 118 * i + 17), fill="white")
        for j, row in enumerate(records):
            draw.text((198*j, 118*i), f"{case} | {row['species_name']}", fill="black")
    board.save(out)
    checked = []
    with Image.open(out) as result:
        for i, records in enumerate(chosen.values()):
            for j, row in enumerate(records):
                for v, key in enumerate(("front_path", "back_path")):
                    x, y = 198*j + 96*v, 118*i + 18
                    restored = np.asarray(result.crop((x, y, x+96, y+96)), dtype=np.int16)
                    with Image.open(DATA / row[key]) as src:
                        original = np.asarray(src, dtype=np.int16)
                    diff = np.abs(original - restored)
                    checked.append(dict(pair=row["pair_id"], view=key, changed_pixels=int(diff.any(-1).sum()),
                                        max_channel_error=int(diff.max())))
    errors = sum(v["changed_pixels"] for v in checked)
    config = json.loads((ROOT / "runs/baseline/config.json").read_text())
    for name, expected in config["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"El código del baseline ha cambiado: {name}")
    report = dict(views=len(checked), pixels=len(checked)*96*96, changed_pixels=errors, details=checked)
    (PREPARED / "real_val_roundtrip.json").write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "details"})
    if len(checked) != 32 or errors:
        raise ValueError("Falló la verificación píxel a píxel")


if __name__ == "__main__":
    main()
