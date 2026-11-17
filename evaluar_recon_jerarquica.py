"""Evalúa la reconstrucción de ambas vistas en una pareja por especie de validación."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from estructura_vecinos import normalized, pair_image
from prepare import PREPARED, TYPES, read_rows
from sprite_jerarquica import HierarchicalGenerator, PosteriorEncoder


@torch.inference_mode()
def evaluate(path, images, types, ema=False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    decoder = HierarchicalGenerator().cuda().eval()
    decoder.load_state_dict(state["generator" if ema else "net"])
    encoder = PosteriorEncoder().cuda().eval()
    encoder.load_state_dict(state["encoder"])
    overlaps, errors, details = [], [], []
    for batch, tags in zip(images.split(24), types.split(24)):
        x = batch.cuda().float() / 127.5 - 1
        zeros = [x.new_zeros(len(x), k, s, s) for k, s in zip(decoder.latent, decoder.sizes)]
        logits, colors, _ = decoder.decode(tags.cuda(), zeros, encoder(x))
        output = decoder.rgb(logits, colors).reshape_as(x)
        actual = x.amin(2) < .92
        predicted = logits[:, :, 0] > 0
        overlaps.append((actual & predicted).sum((-2, -1)) / (actual | predicted).sum((-2, -1)).clamp_min(1))
        def masked_error(mask):
            difference = (x - output).abs().mean(2) * .5
            count = mask.sum((-2, -1))
            return (difference * mask).sum((-2, -1)) / count.masked_fill(count == 0, 1)
        errors.append(masked_error(actual))
        # Exclude the exterior edge and evaluate internal high-contrast pixels.
        light = (x.mean(2) + 1) / 2
        contrast = F.max_pool2d(light, 5, 1, 2) + F.max_pool2d(-light, 5, 1, 2)
        interior = F.max_pool2d((~actual).float(), 5, 1, 2) == 0
        edges = interior & (contrast > .24)
        details.append(masked_error(edges)[edges.any((-2, -1))])
    return {"pasos":state["step"], "ema":ema, "especies":len(images), "IoU_silueta":torch.cat(overlaps).mean().item(),
            "error_RGB_figura":torch.cat(errors).mean().item(), "error_RGB_detalles":torch.cat(details).mean().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("estructura/empalme/reconstruccion.json"))
    parser.add_argument("--ultimo-ema",action="store_true")
    args = parser.parse_args()
    with (PREPARED / "splits.csv").open() as file:
        val = {r["pair_id"] for r in csv.DictReader(file) if r["split"] == "val"}
    species = {}
    for row in sorted(read_rows(), key=lambda r:r["pair_id"]):
        if row["pair_id"] in val:
            species.setdefault(row["species_id"], row)
    images = torch.from_numpy(np.stack([normalized(pair_image(row)) for row in species.values()]))
    types = torch.tensor([[float(r["type_" + t]) for t in TYPES] for r in species.values()])
    report = [evaluate(path, images, types, args.ultimo_ema and i==len(args.checkpoints)-1)
              for i,path in enumerate(args.checkpoints)]
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(report)


if __name__ == "__main__":
    main()
