"""Learn fine strokes from held-in real sprite pairs with deliberately simplified interiors."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from generate import render
from sprite_detail import DetailedGenerator
from train import batch, train_rows
from train_aegan_pairs import stroke_loss


def simplify(pairs, step):
    views = pairs.reshape(-1, 3, 96, 96)
    mask = (views.amin(1, keepdim=True) < .97).float()
    radius = (3, 5, 5)[step % 3]
    smooth = F.avg_pool2d(views * mask, radius, 1, radius // 2)
    smooth = smooth / F.avg_pool2d(mask, radius, 1, radius // 2).clamp_min(.01)
    # Retain some high-contrast marks, as in the drawings we will actually refine.
    mix = (.32, .54, .71, .85)[step % 4]
    flat = (views.lerp(smooth, mix) * 5).round() / 5
    return (flat * mask + 1 - mask).clamp(-1, 1).reshape_as(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(46)
    state = torch.load(args.base, map_location="cpu", weights_only=True)
    model = DetailedGenerator().cuda()
    model.base.load_state_dict(state["generator"])
    model.base.requires_grad_(False)
    opt = torch.optim.AdamW(model.detail.parameters(), lr=.0008)
    rows = train_rows()
    with (args.out / "metrics.csv").open("w", newline="") as stream:
        log = csv.writer(stream)
        log.writerow(["step", "figure_l1", "stroke"])
        for step in range(1, args.steps + 1):
            rng = np.random.default_rng([46, step])
            real, _ = batch(rows, rng.integers(len(rows), size=12), "cuda")
            prediction = model.detail(simplify(real, step))
            target = real.reshape(-1, 3, 96, 96)
            mask = target.amin(1) < .97
            error = (prediction.reshape_as(target) - target).abs().mean(1)
            figure = (mask * error).sum() / mask.sum().clamp_min(1)
            strokes = stroke_loss(real, prediction)
            opt.zero_grad(set_to_none=True)
            (figure + .55 * strokes).backward()
            torch.nn.utils.clip_grad_norm_(model.detail.parameters(), 5, error_if_nonfinite=True)
            opt.step()
            if step % 100 == 0 or step == args.steps:
                log.writerow([step, figure.item(), strokes.item()])
                stream.flush()
    checkpoint = args.out / f"detalle_{args.steps:05}.pt"
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(dict(architecture="paired_learned_detail", generator=model.state_dict(),
                    base_step=state["step"], detail_steps=args.steps, base=str(args.base)), temporary)
    temporary.replace(checkpoint)
    render(model, ["fire", "water+dragon", "bug+poison"], [96137, 96553, 97001, 97367, 97931, 98317],
           checkpoint.with_suffix(".png"))


if __name__ == "__main__":
    main()
