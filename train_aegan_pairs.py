"""Conditional AEGAN pilot trained directly on paired 96 px sprites."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from generate import render
from model import Discriminator, Generator, LATENT
from prepare import TYPES
from train import batch, train_rows
from train_pairs_cycle import PairEncoder, reconstruction


def stroke_loss(real, restored):
    # Give sparse, contrasting marks inside the sprite a voice in reconstruction.
    views = real.reshape(-1, 3, 96, 96)
    predicted = restored.reshape_as(views)
    foreground = views.amin(1, keepdim=True) < 0.90
    losses = []
    for axis in (2, 3):
        before = foreground.narrow(axis, 0, foreground.size(axis) - 1)
        after = foreground.narrow(axis, 1, foreground.size(axis) - 1)
        difference = views.diff(dim=axis)
        inside_edge = before & after & (difference.abs().amax(1, keepdim=True) > 0.30)
        outer_edge = before ^ after
        error = (predicted.diff(dim=axis) - difference).abs().mean(1, keepdim=True)
        for edge in (inside_edge, outer_edge):
            losses.append(((error * edge).sum((2, 3)) / edge.sum((2, 3)).clamp_min(1)).mean())
    return torch.stack(losses).mean()


def train(folder, steps, size, resume=None, stroke_weight=0.0):
    if min(steps, size) < 1:
        raise ValueError("El lote y el número de pasos deben ser positivos")
    torch.manual_seed(310)
    rows = train_rows()
    g, d, e = Generator().cuda(), Discriminator().cuda(), PairEncoder().cuda()
    # PairEncoder also supports variational training. Here only its mean is used.
    e.output = nn.Linear(e.output.in_features, LATENT).cuda()
    critic = nn.Sequential(nn.Linear(LATENT + len(TYPES), 192), nn.LeakyReLU(0.2),
                           nn.Linear(192, 1)).cuda()
    disc_params = [*d.parameters(), *critic.parameters()]
    gen_params = [*g.parameters(), *e.parameters()]
    od, og = [torch.optim.Adam(p, lr=0.0002, betas=(0.0, 0.9))
              for p in (disc_params, gen_params)]
    initial = 0
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        initial = state["step"]
        if size != state.get("batch", 16) or initial >= steps:
            raise ValueError("La continuación requiere el mismo lote y un paso final mayor")
        for module, key in ((g, "generator"), (e, "encoder"), (d, "discriminator"),
                            (critic, "latent_critic"), (og, "og"), (od, "od")):
            module.load_state_dict(state[key])
        if "cuda_rng" in state:
            torch.cuda.set_rng_state(state["cuda_rng"])
        else:
            # The pilot drew exactly one CUDA noise tensor per step.
            for _ in range(initial):
                torch.randn(size, LATENT, device="cuda")
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with (folder / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "reconstruction", "latent_cycle", "seconds"])
        for step in range(initial + 1, steps + 1):
            rng = np.random.default_rng([310, step])
            real, y = batch(rows, rng.integers(len(rows), size=size), "cuda")
            z = torch.randn(size, LATENT, device="cuda")
            synthetic = g(z, y)
            encoded = e.output(e.body(real))
            restored = g(encoded, y)
            d.requires_grad_(True)
            critic.requires_grad_(True)
            od.zero_grad(set_to_none=True)
            real_score, fake_score = d(torch.cat([real, synthetic.detach()]), y.repeat(2, 1)).chunk(2)
            latent_scores = critic(torch.cat([torch.cat([z, encoded.detach()]), y.repeat(2, 1)], dim=1))
            prior_score, encoded_score = latent_scores.chunk(2)
            loss_d = (F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
                      + F.relu(1 - prior_score).mean() + F.relu(1 + encoded_score).mean())
            loss_d.backward()
            od.step()
            d.requires_grad_(False)
            critic.requires_grad_(False)
            og.zero_grad(set_to_none=True)
            rec = reconstruction(real, restored)
            cycle = F.l1_loss(e.output(e.body(synthetic)), z)
            loss = (8 * rec + cycle - d(synthetic, y).mean()
                    - critic(torch.cat([encoded, y], dim=1)).mean())
            if stroke_weight:
                loss = loss + stroke_weight * stroke_loss(real, restored)
            loss.backward()
            nn.utils.clip_grad_norm_(gen_params, 10, error_if_nonfinite=True)
            og.step()
            if step % 25 == 0 or step == steps:
                writer.writerow([step, rec.item(), cycle.item(), round(time.monotonic() - start, 2)])
                stream.flush()
            if step in (100, 500, steps) or step % 1000 == 0:
                target = folder / f"step_{step:05}.pt"
                temporary = target.with_suffix(".tmp")
                torch.save(dict(step=step, generator=g.state_dict(), encoder=e.state_dict(),
                                discriminator=d.state_dict(), latent_critic=critic.state_dict(),
                                og=og.state_dict(), od=od.state_dict(), batch=size,
                                cuda_rng=torch.cuda.get_rng_state(), stroke_weight=stroke_weight), temporary)
                temporary.replace(target)
                render(g, ["fire", "water+dragon", "bug+poison"],
                       [96137, 96553, 97001, 97367, 97931, 98317], target.with_suffix(".png"))
                print(f"{step} steps in {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stroke-weight", type=float, default=0.0)
    args = parser.parse_args()
    train(args.folder, args.steps, args.batch, args.resume, args.stroke_weight)
