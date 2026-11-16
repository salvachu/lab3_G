"""Continuación controlada desde 2.500 con un término adversarial sobre cada silueta."""
import csv
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm

from generate import render
from model import LATENT, Discriminator, Generator
from prepare import CASES, DATA, PREPARED, ROOT, SEEDS
from train import batch, train_rows

OUT = ROOT / "runs/siluetas_5000"
CONTROL = ROOT / "runs/baseline_5000"
WEIGHT = 0.5
THRESHOLD = 0.88
TEMPERATURE = 0.035


def silhouettes(rgb_pairs):
    views = rgb_pairs.reshape(-1, 3, 96, 96)
    darkest = (views.amin(dim=1, keepdim=True) + 1) * 0.5
    occupancy = ((THRESHOLD - darkest) / TEMPERATURE).sigmoid()
    return F.avg_pool2d(occupancy, 4)


class ShapeCritic(nn.Sequential):
    def __init__(self):
        layers = []
        for src, dst in ((1, 24), (24, 48), (48, 64)):
            layers.extend([spectral_norm(nn.Conv2d(src, dst, 4, stride=2, padding=1)),
                           nn.LeakyReLU(0.2)])
        super().__init__(*layers, nn.Flatten(), spectral_norm(nn.Linear(64*3*3, 1)))


def hinge(scores):
    real, fake = scores.chunk(2)
    return F.relu(1 - real).mean() + F.relu(1 + fake).mean()


def update(loss, module, optimizer):
    loss.backward()
    nn.utils.clip_grad_norm_(module.parameters(), float("inf"), error_if_nonfinite=True)
    optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modular-tipos", action="store_true")
    args = parser.parse_args()
    out = ROOT / "runs/tipos_5000" if args.modular_tipos else OUT
    cfg = json.loads((CONTROL / "config.json").read_text())
    start_path = ROOT / cfg["resume"]
    if out.exists():
        raise FileExistsError(f"La corrida ya existe: {out}")
    device = torch.device(cfg["device"])
    seed, size = cfg["seed"], cfg["batch"]
    torch.manual_seed(seed)
    rows = train_rows()
    for name, digest in cfg["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Cambió el código del control: {name}")
    data_hash = hashlib.sha256((PREPARED / "splits.csv").read_bytes() + (DATA / "pairs.csv").read_bytes()).hexdigest()
    state = torch.load(start_path, map_location="cpu", weights_only=True)
    if (state["step"], state["seed"], state["batch"], state["data"]) != (2500, seed, size, data_hash):
        raise ValueError("El checkpoint inicial no corresponde al experimento")
    if data_hash != cfg["data"] or len(rows) != cfg["train_pairs"] or cfg["steps"] != 5000:
        raise ValueError("Los datos o el presupuesto difieren del control")

    g, d = Generator().to(device), Discriminator().to(device)
    g.load_state_dict(state["generator"])
    d.load_state_dict(state["discriminator"])
    adam = dict(lr=0.0002, betas=(0.0, 0.9))
    go, do = torch.optim.Adam(g.parameters(), **adam), torch.optim.Adam(d.parameters(), **adam)
    go.load_state_dict(state["go"])
    do.load_state_dict(state["do"])
    shape = ShapeCritic().to(device)
    so = torch.optim.Adam(shape.parameters(), **adam)
    if args.modular_tipos:
        from type_generator import TypeGenerator
        typed = TypeGenerator().to(device)
        missing, unexpected = typed.load_state_dict(g.state_dict(), strict=False)
        if unexpected or set(missing) != {f"affines.{i}.weight" for i in range(4)}:
            raise ValueError("Pesos iniciales incompatibles")
        inherited = [v for n, v in typed.named_parameters() if not n.startswith("affines.")]
        go = torch.optim.Adam(inherited, **adam)
        go.load_state_dict(state["go"])
        go.add_param_group({"params": typed.affines.parameters()})
        g = typed

    out.mkdir()
    cfg.update(run=str(out.relative_to(ROOT)), silhouette_weight=WEIGHT, rgb_threshold=THRESHOLD,
               sigmoid_temperature=TEMPERATURE, mask_size=24, auxiliary_adam=adam,
               initial_sha256=hashlib.sha256(start_path.read_bytes()).hexdigest())
    cfg["source_sha256"][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cfg["type_modulation"] = args.modular_tipos
    if args.modular_tipos:
        cfg["source_sha256"]["type_generator.py"] = hashlib.sha256((ROOT/"type_generator.py").read_bytes()).hexdigest()
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    begin = time.monotonic()
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "d_hinge", "d_shape", "g_rgb", "g_shape", "g_loss", "seconds", "peak_gpu_MiB"])
        for step in range(state["step"] + 1, cfg["steps"] + 1):
            # Misma secuencia de índices y ruido por actualización que train.py.
            rng = np.random.default_rng([seed, step])
            indices = rng.integers(len(rows), size=size)
            real, y = batch(rows, indices, device)
            latent_rng = torch.Generator(device=device).manual_seed(int(rng.integers(2**62)))
            z = torch.randn(size, LATENT, device=device, generator=latent_rng)
            fake = g(z, y)
            d.requires_grad_(True)
            do.zero_grad(set_to_none=True)
            dl = hinge(d(torch.cat([real, fake.detach()]), torch.cat([y, y])))
            update(dl, d, do)
            d.requires_grad_(False)

            fake_masks = silhouettes(fake)
            shape.requires_grad_(True)
            so.zero_grad(set_to_none=True)
            sl = hinge(shape(torch.cat([silhouettes(real), fake_masks.detach()])))
            update(sl, shape, so)
            shape.requires_grad_(False)

            go.zero_grad(set_to_none=True)
            rgb_loss = -d(fake, y).mean()
            shape_loss = -shape(fake_masks).mean()
            gl = rgb_loss + WEIGHT * shape_loss
            update(gl, g, go)
            peak = torch.cuda.max_memory_allocated() / 2**20
            writer.writerow([step, dl.item(), sl.item(), rgb_loss.item(), shape_loss.item(), gl.item(),
                             round(time.monotonic() - begin, 2), round(peak, 1)])
            if step % cfg["save_every"] == 0 or step == cfg["steps"]:
                f.flush()
                checkpoint = out / f"step_{step:05}.pt"
                temp = checkpoint.with_suffix(".tmp")
                torch.save(dict(step=step, generator=g.state_dict(), discriminator=d.state_dict(),
                                go=go.state_dict(), do=do.state_dict(), shape=shape.state_dict(), so=so.state_dict(),
                                seed=seed, batch=size, data=data_hash, silhouette_weight=WEIGHT,
                                type_modulation=args.modular_tipos), temp)
                temp.replace(checkpoint)
                render(g, CASES, SEEDS, checkpoint.with_suffix(".png"))
                print(f"{step}/{cfg['steps']} | G RGB {rgb_loss.item():.3f} | G silueta {shape_loss.item():.3f}", flush=True)


if __name__ == "__main__":
    main()
