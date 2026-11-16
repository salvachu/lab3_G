import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from generate import render
from model import LATENT, Discriminator, Generator
from prepare import CASES, DATA, PREPARED, ROOT, SEEDS, TYPES, read_rows


def train_rows():
    with (PREPARED / "splits.csv").open(newline="") as f:
        ids = {r["pair_id"] for r in csv.DictReader(f) if r["split"] == "train"}
    rows = [r for r in read_rows() if r["pair_id"] in ids]
    if not rows or len(rows) != len(ids):
        raise ValueError("Repite prepare.py para verificar la partición")
    return rows


def batch(rows, indices, device):
    pairs, labels = [], []
    for idx in indices:
        row = rows[idx]
        views = []
        for key in ("front_path", "back_path"):
            with Image.open(DATA / row[key]) as image:
                views.append(np.asarray(image))
        pairs.append(np.concatenate(views, axis=-1))
        labels.append([int(row["type_" + t]) for t in TYPES])
    pixels = torch.from_numpy(np.stack(pairs)).permute(0, 3, 1, 2).to(device, dtype=torch.float32)
    return pixels.div_(127.5).sub_(1), torch.tensor(labels, device=device, dtype=torch.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--steps", type=int, default=300, help="Paso total de destino")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--seed", type=int, default=310)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    if min(a.steps, a.batch, a.save_every) < 1 or a.run.exists():
        p.error("Usa una carpeta de salida nueva y valores positivos")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    rows = train_rows()
    g, d = Generator().to(device), Discriminator().to(device)
    go = torch.optim.Adam(g.parameters(), lr=0.0002, betas=(0.0, 0.9))
    do = torch.optim.Adam(d.parameters(), lr=0.0002, betas=(0.0, 0.9))
    signature = hashlib.sha256((PREPARED / "splits.csv").read_bytes() + (DATA / "pairs.csv").read_bytes()).hexdigest()
    start = 0
    if a.resume:
        state = torch.load(a.resume, map_location="cpu", weights_only=True)
        if (state["data"], state["seed"], state["batch"]) != (signature, a.seed, a.batch):
            p.error("La continuación requiere los mismos datos, seed y batch")
        g.load_state_dict(state["generator"])
        d.load_state_dict(state["discriminator"])
        go.load_state_dict(state["go"])
        do.load_state_dict(state["do"])
        start = state["step"]
        if start >= a.steps:
            p.error("--steps debe superar el paso del checkpoint")
    a.run.mkdir(parents=True)
    config = dict(vars(a), device=str(device), gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                  torch=str(torch.__version__), data=signature, train_pairs=len(rows),
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
                                 for n in ("prepare.py", "model.py", "train.py", "generate.py")})
    (a.run / "config.json").write_text(json.dumps(config, indent=2, default=str))
    begin = time.monotonic()
    with (a.run / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "d_hinge", "g_loss", "seconds", "peak_gpu_MiB"])
        for step in range(start + 1, a.steps + 1):
            rng = np.random.default_rng([a.seed, step])
            indices = rng.integers(len(rows), size=a.batch)
            real, y = batch(rows, indices, device)
            latent_rng = torch.Generator(device=device).manual_seed(int(rng.integers(2**62)))
            z = torch.randn(a.batch, LATENT, device=device, generator=latent_rng)
            fake = g(z, y)
            d.requires_grad_(True)
            do.zero_grad(set_to_none=True)
            scores = d(torch.cat([real, fake.detach()]), torch.cat([y, y]))
            real_score, fake_score = scores.chunk(2)
            dl = F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
            dl.backward()
            torch.nn.utils.clip_grad_norm_(d.parameters(), float("inf"), error_if_nonfinite=True)
            do.step()
            d.requires_grad_(False)
            go.zero_grad(set_to_none=True)
            gl = -d(fake, y).mean()
            gl.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), float("inf"), error_if_nonfinite=True)
            go.step()
            peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
            writer.writerow([step, dl.item(), gl.item(), round(time.monotonic() - begin, 2), round(peak, 1)])
            if step % a.save_every == 0 or step == a.steps:
                f.flush()
                checkpoint = a.run / f"step_{step:05}.pt"
                temp = checkpoint.with_suffix(".tmp")
                torch.save(dict(step=step, generator=g.state_dict(), discriminator=d.state_dict(),
                                go=go.state_dict(), do=do.state_dict(), data=signature, seed=a.seed, batch=a.batch), temp)
                temp.replace(checkpoint)
                render(g, CASES, SEEDS, checkpoint.with_suffix(".png"))
                print(f"{step}/{a.steps} | D {dl.item():.3f} | G {gl.item():.3f}", flush=True)


if __name__ == "__main__":
    main()
