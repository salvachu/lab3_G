"""Fit a conditional flow to paired body measurements from the training partition."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sprite_diversa import BodyPrior, DiverseGenerator, FAMILIES, PATCH, SIZE
from sprite_ficha import measure
from sprite_profiles import SHAPE, describe
from train_estructura import dataset


def measurements(data, device):
    fields = []
    for batch in data["images"].split(96):
        pair = batch.to(device).float() / 127.5 - 1
        outline = describe(pair)[0].flatten(1)
        views = pair.flatten(0, 1)
        mask = (views.amin(1, keepdim=True) < .90).float()
        gray = (views.mean(1, keepdim=True) + 1) / 2
        avg = (gray * mask).sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True).clamp_min(1)
        shading = F.adaptive_avg_pool2d((gray - avg) * mask, PATCH)
        shading /= F.adaptive_avg_pool2d(mask, PATCH).clamp_min(.12)
        fields.append(torch.cat([outline, measure(pair), shading.reshape(len(pair), -1)], 1))
    return torch.cat(fields)


def save(state, path):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2600)
    args = parser.parse_args()
    out = Path("runs/diversa")
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.manual_seed(42900)
    data = dataset()
    annotations = torch.load("estructura/anotacion_train/prior.pt", weights_only=True)
    if annotations["data"] != data["data"]:
        raise ValueError("Las etiquetas de anatomía pertenecen a otra partición")
    groups = F.one_hot(annotations["groups"].to(device), FAMILIES).float()
    y = data["y"].to(device)
    target = torch.cat([groups, measurements(data, device)], 1)
    species = torch.as_tensor(np.unique(data["species"], return_inverse=True)[1], device=device)
    species_weight = 1 / torch.bincount(species)[species]
    frequency = (y * species_weight[:, None]).sum(0)
    weights = species_weight * (y @ frequency.rsqrt()) / y.sum(1)
    weights /= weights.sum()
    net = BodyPrior().to(device)
    net.mean.copy_(weights @ target)
    net.std.copy_((weights @ (target - net.mean).square()).sqrt().clamp_min(.045))
    target = ((target - net.mean) / net.std).clamp(-5, 5)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0008)
    checkpoint = out / "checkpoint.pt"
    start = 0
    if checkpoint.exists():
        previous = torch.load(checkpoint, map_location=device, weights_only=True)
        if previous["data"] != data["data"]:
            raise ValueError("Cambió la partición de entrenamiento")
        net.load_state_dict(previous["net"])
        ema.load_state_dict(previous["ema"])
        optimizer.load_state_dict(previous["optimizer"])
        start = previous["step"]
    print(len(y), "pares;", len(torch.unique(species)), "especies", flush=True)
    for step in range(start+1, args.steps+1):
        rng = torch.Generator(device=device).manual_seed(9200000 + step)
        ix = torch.multinomial(weights, 192, replacement=True, generator=rng)
        noise = torch.randn((len(ix), SIZE), generator=rng, device=device)
        time = torch.rand((len(ix), 1), generator=rng, device=device)
        velocity = net(torch.lerp(noise, target[ix], time), time, y[ix])
        errors = (velocity - (target[ix] - noise)).square()
        sections = errors.split([FAMILIES, SHAPE, 6, 2*PATCH**2], dim=1)
        loss = sum(w * field.mean() for w, field in zip((.26, .36, .23, .15), sections))
        optimizer.param_groups[0]["lr"] = .0008 / (1 + step/1800)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), list(net.parameters()), .02)
        if step % 400 == 0 or step == args.steps:
            save(dict(step=step, net=net.state_dict(), ema=ema.state_dict(),
                      optimizer=optimizer.state_dict(), data=data["data"]), checkpoint)
            print(step, round(loss.item(), 4), flush=True)
    model = DiverseGenerator()
    for module, path in ((model.ficha, "ficha"), (model.insect, "ficha_todos")):
        frozen = torch.load(f"pesos/{path}.pth", map_location="cpu", weights_only=True)
        module.load_state_dict(frozen["generator"])
    model.prior.load_state_dict(ema.state_dict())
    painter_state = torch.load("pesos/trazo.pth", map_location="cpu", weights_only=True)["generator"]
    model.painter.load_state_dict({k.removeprefix("painter."):v for k,v in painter_state.items() if k.startswith("painter.")})
    save(dict(architecture="paired_diverse", generator=model.state_dict(), data=data["data"],
              prior_steps=max(start, args.steps)), Path("pesos/diversa.pth"))


if __name__ == "__main__":
    main()
