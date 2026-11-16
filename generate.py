import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from model import LATENT, Generator
from prepare import CASES, SEEDS, TYPES


@torch.inference_mode()
def render(model, cases, seeds, output):
    device = next(model.parameters()).device
    training = model.training
    model.eval()
    noise = torch.stack([torch.randn(LATENT, generator=torch.Generator().manual_seed(s))
                         for s in seeds]).to(device)
    board = Image.new("RGB", (198 * len(seeds), 118 * len(cases)), "white")
    draw = ImageDraw.Draw(board)
    for i, case in enumerate(cases):
        selected = case.split("+")
        if len(selected) not in (1, 2) or len(set(selected)) != len(selected) or set(selected) - set(TYPES):
            raise ValueError(f"Usa uno o dos tipos distintos de {', '.join(TYPES)}")
        y = torch.tensor([t in selected for t in TYPES], dtype=torch.float32, device=device)
        pairs = model(noise, y.expand(len(seeds), -1)).reshape(-1, 2, 3, 96, 96)
        pixels = ((pairs.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy()
        for j, seed in enumerate(seeds):
            draw.text((198 * j, 118 * i), f"{case} | z={seed}", fill="black")
            for view in (0, 1):
                board.paste(Image.fromarray(pixels[j, view]), (198 * j + 96 * view, 118 * i + 18))
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output)
    model.train(training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--types", nargs="+", default=CASES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output", type=Path, default=Path("samples.png"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Generator().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["generator"])
    render(model, args.types, args.seeds, args.output)
