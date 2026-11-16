"""Regenerate the selected historical sprites and compare both views to the original boards."""
import numpy as np
import torch
from PIL import Image

from interfaz import ROOT, WEIGHTS
from model import LATENT
from prepare import TYPES
from sample_cycles import load_generator


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    originals = ROOT / "estructura"
    with Image.open(originals/"ficha/fantasma_con_gpt.png") as ghosts, Image.open(
            originals/"perfiles_pareados/comparacion.png") as profiles:
        for model, column in (("ficha", 535), ("detalle", 335), ("perfiles", 535), ("pigmentos", 735)):
            if model == "ficha":
                board = ghosts
                cases = [(row, ["ghost"], seed) for row, seed in enumerate((96553, 121211, 121212))]
            else:
                board = profiles
                cases = [(1, ["dragon"], 318071), (2, ["fighting"], 318071), (6, ["bug", "poison"], 318071)]
            net = load_generator(WEIGHTS[model], "cpu")
            for row, types, seed in cases:
                z = torch.randn(1, LATENT, generator=torch.Generator().manual_seed(seed))
                tags = z.new_tensor([[t in types for t in TYPES]])
                views = ((net(z, tags).reshape(2, 3, 96, 96).clamp(-1, 1)+1)*127.5).round()
                for side, pixels in enumerate(views.permute(0, 2, 3, 1).numpy()):
                    x, y = column + 96*side, 74 + 104*row
                    reference = np.asarray(board.crop((x, y, x+96, y+96)).convert("RGB"), dtype=float)
                    first, second = pixels.min(2) < 220, reference.min(2) < 220
                    overlap = (first & second).sum() / (first | second).sum()
                    error = np.abs(pixels-reference).mean()
                    # The original boards were rendered in a batch on CUDA. Allow
                    # small CPU rasterization differences in outlines and shading.
                    min_overlap = .87 if model == "ficha" else .98
                    if overlap < min_overlap or error > 5:
                        raise AssertionError((model, types, seed, side, overlap, error))
                print(model, "+".join(types), seed, "OK")


if __name__ == "__main__":
    main()
