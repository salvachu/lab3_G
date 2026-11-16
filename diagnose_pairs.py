"""Compara reconstrucción VAL y muestreo libre para identificar dónde se pierde la forma."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from gallery_cycles import draw_board
from model import Generator, LATENT
from prepare import PREPARED, read_rows
from train import batch
from train_pairs_cycle import PairEncoder


def rgb_pairs(x):
    arr=((x.clamp(-1,1)+1)*127.5).round().byte().reshape(-1,2,3,96,96)
    arr=arr.permute(0,3,1,4,2).reshape(-1,96,192,3).cpu().numpy()
    return [Image.fromarray(im) for im in arr]


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("checkpoint",type=Path)
    args=parser.parse_args()
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    g,e=Generator().cuda().eval(),PairEncoder().cuda().eval()
    g.load_state_dict(state["generator"])
    e.load_state_dict(state["encoder"])
    manifest=json.loads((PREPARED/"benchmark.json").read_text())
    rows={r["pair_id"]:r for r in read_rows()}
    records=[]
    for case,ids in manifest["real_pairs"].items():
        real,y=batch([rows[key] for key in ids],range(len(ids)),"cuda")
        mu,lv=e(real)
        z=torch.stack([torch.randn(LATENT,generator=torch.Generator().manual_seed(s))
                       for s in manifest["seeds"]]).cuda()
        panels=[rgb_pairs(x) for x in (real,g(mu,y),g(z,y))]
        for j in range(2):
            records.append((case,manifest["seeds"][j],[p[j] for p in panels]))
    draw_board(args.checkpoint.parent/"diagnostico_val.png",["Real VAL","Recon. desde VAL","Muestra libre"],
               records,"Diagnóstico de reconstrucción",scale=1)


if __name__=="__main__":
    main()
