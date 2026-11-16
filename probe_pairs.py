"""Mide si D prefiere la espalda verdadera a otra del mismo tipo en VAL."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from model import Discriminator
from prepare import PREPARED, condition, read_rows
from train import batch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (PREPARED / "splits.csv").open(newline="") as stream:
        val_ids = {r["pair_id"] for r in csv.DictReader(stream) if r["split"] == "val"}
    groups = defaultdict(list)
    for row in read_rows():
        if row["pair_id"] in val_ids:
            groups[tuple(sorted(condition(row)))].append(row)
    pairs, controls = [], []
    rng = random.Random(310)
    for rows in groups.values():
        for real in rows:
            peers = [r for r in rows if r["species_id"] != real["species_id"]
                     and r["back_path"] != real["back_path"]]
            if peers:
                pairs.append(real)
                controls.append(dict(real, back_path=rng.choice(peers)["back_path"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Discriminator().to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["discriminator"])
    margins = []
    for idx in range(0, len(pairs), 24):
        chunk = range(idx, min(idx + 24, len(pairs)))
        real, y = batch(pairs, chunk, device)
        swapped, _ = batch(controls, chunk, device)
        margins.extend((model(real, y) - model(swapped, y)).cpu().tolist())
    out = dict(step=state["step"], selection_seed=310, val_pairs=len(val_ids), eligible=len(pairs),
               conditions=len({tuple(sorted(condition(r))) for r in pairs}),
               real_preferred=sum(m > 0 for m in margins), ties=sum(m == 0 for m in margins),
               fraction_real_preferred=sum(m > 0 for m in margins) / len(margins),
               mean_score_difference=sum(margins) / len(margins))
    args.output.write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
