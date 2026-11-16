"""Control supervisado de capacidad. Nunca usar estos pesos para muestreo libre."""
import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from compare_checkpoints import font
from diagnose_pairs import rgb_pairs
from model import Generator, LATENT
from prepare import ROOT
from train import batch, train_rows

OUT = ROOT / "anatomia/diagnostico"
SPECIES = ["charizard", "pikachu", "psyduck", "scyther", "farfetchd", "lapras", "dodrio", "ponyta"]


def main():
    torch.manual_seed(310)
    rows = train_rows()
    selected = []
    for species in SPECIES:
        candidates = [r for r in rows if r["species_name"] == species and r["is_shiny"] == "0"]
        selected.append(min(candidates, key=lambda r: (r["source_generation"] != "5", r["pair_id"])))
    OUT.mkdir(exist_ok=False)
    (OUT/"pares.json").write_text(json.dumps(selected, indent=2))
    real, y = batch(selected, range(8), "cuda")
    z = torch.randn(8,LATENT,generator=torch.Generator().manual_seed(6201)).cuda()
    g = Generator().cuda()
    opt = torch.optim.Adam(g.parameters(), lr=0.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2400, eta_min=0.00008)
    views = real.reshape(-1, 3, 96, 96)
    fg = (views.amin(1,keepdim=True) < 0.96).float()
    # Les damos peso a los trazos interiores y exteriores de uno o dos píxeles.
    edge = (torch.nn.functional.max_pool2d(views,3,1,1) + torch.nn.functional.max_pool2d(-views,3,1,1))
    weights = 1 + 3*fg + 3*edge.amax(1,keepdim=True)
    with (OUT/"error.csv").open("w",newline="") as f:
        log=csv.writer(f)
        log.writerow(["paso","error_ponderado","MAE_primer_plano_RGB01"])
        for step in range(1,2401):
            pred = g(z,y).reshape_as(views)
            error = (pred - views).abs()
            loss = (weights*error).mean()/weights.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            schedule.step()
            if step%200==0:
                log.writerow([step,loss.item(),(error*fg).sum().item()/(2*3*fg.sum().item())])
    g.eval()
    with torch.inference_mode():
        predicted=g(z,y)
    torch.save(dict(generator=g.state_dict(),z=z.cpu(),step=2400,diagnostic_only=True),OUT/"copia_02400.pt")
    sheet=Image.new("RGB",(570,8*103+62),"white")
    draw=ImageDraw.Draw(sheet)
    draw.text((8,4),"Copia supervisada · ocho parejas TRAIN",font=font(19,True),fill="#213647")
    for x,label in [(140,"Objetivo real"),(353,"Copia · 2.400")]:
        draw.text((x,31),label,font=font(14,True),fill="#213647")
    for i,(a,b) in enumerate(zip(rgb_pairs(real),rgb_pairs(predicted))):
        top=58+i*103
        draw.text((8,top+40),SPECIES[i].capitalize(),font=font(15),fill="#213647")
        sheet.paste(a,(140,top))
        sheet.paste(b,(353,top))
    sheet.save(OUT/"objetivo_y_copia.png")


if __name__=="__main__":
    main()
