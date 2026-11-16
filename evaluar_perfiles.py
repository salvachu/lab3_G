"""Compare paired checkpoints on fixed conditions and six preselected seeds."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diagnose_pairs import rgb_pairs
from evaluar_tipos_duales import reference, silhouette_distance
from gallery_cycles import NAMES, draw_board
from model import LATENT
from prepare import TYPES, read_rows
from sample_cycles import load_generator
from train_sprite_latent import CASES

OUTPUT = Path("estructura/perfiles_pareados")
MODELS = {"Detalle · +1.200": "runs/detalle_1200/detalle_01200.pt",
          "Perfiles · 64 grupos": "runs/perfiles_pareados/modelo.pt",
          "Pigmentos · 24 paletas": "runs/perfiles_pigmentos/modelo.pt"}
SPANISH = "Normal Lucha Volador Veneno Tierra Roca Bicho Fantasma Acero Fuego Agua Planta Eléctrico Psíquico Hielo Dragón Siniestro Hada".split()


def labels(case):
    return [float(t in case.split("+")) for t in TYPES]


def pixel_reference(cells):
    """Match the 96px canvases and use a single foreground palette for both views."""
    boxes=[]
    for cell in cells:
        rgb=np.asarray(cell.convert("RGB"))
        yy,xx=np.where(rgb.min(2)<204)
        boxes.append((xx.min(),yy.min(),xx.max()+1,yy.max()+1))
    scale=64/max(max(r-l,b-t) for l,t,r,b in boxes)
    pair=Image.new("RGB",(192,96),"white")
    for i,(cell,(l,t,r,b)) in enumerate(zip(cells,boxes)):
        size=max(1,round((r-l)*scale)),max(1,round((b-t)*scale))
        pixels=cell.convert("RGB").crop((l,t,r,b)).resize(size,Image.Resampling.BOX)
        pair.paste(pixels,(96*i+(96-size[0])//2,(96-size[1])//2))
    arr=np.array(pair)
    figure=arr.min(2)<231
    swatches=Image.fromarray(arr[figure][None]).quantize(15,method=Image.Quantize.MEDIANCUT)
    quantized=np.asarray(pair.quantize(palette=swatches,dither=Image.Dither.NONE).convert("RGB"))
    arr[figure]=quantized[figure]
    arr[~figure]=255
    return Image.fromarray(arr)


def draw_comparison(headings, records, output=OUTPUT / "comparacion.png", title="Ocho tipos · Modelos del laboratorio y GPT"):
    atlas_path = OUTPUT.parent / "referencias_visuales/ocho_tipos_imagegen.png"
    atlas_order = ["water", "dragon", "fighting", "fire", "electric", "ghost", "bug+poison", "water+dragon"]
    rows = [0, 309, 623, 933, 1254]
    augmented = []
    with Image.open(atlas_path) as atlas:
        for case, seed, sprites in records:
            r, pair_index = divmod(atlas_order.index(case), 2)
            cols = [0, 306, 600, 943, 1254] if r == 3 else [0, 314, 627, 943, 1254]
            cells=[]
            for view in range(2):
                c = pair_index * 2 + view
                cell = atlas.crop((round(cols[c] * atlas.width / 1254), round(rows[r] * atlas.height / 1254),
                                   round(cols[c+1] * atlas.width / 1254), round(rows[r+1] * atlas.height / 1254)))
                cells.append(cell)
            augmented.append((case, seed, [*sprites, pixel_reference(cells)]))
    draw_board(output, [*headings, "GPT · 96 px"], augmented, title, scale=1)


@torch.inference_mode()
def main():
    OUTPUT.mkdir(exist_ok=True, parents=True)
    history = json.loads(Path("estructura/bench_general/protocolo.json").read_text(encoding="utf8"))
    seeds = history["semillas"] + [318071, 318089, 318101]
    cases = list(history["referencias"])
    real = {r["pair_id"]: r for r in read_rows()}
    names = dict(zip(TYPES, SPANISH))
    NAMES.update({c: " / ".join(names[t] for t in c.split("+")) for c in cases})
    z = torch.stack([torch.randn(LATENT, generator=torch.Generator().manual_seed(s)) for s in seeds]).cuda()
    grids = {}
    metrics = {}
    for name, checkpoint in MODELS.items():
        model = load_generator(Path(checkpoint), "cuda")
        samples = {}
        for case in cases:
            y = z.new_tensor(labels(case)).expand(len(z), -1)
            samples[case] = rgb_pairs(model(z, y))
        grids[name] = samples
        distances = [silhouette_distance(row[i], row[j]) for row in samples.values()
                     for i, j in itertools.combinations(range(len(z)), 2)]
        metrics[name] = {"distancia_media_siluetas_entre_semillas": float(np.mean(distances))}
    # Check both views for all 171 supported conditions, including unseen type pairs.
    conditions = TYPES + ["+".join(pair) for pair in itertools.combinations(TYPES, 2)]
    for start in range(0, len(conditions), 4):
        y = z.new_tensor([labels(c) for c in conditions[start:start+4]])
        outputs = model(z[:1].expand(len(y), -1), y)
        assert torch.isfinite(outputs).all()
        assert (outputs.reshape(-1, 2, 3, 96, 96).amin(2) < .88).flatten(2).any(2).all()
    repeated = rgb_pairs(model(z[:1], z.new_tensor([labels(CASES[0])])))[0]
    assert np.array_equal(np.asarray(repeated), np.asarray(grids[name][CASES[0]][0]))
    headings = ["Real · mismo tipo", *MODELS]

    def row(case, i):
        refs = history["referencias"][case]
        rid = refs[i % len(refs)][0]
        return (case, seeds[i], [reference(real[rid])] + [grid[case][i] for grid in grids.values()])

    for case in cases:
        draw_board(OUTPUT / f"{case}.png", headings, [row(case, i) for i in range(len(z))], NAMES[case])
    records = [row(case, 3) for case in CASES]
    draw_comparison(headings, records)
    for k in range(0, len(cases), 8):
        draw_board(OUTPUT / f"tipos_{k//8+1}.png", headings, [row(c, 3) for c in cases[k:k+8]],
                   "Comparación por tipo", scale=1)
    black_shapes = [(c, s, [Image.fromarray(np.where(np.asarray(im).min(2) < 238, 38, 255).astype("uint8")).convert("RGB")
                             for im in images]) for c, s, images in records]
    draw_board(OUTPUT / "siluetas.png", headings, black_shapes, "Siluetas de frente y espalda", scale=1)
    protocol = dict(semillas=seeds, modelos=MODELS, referencias=history["referencias"],
                    condiciones_con_dos_vistas_verificadas=len(conditions), metricas=metrics)
    (OUTPUT / "protocolo.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
