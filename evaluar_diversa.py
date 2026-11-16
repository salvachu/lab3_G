"""Compare within-type variation on consecutive seeds, independently of GPT references."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_pairs import rgb_pairs
from evaluar_perfiles import SPANISH, draw_comparison, labels
from evaluar_tipos_duales import reference, silhouette_distance
from gallery_cycles import NAMES, draw_board
from model import LATENT
from prepare import TYPES, read_rows
from sample_cycles import load_generator

CASES = ["water", "dragon", "fighting", "fire", "electric", "ghost", "bug+poison", "water+dragon"]
SEEDS = list(range(41001, 41005))
HOLDOUT = list(range(42001, 42009))
OUT = Path("estructura/diversa")


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True, parents=True)
    refs = json.loads(Path("estructura/bench_general/protocolo.json").read_text(encoding="utf8"))["referencias"]
    real = {r["pair_id"]:r for r in read_rows()}
    names = dict(zip(TYPES, SPANISH))
    conditions = list(TYPES) + ["+".join(c) for c in itertools.combinations(TYPES, 2)]
    for case in [*conditions,*CASES]:
        NAMES[case] = " / ".join(names[t] for t in case.split("+"))
    seeds = SEEDS + HOLDOUT + [318071]
    z = torch.stack([torch.randn(LATENT, generator=torch.Generator().manual_seed(s)) for s in seeds]).to(device)
    models = {n:load_generator(Path("pesos") / (f+".pth"), device) for n,f in
              [("Ficha","ficha"), ("Cuerpos","ficha_todos"), ("Ficha diversa","diversa")]}
    nets = models["Ficha diversa"]
    stats = {}
    grid = {}
    comparison = []
    for case in dict.fromkeys(CASES + list(TYPES) + ["ghost+psychic"]):
        y = z.new_tensor([labels(case)])
        grid[case] = {}
        for name, model in models.items():
            samples = rgb_pairs(model(z, y.expand(len(z), -1)))
            grid[case][name] = samples
            held = samples[len(SEEDS):-1]
            different = sum(silhouette_distance(a,b) for a,b in itertools.combinations(held, 2))/28
            stats.setdefault(case,{})[name] = {"distancia_silueta":round(different, 4),
                                             "pares_distintos":len({im.tobytes() for im in held})}
        if case in CASES:
            row = reference(real[refs[case][0][0]])
            comparison.append((case, seeds[-1], [row]+[grid[case][m][-1] for m in models]))
    headings = [f"Semilla {s}" for s in SEEDS]
    def records(cases):
        return [(case,None,grid[case]["Ficha diversa"][:len(SEEDS)]) for case in cases]
    draw_board(OUT/"semillas.png", headings, records(CASES), "Ficha diversa · Cuatro criaturas por tipo", scale=1)
    for i in range(3):
        draw_board(OUT/f"tipos_{i+1}.png", headings, records(TYPES[i*6:i*6+6]), "Ficha diversa · Tipos individuales", scale=1)
    draw_comparison(["Real · mismo tipo", "Ficha · 1.200", "Cuerpos · 2.000", "Ficha diversa · +2.600"],
                    comparison, OUT/"comparacion.png", "Misma semilla · Modelos del laboratorio y referencia GPT")
    case = "ghost+psychic"
    NAMES[case] = "Fantasma / Psíquico"
    indices = [-1, *range(len(SEEDS))]
    # Match the local web app's CPU, one request per creature.
    tags = torch.tensor([labels(case)])
    trazo = load_generator(Path("pesos/trazo.pth"), "cpu")
    diversa_cpu = load_generator(Path("pesos/diversa.pth"), "cpu")
    paired = [[rgb_pairs(net(z[i].cpu().unsqueeze(0), tags))[0] for i in indices]
              for net in (trazo.base, trazo, diversa_cpu)]
    draw_board(OUT/"fantasma_psiquico.png", ["Cuerpos · 2.000", "Trazo · +800", "Ficha diversa · +2.600"],
               [(case, seeds[i], list(p)) for i,p in zip(indices, zip(*paired))],
               "Fantasma / Psíquico · Tu ejemplar y cuatro semillas más", scale=1)
    for start in range(0, len(conditions), 4):
        tags = z.new_tensor([labels(c) for c in conditions[start:start+4]])
        result = nets(z[:1].expand(len(tags),-1), tags)
        mask = result.reshape(-1,2,3,96,96).amin(2)<.88
        assert torch.isfinite(result).all() and mask.flatten(2).any(2).all()
    for case in ("fire","ghost","bug+poison"):
        tags=z.new_tensor([labels(case)])
        result=nets(z[:1],tags)
        assert torch.equal(result,nets(z[:1],tags))
        pair=rgb_pairs(result)[0]
        assert np.mean(np.abs(np.asarray(pair,dtype=float)-np.asarray(grid[case]["Ficha diversa"][0],dtype=float))) < .05
    (OUT/"medidas.json").write_text(json.dumps(dict(semillas_visibles=SEEDS, semillas_medicion=HOLDOUT,
        condiciones_con_ambas_vistas=len(conditions), por_tipo=stats), indent=2, ensure_ascii=False),encoding="utf8")


if __name__ == "__main__":
    main()
