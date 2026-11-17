"""Compare corrected backs and the short hierarchical GAN continuation."""
from pathlib import Path

import torch
from diagnose_pairs import rgb_pairs
from evaluar_diversa import CASES
from evaluar_perfiles import SPANISH, labels, draw_comparison
from evaluar_tipos_duales import reference
from gallery_cycles import NAMES, draw_board
from prepare import TYPES, read_rows
from sample_cycles import load_generator
import json


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    out=Path("estructura/dorso")
    out.mkdir(exist_ok=True,parents=True)
    model=load_generator("pesos/empalme_union.pth","cpu")
    rows=[]
    names=dict(zip(TYPES,SPANISH))
    cases=[("bug+poison",41001),("electric",41004),("fighting",41004),("ghost",43002),("fire",43002)]
    for case,s in cases:
        NAMES[case]=" / ".join(names[t] for t in case.split("+"))
        z=torch.randn(1,96,generator=torch.Generator().manual_seed(s))
        y=torch.tensor([labels(case)])
        model.finish_dorsal=False
        before=model(z,y)
        model.finish_dorsal=True
        after=model(z,y)
        assert torch.equal(before[:,:3],after[:,:3])
        rows.append((case,s,[rgb_pairs(before)[0],rgb_pairs(after)[0]]))
    draw_board(out/"espaldas.png",["Empalme · 3.600","Empalme · dorso"],rows,
               "Las mismas criaturas, de frente y de espaldas",scale=2)

    models=[load_generator(path,"cpu") for path in ("pesos/jerarquica_nueva.pth","pesos/jerarquica_20000.pth")]
    references=json.loads(Path("estructura/bench_general/protocolo.json").read_text(encoding="utf8"))["referencias"]
    data={r["pair_id"]:r for r in read_rows()}
    rows=[]
    for case in CASES:
        NAMES[case]=" / ".join(names[t] for t in case.split("+"))
        seed=42003
        z=torch.randn(1,96,generator=torch.Generator().manual_seed(seed))
        y=torch.tensor([labels(case)])
        real=reference(data[references[case][0][0]])
        rows.append((case,seed,[real]+[rgb_pairs(m(z,y))[0] for m in models]))
    draw_comparison(["Real · mismo tipo","GAN · 18.000","GAN · 20.000"],rows,out/"gan.png",
                    "GAN jerárquica · misma semilla en ocho tipos")


if __name__=="__main__":
    main()
