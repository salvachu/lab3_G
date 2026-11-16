"""Continuación recuperable de Plan. Cada bloque guarda pesos y AdamW juntos."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from prepare import DATA, PREPARED, ROOT
from sprite_plans import PlanGenerator, fit_plans
from sprite_tokens import VOCAB, sequence
from train import batch, train_rows


BASE = Path("runs/plan_piezas_5000/step_05000.pt")
STATUS = ROOT / "ESTADO_NOCHE.md"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path, content):
    temporary = path.with_suffix(".writing")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def status(name, checkpoint, command, decision, event=None):
    history = ROOT / "noche/historial.md"
    if event:
        with history.open("a", encoding="utf-8") as stream:
            stream.write(f"- {event}\n")
    past = history.read_text(encoding="utf-8") if history.exists() else ""
    atomic_text(STATUS, f"# Estado del Lab 3\n\nCorrida activa — {name}\n\n"
                f"Último checkpoint verificado — [{checkpoint}]({checkpoint.as_posix()})\n\n"
                f"SHA-256 — `{digest(checkpoint)}`\n\n"
                f"Comando desde la raíz de lab3\n\n```powershell\n{command}\n```\n\n"
                f"Siguiente decisión — {decision}\n\n## Historial de corridas\n\n{past}")


@torch.no_grad()
def training_examples(net, signature):
    cache = ROOT / "noche/ejemplos_plan.pt"
    base_hash = digest(BASE)
    if cache.exists():
        stored = torch.load(cache, weights_only=True)
        if stored["data"] == signature and stored["base"] == base_hash:
            return [stored[k].cuda() for k in ("tokens", "labels", "groups")]
    rows = train_rows()
    masks, tokens, labels = [], [], []
    for first in range(0, len(rows), 32):
        real, y = batch(rows, range(first, min(first+32, len(rows))), "cuda")
        ids, _, _ = net.tokenizer.quantize(real)
        masks.append(F.avg_pool2d((real.reshape(-1,2,3,96,96).amin(2)<0.92).float(),8))
        tokens.append(sequence(ids))
        labels.append(y)
    masks, tokens, labels = map(torch.cat, (masks, tokens, labels))
    # La primera operación aleatoria en CUDA del ajuste original fue fit_plans.
    torch.cuda.manual_seed(310)
    groups, means, occ, counts = fit_plans(masks, tokens)
    support = json.loads((BASE.parent / "config.json").read_text())["support"]
    if counts.tolist()!=support or not torch.equal(means,net.means) or not torch.equal(occ,net.occupancy):
        raise ValueError("La asignación de TRAIN no reproduce los planes originales")
    tmp = cache.with_suffix(".tmp")
    torch.save(dict(data=signature, base=base_hash, tokens=tokens.short().cpu(),
                    labels=labels.cpu(), groups=groups.cpu()), tmp)
    tmp.replace(cache)
    return tokens, labels, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--from-ck", type=Path, default=BASE)
    parser.add_argument("--until", required=True, type=int)
    args = parser.parse_args()
    out, end = args.run, args.until
    out.mkdir(parents=True, exist_ok=True)
    config = dict(start=args.from_ck.as_posix(), start_hash=digest(args.from_ck), end=end)
    cfg = out/"continuacion.json"
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError("La carpeta pertenece a otra corrida")
    atomic_text(cfg, json.dumps(config, indent=2))
    files = list(out.glob("step_*.pt"))
    ck = max(files, key=lambda f:int(f.stem.split("_")[-1])) if files else args.from_ck
    state = torch.load(ck, map_location="cpu", weights_only=True)
    signature = hashlib.sha256((PREPARED/"splits.csv").read_bytes()+(DATA/"pairs.csv").read_bytes()).hexdigest()
    if (state["architecture"], state["data"], state["batch"], state["seed"]) != ("sprite_plans",signature,24,310):
        raise ValueError("Arquitectura o muestreo incompatibles")
    torch.manual_seed(310)
    net = PlanGenerator(state["groups"], state["temperature"]).cuda()
    net.load_state_dict(state["generator"])
    net.tokenizer.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=0.0006,betas=(0.9,0.95),weight_decay=0.02)
    optimizer.load_state_dict(state["po"])
    start = state["step"]
    if start == end:
        print(f"Verificado: {ck}", flush=True)
        return
    if start > end:
        raise ValueError("El destino es anterior al checkpoint")
    command = f".\\.venv\\Scripts\\python noche_plan.py --run {out.as_posix()} --from-ck {args.from_ck.as_posix()} --until {end}"
    decision = "Comparar las mismas 32 parejas de desarrollo antes de seleccionar otro corte o una corrección."
    status(out.name, ck, command, decision, f"Inicio de {out.name} desde {ck.as_posix()} hasta {end:,}.")
    tokens, labels, groups = training_examples(net, signature)
    metrics = out/"metricas.csv"
    if metrics.exists():
        with metrics.open(newline="") as f:
            kept = [line for line in f if not line[0].isdigit() or int(line.split(",")[0])<=start]
        atomic_text(metrics,"".join(kept))
    with metrics.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if stream.tell()==0:
            writer.writerow(["paso","prior_nll","plan_nll"])
        for step in range(start+1, end+1):
            rng = np.random.default_rng([310,step])
            ix = torch.from_numpy(rng.integers(len(groups),size=24)).cuda()
            target = tokens[ix].long()
            layout,bias = net.context(groups[ix])
            logits = net.prior(F.pad(target[:,:-1],(1,0),value=VOCAB),labels[ix],layout=layout)
            logits[:,:,0] += bias
            nll = F.cross_entropy(logits.transpose(1,2),target)
            plan_nll = F.cross_entropy(net.plan_distribution(labels[ix]),groups[ix])
            optimizer.zero_grad(set_to_none=True)
            (nll+plan_nll).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True)
            optimizer.step()
            writer.writerow([step,nll.item(),plan_nll.item()])
            if step%500==0 or step==end:
                stream.flush()
                dest=out/f"step_{step:05}.pt"
                tmp=dest.with_suffix(".tmp")
                torch.save(dict(state, step=step, generator=net.state_dict(), po=optimizer.state_dict()),tmp)
                saved=torch.load(tmp,map_location="cpu",weights_only=True)
                if saved["step"]!=step or any(int(s["step"])!=step-state["codec_steps"] for s in saved["po"]["state"].values()):
                    raise ValueError("Corte incompleto")
                net.load_state_dict(saved["generator"])
                tmp.replace(dest)
                done=step==end
                status("ninguna" if done else out.name,dest,command,decision,
                       f"Cierre de {out.name} en {dest.as_posix()}." if done else None)
                print(f"Verificado {dest} | NLL {nll.item():.3f}", flush=True)


if __name__=="__main__":
    main()
