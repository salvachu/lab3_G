import csv
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from prepare import ROOT
from sprite_patches import K,PatchGenerator,tiles,tile_dist
from train import batch,train_rows


@torch.no_grad()
def learn_table(rows):
    candidates=[]
    idx=np.random.default_rng(310).choice(len(rows),2048,replace=False)
    for i in range(0,len(idx),32):
        x,_=batch(rows,idx[i:i+32],"cuda")
        local=tiles(x).permute(0,2,3,1).reshape(-1,48)
        local=local[local.amin(1)<0.96]
        candidates.append(local[torch.randperm(len(local),device="cuda")[:1000]])
    data=torch.cat(candidates)
    centers=data[torch.randperm(len(data),device="cuda")[:K]].clone()
    centers[0]=1
    for _ in range(16):
        sums=torch.zeros_like(centers)
        mass=torch.zeros(K,device="cuda")
        for part in data.split(4096):
            ids=torch.cdist(part,centers).argmin(1)
            sums.index_add_(0,ids,part)
            mass+=torch.bincount(ids,minlength=K)
        centers=torch.where(mass[:,None]>0,sums/mass[:,None].clamp_min(1),centers)
        centers[0]=1
    # Representante de 4x4 más cercano a cada media: conserva las líneas de un píxel.
    lowest=torch.full((K,),float("inf"),device="cuda")
    table=centers.clone()
    for part in data.split(4096):
        dist,idx=torch.cdist(part,centers).min(0)
        table=torch.where((dist<lowest)[:,None],part[idx],table)
        lowest=torch.minimum(lowest,dist)
    table[0]=1
    return table


def main():
    out=ROOT/"runs/trazos_6000"
    out.mkdir()
    torch.manual_seed(310)
    base=ROOT/"runs/plan_piezas_5000/step_05000.pt"
    s=torch.load(base,map_location="cpu",weights_only=True)
    net=PatchGenerator(s["groups"],s["temperature"]).cuda()
    missing,extra=net.load_state_dict(s["generator"],strict=False)
    if extra or any(not n.startswith(("table","order","context_embedding.","patch_net.")) for n in missing):
        raise ValueError("Checkpoint de Plan incompatible")
    net.requires_grad_(False)
    net.context_embedding.requires_grad_(True)
    net.patch_net.requires_grad_(True)
    rows=train_rows()
    net.table.copy_(learn_table(rows))
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=0.0005)
    (out/"config.json").write_text(json.dumps(dict(start=str(base.relative_to(ROOT)),
        start_hash=hashlib.sha256(base.read_bytes()).hexdigest(),steps=6000,start_step=5000,seed=310,batch=24,
        table_rows=2048,patch_side=4,n_codes=K,data=s["data"],top_k=8,detail_temperature=0.75),indent=2))
    begin=time.monotonic()
    with (out/"metrics.csv").open("w",newline="") as f:
        log=csv.writer(f)
        log.writerow(["step","patch_nll","seconds","peak_gpu_MiB"])
        for step in range(5001,6001):
            rng=np.random.default_rng([310,step])
            real,y=batch(rows,rng.integers(len(rows),size=24),"cuda")
            with torch.no_grad():
                ids,_,_=net.tokenizer.quantize(real)
                coarse=tiles(net.tokenizer.decode(ids,y)[0])
                source=tiles(real)
                active=F.interpolate((ids.reshape(-1,1,12,12)!=0).float(),scale_factor=2)[:,0].bool()
                target=tile_dist(source,net.table).argmin(1)
                order=torch.randint(4,(len(active),1,1),device="cuda")
                known=torch.where(net.order<order,target,K)
                known[~active]=0
                loc=active&(net.order==order)
            pred=net.predict(coarse,known).permute(0,2,3,1)[loc]
            loss=F.cross_entropy(pred,target[loc])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True)
            opt.step()
            log.writerow([step,loss.item(),round(time.monotonic()-begin,2),round(torch.cuda.max_memory_allocated()/2**20,1)])
            if step%250==0:
                f.flush()
                print(f"{step}: NLL {loss.item():.3f}",flush=True)
    tmp=out/"step_06000.tmp"
    torch.save(dict(architecture="sprite_patches",groups=s["groups"],temperature=s["temperature"],step=6000,
                    codec_steps=3000,generator=net.state_dict(),refiner_optimizer=opt.state_dict(),seed=310,
                    batch=24,data=s["data"]),tmp)
    tmp.replace(out/"step_06000.pt")


if __name__=="__main__":
    main()
