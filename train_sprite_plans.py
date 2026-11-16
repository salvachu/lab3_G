import csv
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from prepare import ROOT,DATA,PREPARED
from sprite_plans import PlanGenerator,fit_plans
from sprite_tokens import TokenGenerator,VOCAB,sequence
from train import batch,train_rows


def main():
    out=ROOT/"runs/plan_piezas_5000"
    out.mkdir()
    torch.manual_seed(310)
    base=ROOT/"runs/piezas_5000/step_03000.pt"
    state=torch.load(base,map_location="cpu",weights_only=True)
    data=hashlib.sha256((PREPARED/"splits.csv").read_bytes()+(DATA/"pairs.csv").read_bytes()).hexdigest()
    assert state["step"]==3000 and state["data"]==data
    common=TokenGenerator().cuda()
    common.load_state_dict(state["generator"])
    rows=train_rows()
    masks,tokens,labels=[],[],[]
    with torch.no_grad():
        for i in range(0,len(rows),32):
            x,y=batch(rows,range(i,min(i+32,len(rows))),"cuda")
            ids,_,_=common.tokenizer.quantize(x)
            views=x.reshape(-1,2,3,96,96)
            masks.append(F.avg_pool2d((views.amin(2)<0.92).float(),8))
            tokens.append(sequence(ids))
            labels.append(y)
    masks,tokens,labels=map(torch.cat,(masks,tokens,labels))
    groups,means,occupancy,counts=fit_plans(masks,tokens)
    net=PlanGenerator(len(means)).cuda()
    net.tokenizer.load_state_dict(common.tokenizer.state_dict())
    net.prior.load_state_dict(common.prior.state_dict())
    net.means.copy_(means)
    net.occupancy.copy_(occupancy)
    net.tokenizer.requires_grad_(False)
    po=torch.optim.AdamW([v for v in net.parameters() if v.requires_grad],lr=0.0006,betas=(0.9,0.95),weight_decay=0.02)
    cfg=dict(start=str(base.relative_to(ROOT)),start_hash=hashlib.sha256(base.read_bytes()).hexdigest(),
             data=data,start_step=3000,steps=5000,batch=24,seed=310,groups=len(means),support=counts.tolist(),
             species_per_group=[len({rows[i]["species_id"] for i in (groups==k).nonzero().flatten().tolist()})
                                for k in range(len(means))])
    (out/"config.json").write_text(json.dumps(cfg,indent=2))
    begin=time.monotonic()
    with (out/"metrics.csv").open("w",newline="") as stream:
        log=csv.writer(stream)
        log.writerow(["step","prior_nll","plan_nll","seconds","peak_gpu_MiB"])
        for step in range(3001,5001):
            rng=np.random.default_rng([310,step])
            ix=torch.from_numpy(rng.integers(len(rows),size=24)).cuda()
            target=tokens[ix]
            layout,bias=net.context(groups[ix])
            logits=net.prior(F.pad(target[:,:-1],(1,0),value=VOCAB),labels[ix],layout=layout)
            logits[:,:,0]+=bias
            nll=F.cross_entropy(logits.transpose(1,2),target)
            plan_nll=F.cross_entropy(net.plan_distribution(labels[ix]),groups[ix])
            po.zero_grad(set_to_none=True)
            (nll+plan_nll).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True)
            po.step()
            log.writerow([step,nll.item(),plan_nll.item(),round(time.monotonic()-begin,2),
                          round(torch.cuda.max_memory_allocated()/2**20,1)])
            if step%500==0:
                stream.flush()
                print(f"{step}: prior {nll.item():.3f}",flush=True)
    temp=out/"step_05000.tmp"
    torch.save(dict(architecture="sprite_plans",groups=len(means),temperature=net.temperature,step=5000,
                    codec_steps=3000,generator=net.state_dict(),po=po.state_dict(),data=data,seed=310,batch=24),temp)
    temp.replace(out/"step_05000.pt")


if __name__=="__main__":
    main()
