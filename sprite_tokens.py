"""VQGAN condicional de sprites y distribución autorregresiva conjunta de las dos vistas."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from prepare import TYPES

GRID = 12
VOCAB = 257  # El índice cero corresponde a una tesela blanca de 8 × 8.
LENGTH = 2 * GRID * GRID


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.GroupNorm(8,channels),nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1),
                                  nn.GroupNorm(8,channels),nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1))

    def forward(self, x):
        return x + self.body(x)


class Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for a,b in [(3,24),(24,48),(48,64)]:
            layers.extend([nn.Conv2d(a,b,4,2,1),nn.GroupNorm(8,b),nn.SiLU()])
        self.encoder=nn.Sequential(*layers,Residual(64),nn.Conv2d(64,32,1))
        self.register_buffer("codes",F.normalize(torch.randn(VOCAB,32),dim=1))
        self.register_buffer("counts",torch.ones(VOCAB))
        self.register_buffer("sums",self.codes.clone())
        self.register_buffer("initialized",torch.tensor(False))
        self.types=nn.Linear(len(TYPES),32,bias=False)
        layers=[nn.Conv2d(32,64,3,padding=1),Residual(64)]
        for a,b in [(64,48),(48,32),(32,16)]:
            layers.extend([nn.Conv2d(a,4*b,3,padding=1),nn.PixelShuffle(2),
                           nn.GroupNorm(8,b),nn.SiLU()])
        self.decoder=nn.Sequential(*layers,nn.Conv2d(16,4,3,padding=1))

    def quantize(self, pairs, update=False):
        views=pairs.reshape(-1,3,96,96)
        active=F.max_pool2d(1-views.amin(1,keepdim=True),8).flatten()>0.03
        h=F.normalize(self.encoder(views),dim=1).permute(0,2,3,1).reshape(-1,32)
        values=h.detach()[active]
        with torch.no_grad():
            if update and not self.initialized:
                initial=values[torch.randperm(len(values),device=h.device)[:VOCAB-1]]
                if len(initial)!=VOCAB-1:
                    raise ValueError("Lote insuficiente para inicializar el diccionario")
                self.codes[1:].copy_(initial)
                self.codes[0].zero_()
                self.sums.copy_(self.codes)
                self.initialized.fill_(True)
            ids=torch.zeros(len(h),device=h.device,dtype=torch.long)
            ids[active]=1+(values@self.codes[1:].T).argmax(1)
            quant=self.codes[ids]
            if update:
                counts=torch.bincount(ids[active],minlength=VOCAB).to(h.dtype)
                sums=torch.zeros_like(self.codes).index_add_(0,ids[active],values)
                self.counts.lerp_(counts,0.025)
                self.sums.lerp_(sums,0.025)
                stale=self.counts[1:]<0.15
                if stale.any():
                    dead=stale.nonzero().flatten()+1
                    self.sums[dead]=values[torch.randint(len(values),(len(dead),),device=h.device)]
                    self.counts[dead]=1
                self.codes[1:].copy_(F.normalize(self.sums[1:],dim=1))
        commit=F.mse_loss(h[active],quant[active]) if active.any() else h.sum()*0
        straight=h+(quant-h).detach()
        return ids.reshape(-1,2,GRID,GRID),straight.reshape(-1,GRID,GRID,32).permute(0,3,1,2),commit

    def decode(self, ids, y, straight=None):
        if straight is None:
            straight=self.codes[ids.reshape(-1,GRID,GRID)].permute(0,3,1,2)
        raw=self.decoder(straight+self.types(y).repeat_interleave(2,dim=0)[:,:,None,None])
        mask=F.interpolate((ids.reshape(-1,1,GRID,GRID)!=0).float(),scale_factor=8)
        alpha=raw[:,3:4].sigmoid()*mask
        views=1+alpha*(raw[:,:3].tanh()-1)
        return views.reshape(-1,6,96,96),raw[:,3:4]


class AttentionBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.before=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width)
        self.project=nn.Linear(width,width)
        self.mlp=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,3*width),nn.GELU(),nn.Linear(3*width,width))

    def forward(self, x, cache=None):
        b,n,d=x.shape
        q,k,v=self.qkv(self.before(x)).reshape(b,n,3,4,d//4).permute(2,0,3,1,4)
        if cache is not None:
            if cache:
                k=torch.cat([cache[0],k],dim=2)
                v=torch.cat([cache[1],v],dim=2)
            cache[:]=[k,v]
        att=F.scaled_dot_product_attention(q,k,v,is_causal=cache is None)
        x=x+self.project(att.transpose(1,2).reshape(b,n,d))
        return x+self.mlp(x)


class Prior(nn.Module):
    def __init__(self):
        super().__init__()
        width=128
        self.tokens=nn.Embedding(VOCAB+1,width)
        self.position=nn.Parameter(torch.randn(1,LENGTH,width)*0.025)
        self.view=nn.Embedding(2,width)
        self.labels=nn.Linear(len(TYPES),width,bias=False)
        self.layers=nn.ModuleList(AttentionBlock(width) for _ in range(4))
        self.out=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,VOCAB))

    def forward(self, prev, y, offset=0, caches=None, layout=None):
        pos=torch.arange(offset,offset+prev.size(1),device=prev.device)
        h=self.tokens(prev)+self.position[:,pos]+self.view(pos%2)+self.labels(y)[:,None]
        if layout is not None:
            h=h+layout[:,pos]
        for i,layer in enumerate(self.layers):
            h=layer(h,None if caches is None else caches[i])
        return self.out(h)


def sequence(ids):
    return ids.flatten(2).transpose(1,2).reshape(-1,LENGTH)


class TokenGenerator(nn.Module):
    def __init__(self, temperature=0.9):
        super().__init__()
        self.tokenizer=Tokenizer()
        self.prior=Prior()
        self.temperature=temperature

    @torch.inference_mode()
    def sample_ids(self, z, y, layout=None, blank_bias=None):
        # Un único z determina todos los sorteos, alternando frente y espalda.
        uniforms=[]
        for row in z.detach().cpu().contiguous().numpy():
            seed=int.from_bytes(hashlib.sha256(row.tobytes()).digest()[:8],"little")
            uniforms.append(torch.rand(LENGTH,generator=torch.Generator().manual_seed(seed)))
        draws=torch.stack(uniforms).to(z.device)
        prev=torch.full((len(z),1),VOCAB,device=z.device,dtype=torch.long)
        caches=[[] for _ in self.prior.layers]
        out=[]
        for i in range(LENGTH):
            logits=self.prior(prev,y,i,caches,layout)[:,0]
            if blank_bias is not None:
                logits[:,0]+=blank_bias[:,i]
            prob=(logits/self.temperature).softmax(-1)
            prev=(prob.cumsum(-1)<draws[:,i:i+1]).sum(1,keepdim=True).clamp_max(VOCAB-1)
            out.append(prev)
        return torch.cat(out,1).reshape(-1,GRID*GRID,2).transpose(1,2).reshape(-1,2,GRID,GRID)

    def forward(self,z,y):
        return self.tokenizer.decode(self.sample_ids(z,y),y)[0]
