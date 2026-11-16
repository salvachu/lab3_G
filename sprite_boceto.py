"""Autoencoder de dos vistas con paleta compartida y prior condicional de rango bajo."""
import torch
from torch import nn
from torch.nn import functional as F

from prepare import TYPES

DIM = 64
COLORS = 8
GROUPS = 96
RANK = 20


class Block(nn.Module):
    def __init__(self, before, after):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(before, after, 3, padding=1),
                                  nn.GroupNorm(8, after), nn.SiLU(),
                                  nn.Conv2d(after, after, 3, padding=1))
        self.skip = nn.Conv2d(before, after, 1) if before != after else nn.Identity()

    def forward(self, x):
        return F.silu(self.skip(x) + self.body(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b in [(8,32),(32,48),(48,64),(64,96)]:
            layers.extend([nn.Conv2d(a,b,4,2,1), nn.GroupNorm(8,b), nn.SiLU()])
        self.layers = nn.Sequential(*layers, nn.Flatten(), nn.Linear(96*16,DIM), nn.LayerNorm(DIM))

    def forward(self, rgb, mask):
        return self.layers(torch.cat([rgb,mask[:,:,None]],2).flatten(1,2))


class SketchGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        cond = DIM + len(TYPES)
        self.project = nn.Linear(cond, 112*16)
        self.blocks = nn.ModuleList(Block(a,b) for a,b in [(112,96),(96,72),(72,48),(48,32)])
        self.bias = nn.ModuleList(nn.Linear(cond,b) for b in (96,72,48,32))
        self.maps = nn.ModuleList(nn.Conv2d(b,2*(COLORS+1),1) for b in (96,72,48,32))
        d = DIM + 3*COLORS
        self.register_buffer("centers",torch.zeros(GROUPS,d))
        self.register_buffer("basis",torch.zeros(GROUPS,RANK,d))
        self.register_buffer("scatter",torch.zeros(GROUPS,d))
        self.register_buffer("freq",torch.ones(len(TYPES)+1,GROUPS))

    def decode(self,u,y):
        c = torch.cat([u,y],1)
        h = self.project(c).reshape(-1,112,4,4)
        maps = []
        for block,bias,head in zip(self.blocks,self.bias,self.maps):
            h = block(F.interpolate(h,scale_factor=2,mode="nearest")) + bias(c)[:,:,None,None]
            m = head(h)
            if maps:
                m = m + F.interpolate(maps[-1],scale_factor=2,mode="bilinear",align_corners=False)
            maps.append(m)
        return [m.unflatten(1,(2,COLORS+1)) for m in maps]

    def sample_latent(self,z,y):
        logfreq = self.freq.clamp_min(1.e-6).log()
        logp = y @ logfreq[1:] / y.sum(1,keepdim=True).clamp_min(1)
        if y.shape[0] != z.shape[0]:
            raise ValueError("Cada pareja necesita sus tipos y un solo z")
        p = (1.45*logp).softmax(1)
        choice = (p.cumsum(1)<torch.special.ndtr(z[:,0:1])).sum(1).clamp_max(GROUPS-1)
        comp = (self.basis[choice] * z[:,1:RANK+1,None]).sum(1)
        # El ruido fino se deriva del mismo vector y no consulta el conjunto de datos.
        noise = torch.cat([z[:,RANK+1:],z[:,1:15]],1)[:,:self.centers.shape[1]]
        latent = self.centers[choice] + .87*comp + .43*self.scatter[choice]*noise
        u,pal = latent.split([DIM,3*COLORS],1)
        pal = pal.reshape(-1,COLORS,3).clamp(-1,1)
        pal = torch.cat([torch.ones_like(pal[:,:1]),pal],1)
        return u,pal

    def forward(self,z,y):
        u,pal=self.sample_latent(z,y)
        labels = self.decode(u,y)[-1].argmax(2)
        rgb = pal[torch.arange(len(y),device=y.device)[:,None,None,None],labels].permute(0,1,4,2,3)
        return F.pad(rgb.flatten(1,2),(16,16,16,16),value=1)
