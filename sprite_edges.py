"""Difusión condicional de la máscara nativa de las dos vistas; el RGB procede de Tinta."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from sprite_contours import layer
from sprite_ink import InkGenerator


class EdgeDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.start=layer(6,24)
        self.down48=layer(24,32,2)
        self.down24=layer(32,48,2)
        self.down12=nn.Sequential(layer(48,64,2),layer(64,64))
        self.time=nn.Sequential(nn.Linear(16,96),nn.SiLU(),nn.Linear(96,64))
        self.up24=layer(112,48)
        self.up48=layer(80,32)
        self.up96=nn.Sequential(layer(56,32),layer(32,24),nn.Conv2d(24,2,3,padding=1))
        self.register_buffer("frequencies",2*math.pi*2**torch.arange(8).float())

    def forward(self,noisy,t,coarse,evidence):
        a=self.start(torch.cat([noisy,coarse,evidence],1))
        b=self.down48(a)
        c=self.down24(b)
        phase=t[:,None]*self.frequencies
        time=self.time(torch.cat([phase.sin(),phase.cos()],1))
        x=self.down12(c)+time[:,:,None,None]
        for skip,block in ((c,self.up24),(b,self.up48),(a,self.up96)):
            x=block(torch.cat([F.interpolate(x,scale_factor=2,mode="nearest"),skip],1))
        return x


class EdgeGenerator(InkGenerator):
    def __init__(self,groups,temperature=0.9):
        super().__init__(groups,temperature)
        self.edge=EdgeDenoiser()

    @torch.inference_mode()
    def fine_mask(self,pair,coarse,z):
        evidence=(1-pair.amin(2))/2
        fields=[]
        for row in z.detach().cpu().contiguous().numpy():
            seed=int.from_bytes(hashlib.sha256(b"fine-shape"+row.tobytes()).digest()[:8],"little")
            fields.append(torch.randn(2,96,96,generator=torch.Generator().manual_seed(seed)))
        x=torch.stack(fields).to(z.device)
        steps=24
        for i in range(steps,0,-1):
            t=i/(steps+0.01)
            theta=t*math.pi/2
            next_theta=(i-1)/(steps+0.01)*math.pi/2
            v=self.edge(x,z.new_full((len(z),),t),coarse,evidence)
            clean=(math.cos(theta)*x-math.sin(theta)*v).clamp(-1,1)
            noise=math.sin(theta)*x+math.cos(theta)*v
            x=math.cos(next_theta)*clean+math.sin(next_theta)*noise
        return ((x+1)/2).clamp(0,1)

    def forward(self,z,y):
        pair,coarse=self.render_components(z,y)
        return self.paint(pair,self.fine_mask(pair,coarse,z))
