import torch
from torch import nn

from model import Generator
from prepare import TYPES


class TypeGenerator(Generator):
    """Modulación afin por tipo sobre las cuatro normalizaciones del generador."""
    def __init__(self):
        super().__init__()
        self.affines=nn.ModuleList(nn.Linear(len(TYPES),2*c,bias=False) for c in (96,64,32,24))
        for affine in self.affines:
            nn.init.zeros_(affine.weight)

    def forward(self,noise,types):
        h=self.project(torch.cat((noise,types),dim=1)).reshape(-1,128,6,6)
        for i,affine in enumerate(self.affines):
            start=4*i
            h=self.blocks[start](h)
            h=self.blocks[start+1](h)
            h=self.blocks[start+2](h)
            scale,offset=affine(types).unsqueeze(-1).unsqueeze(-1).chunk(2,dim=1)
            h=self.blocks[start+3](h*(1+scale)+offset)
        raw=self.out(h).reshape(-1,2,4,96,96)
        alpha=raw[:,:,3:4].sigmoid()
        return (1+alpha*(raw[:,:,:3].tanh()-1)).flatten(1,2)
