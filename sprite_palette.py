"""Cabeza de color categórica para refinar los mismos códigos del modelo Plan."""
import torch
from torch import nn
from torch.nn import functional as F

from sprite_plans import PlanGenerator
from sprite_tokens import GRID,Tokenizer


class PaletteTokenizer(Tokenizer):
    def __init__(self, colors):
        super().__init__()
        self.register_buffer("palette",torch.zeros(colors,3))
        self.decision="mode"
        self.detail=nn.Sequential(nn.Conv2d(16,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,colors,1))
        nn.init.zeros_(self.detail[-1].weight)
        nn.init.zeros_(self.detail[-1].bias)

    def palette_decode(self,ids,y,training=False):
        quant=self.codes[ids.reshape(-1,GRID,GRID)].permute(0,3,1,2)
        features=quant+self.types(y).repeat_interleave(2,dim=0)[:,:,None,None]
        for layer in list(self.decoder.children())[:-1]:
            features=layer(features)
        raw=self.decoder[-1](features)
        rgb=1+raw[:,3:4].sigmoid()*(raw[:,:3].tanh()-1)
        # Logits iniciales proporcionales a la distancia al color RGB continuo.
        colors=self.palette
        logits=F.conv2d(rgb,2*colors[:,:,None,None],-colors.square().sum(1))/0.20+self.detail(features)
        if training or self.decision=="mean":
            mean=F.conv2d(logits.softmax(1),colors.T[:,:,None,None])
        if self.decision=="mean":
            # El color de la paleta más cercano a E[RGB] minimiza el error cuadrático esperado.
            nearest=F.conv2d(mean,2*colors[:,:,None,None],-colors.square().sum(1)).argmax(1)
        else:
            nearest=logits.argmax(1)
        hard=colors[nearest].permute(0,3,1,2)
        if training:
            hard=mean+(hard-mean).detach()
        active=F.interpolate((ids.reshape(-1,1,GRID,GRID)!=0).float(),scale_factor=8)
        return (1+active*(hard-1)).reshape(-1,6,96,96),logits

    def decode(self,ids,y,straight=None):
        return self.palette_decode(ids,y)


class PaletteGenerator(PlanGenerator):
    def __init__(self,groups,colors,temperature=0.9):
        super().__init__(groups,temperature)
        self.tokenizer=PaletteTokenizer(colors)
