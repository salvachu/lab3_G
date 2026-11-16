"""Learn a local RGB correction on real sprites; share the weights across views."""
import torch
from torch import nn
from torch.nn import functional as F

from sprite_relieve import ReliefGenerator


class SpriteDetail(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(nn.Conv2d(4, 24, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(24, 24, 3, padding=1), nn.SiLU())
        self.context = nn.Sequential(nn.Conv2d(24, 32, 4, stride=2, padding=1), nn.SiLU(),
                                     nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.SiLU(),
                                     nn.Conv2d(32, 24, 3, padding=1), nn.SiLU())
        self.paint = nn.Sequential(nn.Conv2d(48, 24, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(24, 3, 1))
        nn.init.zeros_(self.paint[-1].weight)
        nn.init.zeros_(self.paint[-1].bias)

    def forward(self, pairs):
        views = pairs.reshape(-1, 3, 96, 96)
        mask = (views.amin(1, keepdim=True) < .97).float()
        local = self.local(torch.cat([views, mask], dim=1))
        context = F.interpolate(self.context(local), size=views.shape[-2:], mode="nearest")
        correction = self.paint(torch.cat([local, context], dim=1))
        return (views + mask * correction).clamp(-1, 1).reshape_as(pairs)


class DetailedGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ReliefGenerator(lineage=True, engraved=True, elemental=True,
                                    prisma=True, insignias=True, joined=True)
        self.base.illustrated = True
        self.detail = SpriteDetail()

    def forward(self, z, types):
        return self.detail(self.base(z, types))
