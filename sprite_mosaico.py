"""Conditional paired latent patch model with an adversarial RGB decoder."""
import torch
from torch import nn
from torch.nn import functional as F

from prepare import TYPES
from sprite_latent import SpriteCodec

GROUPS = 192
PRINTS = 384
ORIGINS = ((0, 0), (0, 3), (3, 0), (3, 3))
FEATURES = 24 * 4 * 4


class MosaicGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.codec = SpriteCodec()
        self.register_buffer("layouts", torch.zeros(GROUPS, FEATURES))
        self.register_buffer("basis", torch.zeros(GROUPS, 5, FEATURES))
        self.register_buffer("patches", torch.zeros(4, PRINTS, 24, 5, 5))
        self.register_buffer("local_layout", torch.zeros(4, PRINTS, FEATURES))
        self.register_buffer("frequency", torch.ones(len(TYPES), len(TYPES), GROUPS) / GROUPS)
        self.register_buffer("given_layout", torch.ones(4, GROUPS, PRINTS) / PRINTS)
        self.register_buffer("mean", torch.zeros(1, 24, 1, 1))
        self.register_buffer("scale", torch.ones(1, 24, 1, 1))
        self.register_buffer("palettes", torch.zeros(GROUPS, 8, 3))

    @staticmethod
    def descriptor(code):
        return F.avg_pool2d(code, 2).flatten(1)

    def compose(self, layout, group, uniforms=None, target=None):
        n = len(layout)
        result = layout.new_zeros(n, 24, 8, 8)
        counts = layout.new_zeros(1, 1, 8, 8)
        for q, (y, x) in enumerate(ORIGINS):
            region = result[:, :, y:y+5, x:x+5]
            weights = counts[:, :, y:y+5, x:x+5]
            bank = self.patches[q]
            if target is not None:
                score = -torch.cdist(target[:, :, y:y+5, x:x+5].flatten(1), bank.flatten(1)).square()
                pick = score.argmax(1)
            else:
                shape_error = torch.cdist(layout, self.local_layout[q]).square() / FEATURES
                score = self.given_layout[q, group].clamp_min(.00001).log() - 3.0*shape_error
                if q:
                    mask = (weights > 0).float()
                    current = region / weights.clamp_min(.001)
                    overlap = torch.cdist((current*mask).flatten(1), (bank*mask).flatten(1)).square()
                    score -= 2.0 * overlap / (24*mask.sum())
                pick = (score - (-uniforms[:,q].clamp(1e-7, 1-1e-7).log()).log()).argmax(1)
            feather = layout.new_tensor([.24, .76, 1., .76, .24])
            wy = feather.clone()
            wx = feather.clone()
            wy[:2] = 1 if y == 0 else wy[:2]
            wy[-2:] = 1 if y else wy[-2:]
            wx[:2] = 1 if x == 0 else wx[:2]
            wx[-2:] = 1 if x else wx[-2:]
            weight = wy[:, None] * wx[None, :]
            region.add_(bank[pick] * weight)
            weights.add_(weight)
        return result / counts

    def sample_codes(self, z, tags):
        axis = torch.arange(len(TYPES), device=z.device)
        i = torch.where(tags>0,axis,len(TYPES)).amin(1)
        j = torch.where(tags>0,axis,0).amax(1)
        u = torch.special.ndtr(z[:, :1])
        group = (self.frequency[i,j].cumsum(1)<u).sum(1).clamp_max(GROUPS-1)
        layout = self.layouts[group]+.65*(self.basis[group]*z[:,1:6,None].clamp(-2,2)).sum(1)
        # Derive the Gumbel draws for all four regions from the same input latent.
        generators = [torch.Generator().manual_seed(int(v)) for v in
                      ((torch.special.ndtr(z[:, 6].double().cpu()) * (2**52-1)).long())]
        uniforms = torch.stack([torch.rand((4, PRINTS),generator=g) for g in generators]).to(z)
        field = self.compose(layout, group, uniforms)
        return (field*self.scale+self.mean).clamp(-1,1), group

    def forward(self, z, tags):
        code, group = self.sample_codes(z,tags)
        pair, logits = self.codec.decode(code, hard=True)
        rgb = pair.reshape(len(z), 2, 3, 64, 64)
        colors = self.palettes[group]
        pixels = rgb.permute(0,1,3,4,2).reshape(len(z),-1,3)
        ids = torch.cdist(pixels, colors).argmin(2)
        snapped = colors.gather(1, ids[:,:,None].expand(-1,-1,3)).reshape(len(z),2,64,64,3).permute(0,1,4,2,3)
        rgb = torch.where(logits[:,:,None]>0, torch.lerp(rgb,snapped,.72), 1.)
        return F.pad(rgb.flatten(1,2), (16,16,16,16), value=1)
