"""Joint latent generator of two silhouettes and their pigment fields."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from prepare import TYPES


class SpriteCodec(nn.Module):
    channels = 12

    def __init__(self):
        super().__init__()
        self.encode_detail = nn.Sequential(
            nn.PixelUnshuffle(4), nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(), nn.Conv2d(64, 8, 1), nn.Tanh())
        self.decode_detail = nn.Sequential(
            nn.Conv2d(12, 80, 3, padding=1), nn.SiLU(),
            nn.Conv2d(80, 4*48, 3, padding=1), nn.PixelShuffle(2), nn.SiLU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.SiLU(),
            nn.Conv2d(48, 4*32, 3, padding=1), nn.PixelShuffle(2), nn.SiLU(),
            nn.Conv2d(32, 4*16, 3, padding=1), nn.PixelShuffle(2), nn.SiLU(),
            nn.Conv2d(16, 4, 3, padding=1))

    def encode(self, pairs):
        x = pairs.reshape(-1, 3, 64, 64)
        mask = (x.amin(1, keepdim=True) < .93).to(x.dtype)
        outline = F.pixel_unshuffle(F.avg_pool2d(mask, 4), 2)
        detail = self.encode_detail(torch.cat([x, 2*mask-1], 1))
        return torch.cat([2*outline-1, detail], 1).reshape(-1, 24, 8, 8)

    def decode(self, code, hard=False):
        fields = code.reshape(-1, self.channels, 8, 8)
        coarse = F.pixel_shuffle(fields[:, :4], 2)
        raw = self.decode_detail(fields)
        logits = raw[:, 3:] + 3 * F.interpolate(coarse, size=(64,64), mode="bilinear", align_corners=False)
        mask = (logits > 0).to(raw.dtype) if hard else logits.sigmoid()
        rgb = 1 + mask * (raw[:, :3].tanh()-1)
        return rgb.reshape(-1, 6, 64, 64), logits.reshape(-1, 2, 64, 64)


class ConditionBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.GroupNorm(8, width)
        self.mod = nn.Linear(width, 2*width)
        self.conv = nn.Conv2d(width, width, 3, dilation=dilation, padding=dilation)
        self.mix = nn.Conv2d(width, width, 1)

    def forward(self, x, cond):
        scale, bias = self.mod(cond).chunk(2, 1)
        h = self.norm(x) * (1+scale[:,:,None,None]) + bias[:,:,None,None]
        return x + self.mix(F.silu(self.conv(F.silu(h))))


class LatentGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.codec = SpriteCodec()
        width = 96
        self.start = nn.Conv2d(24, width, 3, padding=1)
        self.position = nn.Parameter(torch.zeros(1, width, 8, 8))
        self.register_buffer("frequencies", torch.linspace(1, 28, 16))
        self.condition = nn.Sequential(nn.Linear(32+len(TYPES), width), nn.SiLU(), nn.Linear(width,width))
        self.blocks = nn.ModuleList(ConditionBlock(width, d) for d in (1,2,1,3,1,2))
        self.end = nn.Sequential(nn.GroupNorm(8,width), nn.SiLU(), nn.Conv2d(width,24,3,padding=1))
        self.register_buffer("mean", torch.zeros(1,24,1,1))
        self.register_buffer("std", torch.ones(1,24,1,1))

    def velocity(self, noisy, t, labels):
        phase = t[:,None]*self.frequencies
        cond = self.condition(torch.cat([phase.sin(),phase.cos(),2*labels], 1))
        x = self.start(noisy) + self.position
        for block in self.blocks:
            x = block(x,cond)
        return self.end(x)

    @torch.inference_mode()
    def forward(self, z, labels):
        noise = []
        for row in z.detach().cpu().contiguous().numpy():
            seed = int.from_bytes(hashlib.sha256(row.tobytes()).digest()[:8],"little")
            noise.append(torch.randn(24,8,8,generator=torch.Generator().manual_seed(seed)))
        x = torch.stack(noise).to(z)
        count = 32
        for i in range(count,0,-1):
            t = i/count
            theta, after = t*math.pi/2, (i-1)/count*math.pi/2
            v0,v1 = self.velocity(torch.cat([x,x]),x.new_full((2*len(x),),t),
                                  torch.cat([torch.zeros_like(labels),labels])).chunk(2)
            v = v0 + 1.65*(v1-v0)
            clean = math.cos(theta)*x - math.sin(theta)*v
            noise = math.sin(theta)*x + math.cos(theta)*v
            x = math.cos(after)*clean + math.sin(after)*noise
        rgb,_ = self.codec.decode((x*self.std+self.mean).clamp(-1,1), hard=True)
        return F.pad(rgb,(16,16,16,16),value=1)
