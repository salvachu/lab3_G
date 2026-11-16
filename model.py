import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from prepare import TYPES

LATENT = 96


class Generator(nn.Module):
    """Una muestra latente produce dos sprites en un único paso."""
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(LATENT + len(TYPES), 128 * 6 * 6)
        blocks = []
        for before, after in [(128, 96), (96, 64), (64, 32), (32, 24)]:
            blocks.extend([nn.Upsample(scale_factor=2, mode="nearest"),
                           nn.Conv2d(before, after, 3, padding=1, bias=False),
                           nn.BatchNorm2d(after), nn.ReLU()])
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv2d(24, 8, 3, padding=1)

    def forward(self, noise, types):
        h = self.project(torch.cat((noise, types), dim=1)).reshape(-1, 128, 6, 6)
        raw = self.out(self.blocks(h)).reshape(-1, 2, 4, 96, 96)
        # Fondo blanco presente en los datos. La opacidad se aprende sin etiquetas adicionales.
        alpha = raw[:, :, 3:4].sigmoid()
        return (1 + alpha * (raw[:, :, :3].tanh() - 1)).flatten(1, 2)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for before, after in [(6, 32), (32, 64), (64, 96), (96, 128)]:
            blocks.extend([spectral_norm(nn.Conv2d(before, after, 4, stride=2, padding=1)),
                           nn.LeakyReLU(0.2)])
        self.blocks = nn.Sequential(*blocks)
        self.score = spectral_norm(nn.Linear(128, 1))
        self.condition = spectral_norm(nn.Linear(len(TYPES), 128, bias=False))

    def forward(self, pairs, types):
        features = self.blocks(pairs).sum(dim=(2, 3)) / 6
        embedding = self.condition(nn.functional.normalize(types, dim=1))
        return self.score(features).flatten() + (features * embedding).sum(1)
