"""Fit shared front/back pigment pairs from TRAIN, weighting species equally."""
import numpy as np
import torch
from prepare import TYPES


@torch.no_grad()
def fit_pigments(data):
    # The cached quantization was computed on the training pairs in train_boceto.
    saved = torch.load("runs/estructura_boceto/objetivos.pt", weights_only=True, mmap=True)
    assert saved["data"] == data["data"]
    palette = (saved["pal"].cuda() + 1) / 2
    codes = saved["labels"]
    counts = []
    for batch in codes.split(128):
        ids = batch.cuda().long().flatten(1)
        freq = torch.zeros(len(ids), 9, device=ids.device)
        freq.scatter_add_(1, ids, torch.ones_like(ids, dtype=freq.dtype))
        counts.append(freq[:, 1:])
    counts = torch.cat(counts)
    counts /= counts.sum(1, keepdim=True).clamp_min(1)
    chroma = palette.amax(2) - palette.amin(2)
    light = palette.mean(2)
    score = counts.sqrt() * (.12 + chroma) * (.15 + ((light > .14) & (light < .91)).float())
    index = torch.arange(len(palette), device=palette.device)
    main = palette[index,score.argmax(1)]
    contrast = (palette-main[:,None]).square().mean(2).sqrt()
    accent = palette[index,(contrast * (.1 + counts.sqrt()) * (.3 + chroma)).argmax(1)]
    colors = torch.cat([main,accent],1)
    species = torch.tensor(np.unique(data["species"],return_inverse=True)[1],device=colors.device)
    weight = 1 / torch.bincount(species)[species].float()
    gen = torch.Generator(device=colors.device).manual_seed(7923)
    centers = colors[torch.multinomial(weight,24,generator=gen)].clone()
    for _ in range(32):
        ids = torch.cdist(colors,centers).argmin(1)
        mass = colors.new_zeros(24).scatter_add_(0,ids,weight)
        centers = torch.where(mass[:,None] > 0, torch.zeros_like(centers).index_add_(0,ids,colors*weight[:,None])/mass.clamp_min(.001)[:,None], centers)
    bins = torch.nn.functional.one_hot(torch.cdist(colors,centers).argmin(1),24).float()
    ys = data["y"].cuda()
    def histogram(w):
        return (bins*w[:,None]).sum(0)
    def norm(v):
        return v/v.sum()
    global_freq = norm(histogram(weight)+.05)
    single = [norm(histogram(weight*ys[:,i])+2*global_freq) for i in range(len(TYPES))]
    table = colors.new_empty(len(TYPES),len(TYPES),8,24)
    for i in range(len(TYPES)):
        for j in range(i+1):
            joint = histogram(weight*ys[:,i]*ys[:,j])
            backoff = norm((single[i]*single[j]/global_freq).clamp_min(.00001).sqrt())
            conditional = single[i] if i==j else norm(joint+2*backoff)
            distinctive = norm(conditional.pow(1.3)/global_freq.pow(.40))
            table[i,j]=table[j,i]=distinctive.expand(8,-1)
    return centers.reshape(24,2,3), table
