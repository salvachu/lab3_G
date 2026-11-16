"""Rellena las dos máscaras con tonos de una misma paleta y trazo aprendido."""
import torch
from torch import nn
from torch.nn import functional as F

from sprite_contours import ContourGenerator,layer


TONES=6


def luminance(rgb):
    return (rgb*rgb.new_tensor([0.299,0.587,0.114])[None,None,:,None,None]).sum(2)


def roles(rgb):
    """RGB [0,1], dos vistas juntas. La etiqueta TONES representa el fondo."""
    active=rgb.amin(2)<0.96
    luma=luminance(rgb)
    bins=(luma*255).round().long().clamp(0,255).flatten(1)
    hist=rgb.new_zeros(len(rgb),256).scatter_add_(1,bins,active.flatten(1).float())
    cum=hist.cumsum(1)
    fractions=rgb.new_tensor([0.13,0.31,0.50,0.70,0.88])
    thresholds=torch.searchsorted(cum,(cum[:,-1:]*fractions).contiguous())
    idx=(bins[:,:,None]>thresholds[:,None,:]).sum(-1)
    labels=torch.where(active.flatten(1),idx,TONES)
    counts=rgb.new_zeros(len(rgb),TONES+1).scatter_add_(1,labels,torch.ones_like(labels,dtype=rgb.dtype))
    pixels=rgb.transpose(1,2).reshape(len(rgb),3,-1)
    colors=rgb.new_zeros(len(rgb),3,TONES+1).scatter_add_(2,labels[:,None].expand_as(pixels),pixels)
    colors=colors.transpose(1,2)/counts.clamp_min(1)[:,:,None]
    # Si dos cuantiles coinciden se toma el tono vecino disponible.
    axis=torch.arange(TONES,device=rgb.device)
    nearest=((axis[:,None]-axis[None,:]).abs()[None]+100*(counts[:,:TONES]==0)[:,None]).argmin(-1)
    palette=colors.gather(1,nearest[:,:,None].expand(-1,-1,3))
    return labels.reshape_as(luma),palette


class InkHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.full=layer(6,24)
        self.down48=nn.Sequential(layer(24,32,2),layer(32,32))
        self.quarter=nn.Sequential(layer(32,48,2),layer(48,48))
        self.center=nn.Sequential(layer(48,64,2),layer(64,64),layer(64,64))
        self.up24=layer(64+48,48)
        self.up48=layer(48+32,32)
        self.up96=nn.Sequential(layer(32+24,32),layer(32,32),nn.Conv2d(32,2*TONES,1))

    def forward(self,mask,shade):
        inputs=torch.cat([mask,F.avg_pool2d(mask,5,1,2),F.interpolate(shade,size=(96,96),mode="bilinear",align_corners=False)],1)
        a=self.full(inputs)
        b=self.down48(a)
        c=self.quarter(b)
        x=self.center(c)
        for skip,layer_ in ((c,self.up24),(b,self.up48),(a,self.up96)):
            x=layer_(torch.cat([F.interpolate(x,scale_factor=2,mode="nearest"),skip],1))
        return x.reshape(len(mask),2,TONES,96,96)


class InkGenerator(ContourGenerator):
    def __init__(self,groups,temperature=0.9):
        super().__init__(groups,temperature)
        self.ink=InkHead()
        self.rank_tones=False

    @torch.inference_mode()
    def forward(self,z,y):
        pair,mask=self.render_components(z,y)
        return self.paint(pair,mask)

    @torch.inference_mode()
    def paint(self,pair,mask):
        rgb=(pair+1)/2
        _,palette=roles(rgb)
        # La tesela VQ promedia colores. Se oscurece su tono más bajo para el trazo.
        palette[:,0]*=0.55
        shade=F.avg_pool2d(luminance(rgb),4)
        logits=self.ink(mask,shade)
        assigned=logits.argmax(2)
        if self.rank_tones:
            # Los objetivos de TRAIN se definen por cuantiles de la pareja. Recuperar
            # esas proporciones evita pintar toda una cara con la categoría modal.
            levels=torch.arange(TONES,device=pair.device)
            score=(logits.softmax(2)*levels[None,None,:,None,None]).sum(2)
            cuts=pair.new_tensor([0.13,0.31,0.50,0.70,0.88])
            for i in range(len(pair)):
                fg=mask[i]>0.5
                if fg.any():
                    thresholds=torch.quantile(score[i][fg],cuts)
                    assigned[i]=torch.bucketize(score[i].contiguous(),thresholds)
        colored=palette[torch.arange(len(pair),device=pair.device)[:,None,None,None],assigned].permute(0,1,4,2,3)
        return torch.where(mask[:,:,None]>0.5,2*colored-1,1).flatten(1,2)
