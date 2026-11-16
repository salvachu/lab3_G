"""Difusión adversarial de pares alineados. Un único z determina la trayectoria completa."""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from prepare import TYPES


def representation(rgb):
    # N,2,3,H,W en [-1,1]. Separar tinta y ocupación ayuda a aprender líneas interiores.
    fg=(rgb.amin(2,keepdim=True)<0.94).float()
    lum=(rgb*rgb.new_tensor([.299,.587,.114])[None,None,:,None,None]).sum(2,keepdim=True)
    ink=(0.5*(1-lum)*fg).sqrt()
    return torch.cat([rgb,2*fg-1,2*ink-1],2).flatten(1,2)


class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__()
        self.a=nn.Sequential(nn.GroupNorm(8,a),nn.SiLU(),nn.Conv2d(a,b,3,padding=1))
        self.mod=nn.Linear(160,2*b)
        self.norm=nn.GroupNorm(8,b)
        self.b=nn.Conv2d(b,b,3,padding=1)
        self.skip=nn.Identity() if a==b else nn.Conv2d(a,b,1)

    def forward(self,x,c):
        h=self.a(x)
        scale,bias=self.mod(c)[:,:,None,None].chunk(2,1)
        return self.skip(x)+self.b(F.silu(self.norm(h)*(1+scale)+bias))


class SpatialAttention(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.norm=nn.GroupNorm(8,d)
        self.qkv=nn.Conv2d(d,3*d,1)
        self.out=nn.Conv2d(d,d,1)

    def forward(self,x):
        n,d,h,w=x.shape
        q,k,v=self.qkv(self.norm(x)).reshape(n,3,4,d//4,h*w).permute(1,0,2,4,3)
        a=F.scaled_dot_product_attention(q,k,v)
        return x+self.out(a.transpose(2,3).reshape(n,d,h,w))


class PairNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("freq",torch.exp(torch.linspace(0,math.log(1000),32)))
        self.cond=nn.Sequential(nn.Linear(64+len(TYPES),160),nn.SiLU(),nn.Linear(160,160))
        self.start=nn.Conv2d(40,40,3,padding=1)
        self.high=Block(40,40)
        self.down1=nn.Conv2d(40,72,4,2,1)
        self.mid=Block(72,72)
        self.down2=nn.Conv2d(72,112,4,2,1)
        self.low=Block(112,112)
        self.attn=SpatialAttention(112)
        self.bottom=Block(112,112)
        self.up1=Block(112+72,72)
        self.mid_attn=SpatialAttention(72)
        self.up2=Block(72+40,40)
        self.out=nn.Sequential(nn.GroupNorm(8,40),nn.SiLU(),nn.Conv2d(40,40,3,padding=1),nn.PixelShuffle(2))

    def forward(self,x,t,y):
        phase=t[:,None]*self.freq
        c=self.cond(torch.cat([phase.sin(),phase.cos(),y],1))
        a=self.high(self.start(F.pixel_unshuffle(x,2)),c)
        b=self.mid(self.down1(a),c)
        d=self.bottom(self.attn(self.low(self.down2(b),c)),c)
        d=self.mid_attn(self.up1(torch.cat([F.interpolate(d,scale_factor=2),b],1),c))
        return self.out(self.up2(torch.cat([F.interpolate(d,scale_factor=2),a],1),c))


def rgb_from(x):
    f=x.reshape(-1,2,5,64,64)
    return torch.where(f[:,:,3:4]>0,f[:,:,:3].clamp(-1,1),1)


class JointGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=PairNet()
        self.guidance=1.6
        self.iterations=48
        self.correct_clip=False

    @torch.inference_mode()
    def forward(self,z,y):
        starts=[]
        for row in z.detach().cpu().contiguous().numpy():
            key=int.from_bytes(hashlib.sha256(b"paired-field"+row.tobytes()).digest()[:8],"little")
            starts.append(torch.randn(10,64,64,generator=torch.Generator().manual_seed(key)))
        x=torch.stack(starts).to(z.device)
        for i in range(self.iterations,0,-1):
            t=i/(self.iterations+0.01)
            angle=t*math.pi/2
            after=(i-1)/(self.iterations+0.01)*math.pi/2
            v0,v1=self.net(torch.cat([x,x]),z.new_full((2*len(z),),t),torch.cat([0*y,y])).chunk(2)
            v=v0+self.guidance*(v1-v0)
            clean=(math.cos(angle)*x-math.sin(angle)*v).clamp(-1,1)
            if self.correct_clip:
                noise=(x-math.cos(angle)*clean)/math.sin(angle)
            else:
                noise=math.sin(angle)*x+math.cos(angle)*v
            x=math.cos(after)*clean+math.sin(after)*noise
        return F.pad(rgb_from(x).flatten(1,2),(16,16,16,16),value=1)
