"""Type-conditioned hierarchical variational GAN with paired categorical rendering."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

from prepare import TYPES


class Residual(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.core=nn.Sequential(nn.GroupNorm(4,c),nn.SiLU(),nn.Conv2d(c,c,3,padding=1),
                                nn.GroupNorm(4,c),nn.SiLU(),nn.Conv2d(c,c,3,padding=1))

    def forward(self,x):
        return x+.45*self.core(x)


class GaussianLevel(nn.Module):
    def __init__(self,c,k):
        super().__init__()
        self.k=k
        self.prior=nn.Conv2d(c,2*k,3,padding=1)
        self.posterior=nn.Sequential(nn.Conv2d(c*2,c,1),nn.SiLU(),nn.Conv2d(c,2*k,3,padding=1))
        self.inject=nn.Conv2d(k,c,1)
        self.res=Residual(c)

    def forward(self,x,eps,e=None):
        mean,scale=self.prior(x).chunk(2,1)
        std=(1.5*(scale/1.5).tanh()).exp()
        cost=x.new_zeros(len(x))
        if e is not None:
            shift,ratio=self.posterior(torch.cat([x,e],1)).chunk(2,1)
            ratio=2.5*(ratio/2.5).tanh()
            cost=.5*(shift.square()+(2*ratio).exp()-1-2*ratio).flatten(1).sum(1)
            eps=shift+ratio.exp()*eps
        return self.res(x+self.inject(mean+std*eps)),cost,eps


class HierarchicalGenerator(nn.Module):
    widths=(72,64,48)
    sizes=(4,8,16)
    latent=(20,12,8)

    def __init__(self,components=0,prior_method="gaussian"):
        super().__init__()
        self.constant=nn.Parameter(torch.randn(1,72,4,4)*.2)
        self.type_features=nn.Sequential(nn.Linear(len(TYPES),96),nn.SiLU(),nn.Linear(96,96))
        self.bias=nn.ModuleList(nn.Linear(96,c) for c in self.widths)
        self.levels=nn.ModuleList(GaussianLevel(c,k) for c,k in zip(self.widths,self.latent))
        self.up=nn.ModuleList(nn.Conv2d(c,d,3,padding=1) for c,d in zip(self.widths,self.widths[1:]))
        self.palette=nn.Sequential(nn.Linear(72*4*4+96,128),nn.SiLU(),nn.Linear(128,8*3),nn.Tanh())
        self.detail=nn.Sequential(nn.Conv2d(48,32*4,3,padding=1),nn.PixelShuffle(2),Residual(32),
                                  nn.Conv2d(32,24*4,3,padding=1),nn.PixelShuffle(2),nn.SiLU(),nn.Conv2d(24,18,3,padding=1))
        self.components=components
        self.prior_method=prior_method
        if components:
            width=sum(k*s*s for k,s in zip(self.latent,self.sizes))
            self.register_buffer("centres",torch.zeros(components,width,dtype=torch.float16))
            self.register_buffer("directions",torch.zeros(components,5,width,dtype=torch.float16))
            self.register_buffer("frequencies",torch.ones(len(TYPES),len(TYPES),components)/components)

    def decode(self,tags,noise,features=None,encoded=None):
        cond=self.type_features(tags)
        x=self.constant.expand(len(tags),-1,-1,-1)
        costs=[]
        for i,level in enumerate(self.levels):
            if i:
                x=self.up[i-1](F.interpolate(x,scale_factor=2,mode="nearest"))
            x=x+self.bias[i](cond)[:,:,None,None]
            x,kl,eps=level(x,noise[i],None if features is None else features[i])
            if encoded is not None:
                encoded.append(eps.flatten(1))
            costs.append(kl)
            if i==0:
                colors=self.palette(torch.cat([x.flatten(1),cond],1)).reshape(-1,8,3)
        logits=self.detail(x).reshape(len(x),2,9,64,64)
        return logits,colors,costs

    @staticmethod
    def rgb(logits,colors,straight=False):
        # The first channel predicts occupancy separately from the ink indices.
        mask=logits[:,:,:1].sigmoid()
        soft=logits[:,:,1:].softmax(2)
        hard=F.one_hot(soft.argmax(2),8).permute(0,1,4,2,3).to(soft.dtype)
        alpha=(mask>.5).to(mask.dtype)
        if straight:
            hard=soft+(hard-soft).detach()
            alpha=mask+(alpha-mask).detach()
        rgb=torch.einsum("nvkhw,nkc->nvchw",hard,colors)
        return (1+alpha*(rgb-1)).flatten(1,2)

    def forward(self,z,tags):
        # Recover the entire noise pyramid from the single public latent draw.
        count=sum(k*s*s for k,s in zip(self.latent,self.sizes))
        noise=[]
        for row in z.detach().cpu().contiguous().numpy():
            seed=int.from_bytes(hashlib.sha256(row.tobytes()).digest()[:8],"little")
            noise.append(torch.randn(count,generator=torch.Generator().manual_seed(seed)))
        flat=torch.stack(noise).to(z)
        if self.components:
            axis=torch.arange(len(TYPES),device=z.device)
            i=torch.where(tags>0,axis,len(TYPES)).amin(1)
            j=torch.where(tags>0,axis,0).amax(1)
            weights=self.frequencies[i,j]
            group=(weights.cumsum(1)<torch.special.ndtr(z[:,:1])).sum(1).clamp_max(self.components-1)
            if self.prior_method=="convex":
                # Draw two local modes without replacement and interpolate.
                # Each stored direction is shrunk towards a six-species mean.
                order=z[:,1:6].argsort(1,descending=True)
                alpha=(.08+.24*torch.special.ndtr(z[:,6:7]))
                delta=torch.lerp(self.directions[group,order[:,0]].to(z),
                                 self.directions[group,order[:,1]].to(z),alpha)
                flat=self.centres[group].to(z)+delta+.11*flat
            else:
                delta=(self.directions[group].to(z)*z[:,1:6,None].clamp(-2.,2.)).sum(1)
                flat=self.centres[group].to(z)+.83*delta+.17*flat
        parts=flat.split([k*s*s for k,s in zip(self.latent,self.sizes)],1)
        temperatures=(1.,1.,1.) if self.components else (1.,.86,.72)
        pyramid=[v.reshape(-1,k,s,s)*temp for v,k,s,temp in zip(parts,self.latent,self.sizes,temperatures)]
        logits,colors,_=self.decode(tags,pyramid)
        return F.pad(self.rgb(logits,colors),(16,16,16,16),value=1)


class PosteriorEncoder(nn.Module):
    """Training-only network, omitted from inference exports."""
    def __init__(self):
        super().__init__()
        self.start=nn.Sequential(nn.PixelUnshuffle(2),nn.Conv2d(32,32,3,padding=1),Residual(32))
        self.down=nn.ModuleList(nn.Sequential(nn.Conv2d(a,b,4,2,1),Residual(b)) for a,b in
                                ((32,48),(48,64),(64,72)))

    def forward(self,pair):
        views=pair.reshape(-1,2,3,64,64)
        mask=(views.amin(2,keepdim=True)<.92).to(views.dtype)
        x=self.start(torch.cat([views,mask],2).flatten(1,2))
        feats=[]
        for down in self.down:
            x=down(x)
            feats.append(x)
        return feats[::-1]
