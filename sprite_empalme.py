"""Generate paired sprites from partial TRAIN drawings and an adversarial seam painter."""
import torch
from torch import nn
from torch.nn import functional as F

from prepare import TYPES

SIZE = 64
START, END = 21, 43
COLORS = 9
INK_GROUPS = 48


def outline_features(mask):
    """Relative horizontal position, width and coverage through the overlap."""
    x = torch.arange(SIZE,device=mask.device,dtype=torch.float32)
    mass = mask.sum(-1,dtype=torch.float32)
    centre = (mask*x).sum(-1)/mass.clamp_min(1)
    centre = torch.where(mass>0,centre,(SIZE-1)/2)
    return torch.cat([centre.mean(-1)/SIZE, mass.mean(-1)/SIZE,
                      F.adaptive_avg_pool2d(mask.float(),(3,8)).flatten(1)],1)


class SeamPainter(nn.Module):
    def __init__(self):
        super().__init__()
        self.start=nn.Conv2d(COLORS+1,32,3,padding=1)
        self.types=nn.Linear(len(TYPES),32)
        self.layers=nn.Sequential(nn.SiLU(),nn.Conv2d(32,40,3,padding=2,dilation=2),nn.SiLU(),
                                  nn.Conv2d(40,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,COLORS,1))

    def forward(self, indices, band, tags):
        original=F.one_hot(indices.long(),COLORS).permute(0,1,4,2,3).flatten(0,1).float()
        width=band.reshape(-1,1,SIZE,SIZE)
        features=self.start(torch.cat([original,width],1))+self.types(tags).repeat_interleave(2,0)[:,:,None,None]
        local=F.avg_pool2d(original,5,1,2)
        return (2.4*original+1.5*local+width*self.layers(features)).reshape(len(tags),2,COLORS,SIZE,SIZE)


class SpliceGenerator(nn.Module):
    def __init__(self, heads, bodies, joined=False):
        super().__init__()
        self.joined=joined
        self.finish_dorsal=False
        for role,count,height in (("head",heads,END),("body",bodies,SIZE-START)):
            self.register_buffer(role+"_tiles",torch.zeros(count,2,height,SIZE,dtype=torch.uint8))
            self.register_buffer(role+"_colors",torch.zeros(count,COLORS,3))
            self.register_buffer(role+"_types",torch.zeros(count,len(TYPES)))
            self.register_buffer(role+"_shape",torch.zeros(count,52))
        self.painter=SeamPainter()
        self.register_buffer("ink",torch.zeros(INK_GROUPS,COLORS,3))
        self.register_buffer("ink_weights",torch.ones(len(TYPES),len(TYPES),INK_GROUPS)/INK_GROUPS)

    @staticmethod
    def choose(weights,u):
        prob=weights/weights.sum(1,keepdim=True)
        return (prob.cumsum(1)<u[:,None]).sum(1).clamp_max(weights.shape[1]-1)

    def select(self,z,tags):
        axis=torch.arange(len(TYPES),device=z.device)
        first=torch.where(tags>0,axis,len(TYPES)).amin(1)
        second=torch.where(tags>0,axis,0).amax(1)
        switch=z[:,0]>0
        a,b=torch.where(switch,first,second),torch.where(switch,second,first)
        # Both halves come from disjoint species banks. A dual condition can
        # place either type in the upper drawing and the other in the lower one.
        u=torch.special.ndtr(z)
        def eligible(bank,main,other):
            single=(main==other)[:,None]
            bonus=torch.where(single,1.+1.5*(bank.sum(1)==1),1.+3.*bank[:,other].T)
            return bank[:,main].T*bonus
        weights=eligible(self.head_types,a,b)
        h=self.choose(weights,u[:,1])
        distance=(self.head_shape[h,None,4:]-self.body_shape[None,:,4:]).square().mean(-1)
        width=(self.head_shape[h,None,2:4]-self.body_shape[None,:,2:4]).square().mean(-1)
        weights=eligible(self.body_types,b,a)*(-10*distance-9*width).exp()
        # Rank several type-compatible lower bodies by how much solid tissue
        # actually meets in the two views after alignment. Use the weaker view
        # as well as their average, so a good front cannot hide a split back.
        draws=u[:,[2,11,12,13,14,15,16,17]]
        cdf=(weights/weights.sum(1,keepdim=True)).cumsum(1)
        options=torch.searchsorted(cdf.contiguous(),draws.contiguous()).clamp_max(weights.size(1)-1)
        count=options.size(1)
        upper,lower=self.register_parts(z.repeat_interleave(count,0),h.repeat_interleave(count),options.flatten())
        top=upper[:,:,27:37]>0
        bottom=lower[:,:,27:37]>0
        shared=(top&bottom).sum((2,3))
        coverage=2*shared/(top.sum((2,3))+bottom.sum((2,3))).clamp_min(1)
        continuity=coverage.mean(1)+.6*coverage.amin(1)
        winner=continuity.reshape(len(z),count).argmax(1,keepdim=True)
        return h,options.gather(1,winner)[:,0]

    @staticmethod
    def warp(ids,z,centres,xscale,yscale,shift):
        n=len(ids)
        axis=torch.arange(SIZE,dtype=z.dtype,device=z.device)
        yy=axis[None,None,:,None]-31.5
        xx=axis[None,None,None,:]-centres[:,:,None,None]
        bend=.06*z[:,7:8,None,None].tanh()*yy
        src_x=(xx-shift[:,:,None,None]-bend)/xscale[:,None,None,None]+centres[:,:,None,None]
        src_y=yy/yscale[:,None,None,None]+31.5+torch.zeros_like(src_x)
        grid=torch.stack((2*src_x/(SIZE-1)-1,2*src_y/(SIZE-1)-1),-1)
        return F.grid_sample(ids.flatten(0,1)[:,None].float(),grid.reshape(-1,SIZE,SIZE,2),
                             mode="nearest",align_corners=True).reshape(n,2,SIZE,SIZE).to(torch.long)

    def register_parts(self,z,h,b):
        upper=F.pad(self.head_tiles[h],(0,0,0,SIZE-END))
        lower=F.pad(self.body_tiles[b],(0,0,START,0))
        head_c,body_c=self.head_shape[h,:2]*SIZE,self.body_shape[b,:2]*SIZE
        width_ratio=((self.head_shape[h,2:4].mean(1)+.05)/(self.body_shape[b,2:4].mean(1)+.05)).clamp(.72,1.4)
        upper=self.warp(upper,z,head_c,1+.12*z[:,3].tanh(),1+.11*z[:,4].tanh(),torch.zeros_like(head_c))
        lower=self.warp(lower,z,body_c,width_ratio.sqrt()*(1+.09*z[:,5].tanh()),1+.13*z[:,6].tanh(),
                        (head_c-body_c).clamp(-9,9))
        return upper,lower

    @staticmethod
    def align_torso(upper,lower):
        """Gently fit the lower torso to the upper trunk, fading out before the feet."""
        axis=torch.arange(SIZE,device=lower.device,dtype=torch.float32)
        # Find a vertical shift for each view before matching the trunk widths.
        # Restrict it to the middle third so a good pair keeps its head and feet.
        offsets=axis.new_tensor([0,-2,2,-4,4,-6,6])
        fade=(1-(axis-31).abs()/17).clamp_min(0)
        source=(axis[None]+offsets[:,None]*fade).round().long().clamp(0,SIZE-1)
        top=upper[:,:,27:37]>0
        candidates=lower[:,:,source[:,27:37],:]>0
        shared=(top[:,:,None]&candidates).sum((-2,-1))
        total=top.sum((-2,-1))[:,:,None]+candidates.sum((-2,-1))
        overlap=2*shared/total.clamp_min(1)
        score=overlap-.014*offsets.abs()
        choice=score.argmax(-1)
        lower=lower.gather(2,source[choice][...,None].expand_as(lower))
        def span(tile):
            mask=(tile>0).float()
            mass=mask.sum(-1,keepdim=True)
            cumulative=mask.cumsum(-1)
            left=(cumulative<.17*mass).sum(-1)
            right=(cumulative<.83*mass).sum(-1)
            return (left+right)*.5,(right-left).float(),mass.squeeze(-1)
        top,upper_width,upper_mass=span(upper)
        base,lower_width,lower_mass=span(lower)
        valid=(upper_mass>4)&(lower_mass>4)&(axis>=25)&(axis<=37)
        def smooth(v):
            return F.avg_pool1d((v*valid).float(),9,1,4)/F.avg_pool1d(valid.float(),9,1,4).clamp_min(1/9)
        # Clamp stretches so that a wing crossing the trunk cannot swallow a limb.
        shift=smooth((top-base).clamp(-7,7))
        stretch=smooth(((upper_width+2)/(lower_width+2)).clamp(.78,1.28)-1)
        envelope=(1-(axis-31).abs()/14).clamp_min(0)
        x=(axis-base[...,None]-shift[...,None]*envelope[:,None])/(1+stretch[...,None]*envelope[:,None])+base[...,None]
        aligned=lower.gather(-1,x.round().long().clamp(0,SIZE-1))
        return aligned.masked_fill((x<0)|(x>SIZE-1),0)

    def draft(self,z,tags,with_back_source=False):
        h,b=self.select(z,tags)
        upper,lower=self.register_parts(z,h,b)
        if self.joined:
            lower=self.align_torso(upper,lower)
        # Find the least visible left-to-right cut through the two drawings.
        start,end=25,39
        top,base=upper[:,:,start:end],lower[:,:,start:end]
        yy=torch.arange(start,end,device=z.device)[None,None,:,None]
        mid=(31.5+2.6*z[:,8].tanh())[:,None,None,None]
        energy=2.2*((top==0)^(base==0)).float()+.065*(top-base).abs()+.075*(yy-mid).abs()
        if self.joined:
            # Penalize cuts through incompatible markings and silhouette edges.
            difference=((top-base).abs().float()/8+2*((top>0)^(base>0)).float())
            neighbourhood=F.avg_pool2d(difference,(5,3),1,(2,1),count_include_pad=False)
            energy=energy+1.3*neighbourhood
        dp=energy[:,:,:,0]
        history=[]
        for x in range(1,SIZE):
            padded=F.pad(dp,(1,1),value=1e4)
            options=torch.stack((padded[:,:,:-2]+.16,dp,padded[:,:,2:]+.16),-1)
            best,direction=options.min(-1)
            dp=energy[:,:,:,x]+best
            history.append(direction-1)
        pos=dp.argmin(-1,keepdim=True)
        path=[pos]
        for direction in reversed(history):
            pos=pos+direction.gather(2,pos)
            path.append(pos)
        seam=torch.cat(path[::-1],-1)[:,:,None,:]+start
        distance=torch.arange(SIZE,device=z.device)[None,None,:,None]-seam
        indices=torch.where(distance<=0,upper,lower)
        if self.joined:
            # Match the local midtones that actually meet without recolouring the
            # whole lower body. Keep the darkest outlines and highlights intact.
            upper_tone=upper.gather(2,seam)
            lower_tone=lower.gather(2,seam+1)
            valid=(upper_tone>1)&(lower_tone>1)&(upper_tone<8)&(lower_tone<8)
            weight=F.avg_pool2d(valid.float(),(1,7),1,(0,3))
            difference=F.avg_pool2d(((upper_tone-lower_tone)*valid).float(),(1,7),1,(0,3))
            correction=(difference/weight.clamp_min(.15)).clamp(-1.7,1.7)
            correction=(correction*(1-(distance-1).abs()/6).clamp_min(0)).round().long()
            adjust=(distance>0)&(indices>1)&(indices<8)&(weight>.25)
            indices=torch.where(adjust,(indices+correction).clamp(2,7),indices)
        band=(1-distance.abs()/6).clamp_min(0)
        # Use one palette for the two views. Both sources use intensity-ordered
        # colour indices, so highlights and dark ink keep their respective roles.
        mix=(.48+.32*z[:,9].tanh())[:,None,None]
        inherited=torch.lerp(self.head_colors[h],self.body_colors[b],mix)
        axis=torch.arange(len(TYPES),device=z.device)
        i=torch.where(tags>0,axis,len(TYPES)).amin(1)
        j=torch.where(tags>0,axis,0).amax(1)
        weights=self.ink_weights[i,j]
        distance=(self.ink[None,:,2:8]-inherited[:,None,2:8]).square().mean((2,3))
        ink_id=self.choose(weights*(-2.5*distance).exp(),torch.special.ndtr(z[:,20]))
        colors=torch.lerp(inherited,self.ink[ink_id],.70)
        if with_back_source:
            return indices,band,colors,lower[:,1]
        return indices,band,colors

    @staticmethod
    def colorize(indices,colors):
        return colors[torch.arange(len(indices),device=indices.device)[:,None,None,None],indices].permute(0,1,4,2,3).flatten(1,2)

    def forward(self,z,tags):
        draft=self.draft(z,tags,with_back_source=self.finish_dorsal)
        ids,band,colors=draft[:3]
        ids=self.painter(ids,band,tags).argmax(2)
        if self.finish_dorsal:
            from sprite_dorso import finish_back
            seams=band[:,1].argmax(1).detach().cpu().numpy()
            backs=ids[:,1].detach().cpu().numpy()
            sources=draft[3].detach().cpu().numpy()
            ids[:,1]=torch.stack([torch.from_numpy(finish_back(back,seam,source))
                                  for back,seam,source in zip(backs,seams,sources)]).to(ids.device)
        return F.pad(self.colorize(ids,colors),(16,16,16,16),value=1)
