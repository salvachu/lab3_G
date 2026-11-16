"""Conditional sprite generation from measured proportions, silhouette and colour."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from prepare import TYPES
from sprite_profiles import ProfileGenerator

GROUPS = 64
FIELDS = ("ancho_sobre_alto", "ensanche_inferior", "ocupacion", "rojo", "verde", "azul")


def measure(pairs):
    """Describe both foregrounds, giving the two views equal weight for proportions."""
    rgb = pairs.reshape(len(pairs), 2, 3, *pairs.shape[-2:])
    mask = rgb.amin(2) < .95
    h, w = mask.shape[-2:]
    axis = torch.arange(h, device=rgb.device)
    rows, cols = mask.any(-1), mask.any(-2)
    top = torch.where(rows, axis, h).amin(-1)
    height = (torch.where(rows, axis, 0).amax(-1) - top + 1).clamp_min(1)
    width = (torch.where(cols, axis, 0).amax(-1) - torch.where(cols, axis, w).amin(-1) + 1).clamp_min(1)
    area = mask.sum((-1,-2)).clamp_min(1)
    bottom = (mask * (axis >= top[:,:,None]+height[:,:,None]*.5)[:,:,:,None]).sum((-1,-2))
    proportions = torch.stack([width/height, 2*bottom/area-1, area/(width*height)], -1).mean(1)
    pixels = ((rgb+1)/2).permute(0,1,3,4,2).reshape(len(pairs),-1,3)
    light = pixels.amax(-1)
    chroma = light-pixels.amin(-1)
    weights = mask.flatten(1)*(.025+chroma.square())*(.2+.8*light)
    bins = (pixels*3.999).long()
    index = bins[:,:,0]*16 + bins[:,:,1]*4 + bins[:,:,2]
    hist = weights.new_zeros(len(pairs),64).scatter_add_(1,index,weights)
    selected = weights * (index == hist.argmax(1,keepdim=True))
    color = (pixels*selected[:,:,None]).sum(1)/selected.sum(1,keepdim=True).clamp_min(.001)
    return torch.cat([proportions,color],1)


def ghost_pair(z, traits, detailed=False):
    """Pixel silhouettes of a floating mantle, a lantern and a wisp, seen from either side."""
    def field(a):
        return a.reshape(-1,1,1,1)
    axis = torch.linspace(-1,1,96,device=z.device)
    x, y = axis[None,None,None,:], axis[None,None,:,None]
    view = z.new_tensor([-1.,1.])[None,:,None,None]
    r, flare, fill = [field(a) for a in traits[:,:3].T]
    style = torch.special.ndtr(field(z[:,18]))
    if detailed:
        # Proportion changes cover the entire character, including outstretched
        # hands. Holding z fixes its family during an explicit card intervention.
        stretch=r.clamp(.57,1.58).pow(.75)
        x=x/stretch
        y=y*stretch.pow(.22)
    lantern = (style<.32).float()
    wisp = (style>.65).float()
    phase = field(z[:,24].tanh())
    side = view*field(torch.where(z[:,22]>0,1.,-1.))
    height=.83+.08*field(z[:,26].tanh())
    t=(y+.62)/height
    q=t.clamp(0,1)
    breadth=(.35+.24*(r-.65)).clamp(.28,.62)
    offset=side*(.06+.14*wisp)*(1-q).pow(2)
    u=(x-offset)/breadth
    dome=(1-(1-q).pow(2)).clamp_min(0).sqrt()
    width=dome*(.60+(.56+flare*.48)*q)
    width=width*(1-lantern)+lantern*(.46+.58*(q*math.pi).sin().clamp_min(0).pow(.65))
    width=width*(1-.22*wisp*q)
    folds=(u*8+phase+view*.6).cos()
    hem=1+.115*folds+.06*(1-fill)*u.abs()
    body=(u.abs()<width)&(t>=0)&(t<hem)
    # Curve joined to the underside; it tapers to a point.
    tail_t=((y-.10)/(.64+.08*phase)).clamp(0,1)
    curve=side*((.12+.12*wisp)*(6.4*tail_t+.2).sin()+.10*tail_t)
    tail_width=(.15+.11*wisp)*(1-tail_t).pow(.75)
    tail=(x-curve).abs()<tail_width
    tail=tail&(y>.09)&(tail_t<1)
    # A folded crest or handle is present at the same attachment point on both views.
    arc_x=(x-side*.12)/(.18+.04*phase)
    arc_y=(y+.60)/(.16+.035*phase)
    arc=arc_x.square()+arc_y.square()
    crest=(arc<1)&(arc>(.34+.28*lantern))&(y<-.54)
    shape=body|tail|crest
    front=view<0
    main=traits[:,None,3:,None,None].clamp(.16,.91)
    main=main+(1-main)*.10
    deep=main*.31+main.new_tensor([.035,.025,.062])[None,None,:,None,None]
    glow=(1-main)*.53+main.new_tensor([.37,.43,.46])[None,None,:,None,None]
    def paint(canvas, region, color):
        return torch.where(region[:,:,None],color,canvas)
    canvas=main.expand(-1,2,-1,96,96)
    canvas=paint(canvas,(folds>.25)&(t>.38)|((u>.33)&(t<.49)),main*.70)
    canvas=paint(canvas,(folds<-.45)&(t>.53),main*.81+.19)
    canvas=paint(canvas,tail,main*.68)
    canvas=paint(canvas,tail&((x-curve)*side<-.025),main*.74+.22)
    opening=((u+.13)/(.66-.13*lantern)).square()+((t-.52)/(.30+.04*wisp)).square()<1
    canvas=paint(canvas,front&opening,deep)
    for sign in (-1,1):
        ex=(u+.13-sign*.245)
        ey=t-.51-sign*ex*(.23+.25*phase)
        eye=(ex/(.105+.032*wisp)).square()+(ey/(.09+.04*lantern)).square()<1
        canvas=paint(canvas,front&eye&opening,glow)
    clasp=(u.abs()+((t-.34)*2.7).abs())<(.20+.06*lantern)
    canvas=paint(canvas,(~front)&clasp,main*.40+.55)
    if detailed:
        # A lantern has hinged arms and glass, a mask spirit has flame-ears
        # and a long scarf. Both projections share their proportions and colours.
        warm=glow*.52+main.new_tensor([.96,.71,.29])[None,None,:,None,None]*.48
        bone=main*.28+.67
        dark=deep*.77
        def ellipse(px,py,rx,ry):
            return ((x-px)/rx).square()+((y-py)/ry).square()<1
        lantern_mask=lantern.bool()
        wisp_mask=wisp.bool()
        sides=x.abs()
        handle=(((x/.125).square()+((y+.53)/.18).square()-1).abs()<.24)&(y<-.48)
        lid=(sides<(.075+((y+.53)/.16).clamp(0,1)*.27))&(y>-.53)&(y<-.37)
        room_w=.32+.04*phase
        bevel=.06*(x/.30).clamp(-1,1)*view
        room=(sides<room_w-.09*((y+.10)*3.7).abs())&(y>=-.37+bevel)&(y<.17+bevel)
        foot=(sides<(.15+.18*((.33-y)/.16).clamp(0,1)))&(y>=.17)&(y<.34)
        arm_y=.055-.65*(sides-.25).clamp_min(0)-.13*((sides-.38)*8).clamp(0,1).square()
        arms=(sides>.18)&(sides<.57)&((y-arm_y).abs()<.038)
        flame_y=y+.20+.07*side*x.sign()
        flame_h=((flame_y+.22)/.26).clamp(0,1)
        flame_center=.52+.08*(1-flame_h).square()
        flame=((sides-flame_center).abs()<.085*flame_h.sqrt()*(1-.25*flame_h))&(flame_y>-.22)&(flame_y<.04)
        lantern_shape=handle|lid|room|foot|arms|flame|(tail&(y>.26))
        panes=room&(sides<room_w-.05)
        crossbar=room&((y+.28).abs()<.025)
        glass=main*.63+.25
        lantern_rgb=paint(canvas,lantern_shape,main*.68)
        lantern_rgb=paint(lantern_rgb,panes,glass)
        lantern_rgb=paint(lantern_rgb,panes&((x+.07*view).abs()<.105),main*.53+.45)
        lantern_rgb=paint(lantern_rgb,room&(x*view<-.16),main*.58)
        lantern_rgb=paint(lantern_rgb,room&((x*view+.16).abs()<.018)|crossbar,deep)
        lantern_rgb=paint(lantern_rgb,foot&(y<.23)|lid&(y>-.42),bone)
        lantern_rgb=paint(lantern_rgb,flame,warm)
        lantern_rgb=paint(lantern_rgb,flame&(flame_h>.55)&(sides<flame_center+.01),glow)
        face=ellipse(-.038,-.045,.18,.155)
        lantern_rgb=paint(lantern_rgb,front&face,deep)
        for sign in (-1,1):
            lantern_rgb=paint(lantern_rgb,front&face&ellipse(sign*.073-.04,-.058,.032,.057),glow)
        grin=(y>.025)&(y<.046)&((x+.037).abs()<.060)
        lantern_rgb=paint(lantern_rgb,front&grin,glow)
        emblem=(x.abs()+((y+.03)*.85).abs()<.105)
        lantern_rgb=paint(lantern_rgb,(~front)&emblem,warm)
        canvas=torch.where(lantern_mask[:,:,None],lantern_rgb,canvas)
        shape=torch.where(lantern_mask,lantern_shape,shape)
        # A tapered mask with long ears, a cheek ruff, and a trailing ribbon.
        head_w=(.34*r.clamp(.78,1.36))
        mx=(x-.045*view)/head_w
        my=(y+.22)/.31
        skull=(mx.square()+(my+.13*mx.abs()).square()<1)
        e=((-.36-y)/(.37+.07*phase)).clamp(0,1)
        ear=(y<-.33)&(y>-.73-.07*phase)&((sides-(.19+.17*e)).abs()<.108*(1-e).pow(.65))
        collar_level=((y+.055)/.35).clamp(0,1)
        collar=(y>-.055)&(y<.27+.085*(17*x).cos())&(sides<.25+.15*collar_level)
        plume_t=((y-.16)/.55).clamp(0,1)
        plume_x=side*(.34*plume_t+.12*(plume_t*6.8).sin())
        plume=(y>.15)&(y<.71)&((x-plume_x).abs()<(.17*(1-plume_t).sqrt()))
        wisps=ellipse(.49,.15+.08*phase,.065,.10)|ellipse(-.49,.04-.05*phase,.064,.091)
        whiskers=(sides>.21)&(sides<.48)&((y+.14+.68*(sides-.22)).abs()<(.049-.11*(sides-.22)))
        mask_shape=skull|ear|collar|plume|wisps|whiskers
        mask_rgb=paint(canvas,mask_shape,main)
        mask_rgb=paint(mask_rgb,ear&(sides<.2+.16*e),warm)
        mask_rgb=paint(mask_rgb,collar,main*.65)
        mask_rgb=paint(mask_rgb,collar&((18*x+view*.9).cos()>.25),main*.70+.22)
        mask_rgb=paint(mask_rgb,plume&(side*(x-plume_x)<-.02)|wisps,glow)
        porcelain=skull&(my>-.70+.30*mx.abs())&(my<1.03-.47*mx.abs())
        mask_rgb=paint(mask_rgb,front&porcelain,bone)
        for sign in (-1,1):
            sx=mx-sign*.47
            eye=(sx/.27).square()+((my+.07-sign*sx*.57)/.22).square()<1
            mask_rgb=paint(mask_rgb,front&eye,dark)
            mask_rgb=paint(mask_rgb,front&eye&(sx.abs()<.055),glow)
        mask_rgb=paint(mask_rgb,front&skull&((mx.abs()+((my-.46)*1.35).abs())<.16),dark)
        mask_rgb=paint(mask_rgb,skull&((mx.abs()+((my+.39)*.81).abs())<.14),warm)
        canvas=torch.where(wisp_mask[:,:,None],mask_rgb,canvas)
        shape=torch.where(wisp_mask,mask_shape,shape)
        # Pick out the crown of the mantle and its scalloped hem.
        mantle=(~lantern_mask)&(~wisp_mask)
        canvas=paint(canvas,mantle&body&(t<.37+.08*folds)&(u<.48),main*.75+.23)
        canvas=paint(canvas,mantle&body&(t>hem-.08),main*.59+.34)
    outline=shape & (F.max_pool2d((~shape).float(),3,1,1)>.5)
    canvas=paint(canvas,outline,deep*.65)
    rgb=torch.where(shape[:,:,None],2*canvas.clamp(0,1)-1,1.)
    if detailed:
        # Preserve the relative view sizes while matching the card after adding limbs.
        coord=torch.arange(96,device=z.device)
        def length(occupied):
            return torch.where(occupied,coord,0).amax(-1)-torch.where(occupied,coord,96).amin(-1)+1
        width=length(shape.any(-2))
        height=length(shape.any(-1))
        scale=(traits[:,0]/(width/height).mean(1)).clamp(.72,1.38)
        scale=torch.minimum(scale,90/width.amax(1))
        xx=axis[None,None,:].expand(len(z),96,96)/scale[:,None,None]
        yy=axis[None,:,None].expand_as(xx)
        grid=torch.stack([xx,yy],-1)
        rgb=1+F.grid_sample(rgb.flatten(0,1)-1,grid.repeat_interleave(2,0),align_corners=True,mode="nearest").reshape_as(rgb)
    return rgb.flatten(1,2)


class FichaGenerator(nn.Module):
    def __init__(self,detailed=False,all_bodies=False):
        super().__init__()
        self.base=ProfileGenerator(learned_palette=True)
        self.register_buffer("centers",torch.zeros(GROUPS,len(FIELDS)))
        self.register_buffer("spread",torch.zeros_like(self.centers))
        self.register_buffer("freq",torch.ones(len(TYPES),len(TYPES),GROUPS)/GROUPS)
        self.local=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,24,3,padding=1))
        self.condition=nn.Linear(len(FIELDS)+len(TYPES),48)
        self.paint=nn.Sequential(nn.SiLU(),nn.Conv2d(24,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,3,1))
        nn.init.zeros_(self.paint[-1].weight)
        nn.init.zeros_(self.paint[-1].bias)
        self.overrides={}
        self.detailed=detailed
        self.all_bodies=all_bodies

    def describe(self,z,y):
        i,j=self.base.drawing.base.type_ids(y)
        choice=self.base.drawing.base.gumbel(self.freq[i,j],z[:,28:92],1.05)
        traits=self.centers[choice]+.42*self.spread[choice]*z[:,3:9].clamp(-2,2)
        traits=traits.clamp(traits.new_tensor([.30,-.85,.1,0,0,0]),traits.new_tensor([2.7,.85,1,1,1,1]))
        for key,value in self.overrides.items():
            traits[:,FIELDS.index(key)]=value
        return traits

    def draft(self,z,y,traits):
        pure=(y[:,TYPES.index("ghost")]>0)&(y.sum(1)==1)
        pairs=z.new_empty(len(z),6,96,96)
        if pure.any():
            ghosts=ghost_pair(z[pure],traits[pure],self.detailed)
            if self.detailed:
                i,j=self.base.drawing.base.type_ids(y[pure])
                ghosts=self.base.apply_profile(ghosts,z[pure],i,j,traits[pure])
            pairs[pure]=ghosts
        if (~pure).any():
            noise,tags,cards=z[~pure],y[~pure],traits[~pure]
            if self.all_bodies:
                from sprite_cuerpos import paired_bodies
                bodies=paired_bodies(noise,tags,cards)
                i,j=self.base.drawing.base.type_ids(tags)
                pairs[~pure]=self.base.apply_profile(bodies,noise,i,j,cards)
            else:
                pairs[~pure]=self.base(noise,tags,cards)
        return pairs

    def finish(self,pairs,y,traits):
        views=pairs.reshape(-1,3,*pairs.shape[-2:])
        style=self.condition(torch.cat([y,traits],1)).repeat_interleave(2,0)
        scale,bias=style[:,:,None,None].chunk(2,1)
        correction=self.paint(self.local(views)*(1+scale)+bias).tanh()
        mask=(views.amin(1,keepdim=True)<.96)
        return (views+.28*correction*mask).clamp(-1,1).reshape_as(pairs)

    def forward(self,z,y):
        traits=self.describe(z,y)
        draft=self.draft(z,y,traits)
        painted=self.finish(draft,y,traits)
        # Crisp input edges need little sharpening. Keep the learned correction
        # strongest within flat colour regions to avoid a bright halo on the ink.
        if self.detailed:
            views=draft.reshape(-1,3,96,96)
            contrast=(F.max_pool2d(views,3,1,1)+F.max_pool2d(-views,3,1,1)).amax(1,keepdim=True)
            gain=(.85-.41*contrast).clamp(.17,.85)
            return torch.lerp(views,painted.reshape_as(views),gain).reshape_as(draft)
        return painted
