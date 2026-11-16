"""Generador de partes articuladas con pesos compartidos y un renderizador diferenciable."""
import torch
from torch import nn
from torch.nn import functional as F
from prepare import TYPES


class PartsGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.prior=nn.Sequential(nn.Linear(len(TYPES),64),nn.SiLU(),nn.Linear(64,64))
        self.geometry=nn.Sequential(nn.Linear(32+len(TYPES),128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,44))
        self.palette=nn.Sequential(nn.Linear(32+len(TYPES),96),nn.SiLU(),nn.Linear(96,9))
        self.register_buffer("coords",torch.linspace(-1,1,64))

    def distribution(self,y):
        mu,logs=self.prior(y).chunk(2,1)
        return mu,logs.clamp(-2,1)

    def render(self,u,y,hard=False):
        cond=torch.cat([u,y],1)
        # Un componente directo de z preserva diversidad de proporciones aun al comienzo.
        p=(self.geometry(cond)+F.pad(u,(0,12))*.72).sigmoid()
        colors=self.palette(cond).reshape(-1,3,3).sigmoid()
        n=len(y)
        # Cada pieza está presente en las dos proyecciones. La cara solo se dibuja delante.
        def v(j): return p[:,j:j+1]
        view=p.new_tensor([[0.,1.]])
        facing=2*view-1
        width=.27+.18*v(0)
        height=.25+.22*v(1)
        headw=.17+.15*v(2)
        headh=.16+.15*v(3)
        lean=.09+.23*v(4)
        torso_y=.17-.11*v(5)
        hx=facing*lean
        hy=torso_y-height*.67-headh*.39
        bodyx=-.18*hx
        zero=0*hx
        items=[]
        x=self.coords[None,None,None,:]
        yy=self.coords[None,None,:,None]
        def ellipse(cx,cy,rx,ry,angle,role,kind=0,opacity=1):
            items.append(((cx+zero),(cy+zero),(rx+zero),(ry+zero),(angle+zero),role,kind,opacity+zero))
        # Las orejas, alas y cola admiten siluetas puntiagudas, redondas o alargadas.
        tailbase=bodyx-facing*width*.64
        tailend=tailbase-facing*(.15+.25*v(6))
        taily=torso_y-(v(7)-.38)*.53
        ellipse((tailbase+tailend)/2,(torso_y+taily)/2,.085+.075*v(8),.16+.10*v(6),
                facing*(.65+v(7)*1.4),0)
        ellipse(tailend,taily-.05,.065+.09*v(9),.09+.12*v(10),facing*(v(7)-.6),1,1)
        for sign in (-1,1):
            side=sign*(.10+.18*v(11))
            ellipse(bodyx+side,torso_y+height*.72,.075+.09*v(12),.13+.16*v(13),
                    -sign*(.15+.43*v(14)),0)
            ellipse(bodyx+side-facing*.04,torso_y+height*.72+.09+.12*v(13),
                    .09+.09*v(15),.07+.055*v(16),facing*.14,1)
        for sign in (-1,1):
            side=sign*width*.81
            ellipse(bodyx+side,torso_y-.09,.075+.10*v(17),.14+.20*v(18),
                    sign*(.40+1.1*v(19)),1,1)
        ellipse(bodyx,torso_y,width,height,facing*(v(20)-.5)*.55,0)
        for sign in (-1,1):
            ellipse(hx+sign*headw*.53,hy-headh*(.83+.20*v(21)),
                    .048+.07*v(22),.08+.18*v(23),sign*(.12+.68*v(24)),1,1)
        ellipse(hx,hy,headw,headh,facing*(v(25)-.5)*.45,0)
        ellipse(hx+facing*headw*.66,hy+headh*.19,.09+.12*v(26),.056+.085*v(27),
                facing*(v(28)-.5)*.55,2,0,1-.77*view)
        params=[torch.stack([item[i] for item in items],2)[:,:,:,None,None] for i in range(5)]
        cx,cy,rx,ry,ang=params
        dx=x.unsqueeze(2)-cx
        dy=yy.unsqueeze(2)-cy
        ax=(dx*ang.cos()+dy*ang.sin())/rx
        ay=(-dx*ang.sin()+dy*ang.cos())/ry
        point=p.new_tensor([item[6] for item in items])[None,None,:,None,None]
        distance=((ax/(1+.47*point*ay.clamp(-1,1))).square()+ay.square()+1e-5).sqrt()
        minradius=torch.minimum(rx,ry)
        sd=(1-distance)*minradius
        opacity=torch.stack([item[7] for item in items],2)[:,:,:,None,None]
        def threshold(s):
            return (s>0).float() if hard else (s*90).sigmoid()
        mask=threshold(sd)*opacity
        interior=threshold(sd-.026)
        lit=(-ax*.24-ay*.50+.22+.14*torch.cos(4*ay+2*ax))
        role=[item[5] for item in items]
        base=colors[:,role][:,None,:,:,None,None]
        highlight=threshold(lit-.40)[:,:,:,None]
        shadow=threshold(.02-lit)[:,:,:,None]
        pigment=base*(1-.36*shadow)+(1-base)*.29*highlight
        line=colors[:,0][:,None,None,:,None,None]*.23+.025
        # Marcas dorsales emparejadas y una mancha ventral más clara en el frente.
        stripes=threshold(.45-(ax*2.8+ay*1.1+v(29).reshape(n,1,1,1,1)*4).sin())
        body_layer=8
        stripe_layers=p.new_zeros(1,1,len(items),1,1)
        stripe_layers[:,:,body_layer]=1
        marking=stripes*threshold(-.07-ay)*stripe_layers
        pigment=pigment*(1-.55*marking[:,:,:,None])+colors[:,1][:,None,None,:,None,None]*.55*marking[:,:,:,None]
        surface=line*(1-interior[:,:,:,None])+pigment*interior[:,:,:,None]
        # Composición por profundidad, conservando los bordes internos visibles.
        future=(1-mask).flip(2).cumprod(2).flip(2)
        visible=mask*torch.cat([future[:,:,1:],torch.ones_like(future[:,:,:1])],2)
        rgb=future[:,:,0,None]+(surface*visible[:,:,:,None]).sum(2)
        # El ojo y su reflejo son anclajes faciales explícitos; se ocultan en la espalda.
        ex=hx-headw*.26
        ey=hy-headh*.08
        eye=((x-ex[:,:,None,None])/(.033+.026*v(30))[:,:,None,None]).square()+((yy-ey[:,:,None,None])/(.041+.03*v(31))[:,:,None,None]).square()
        eye_alpha=threshold((1-eye)*.044)*(1-view[:,:,None,None])
        rgb=rgb*(1-eye_alpha[:,:,None])+line[:,:,0]*eye_alpha[:,:,None]
        glint=((x-ex[:,:,None,None]+.014).square()+(yy-ey[:,:,None,None]+.018).square())
        gleam=threshold((.00030-glint)*30)*eye_alpha
        rgb=rgb*(1-gleam[:,:,None])+gleam[:,:,None]
        return 2*rgb.clamp(0,1)-1,1-future[:,:,0]

    def forward(self,z,y):
        mu,logs=self.distribution(y)
        rgb,_=self.render(mu+logs.exp()*z[:,:32],y,hard=True)
        return F.pad(rgb.flatten(1,2),(16,16,16,16),value=1)
