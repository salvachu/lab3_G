"""Generador articulado con cuatro topologías, estadísticas de TRAIN y dos proyecciones."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from prepare import TYPES


class FamilyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("axis",torch.linspace(-1,1,64))
        self.register_buffer("freq",torch.full((len(TYPES)+1,4),.25))
        self.register_buffer("pigments",torch.full((len(TYPES)+1,4,2,3),.52))
        self.geometry=nn.Sequential(nn.Linear(32+len(TYPES)+4,112),nn.SiLU(),nn.Linear(112,45))
        self.shades=nn.Sequential(nn.Linear(32+len(TYPES)+4,96),nn.SiLU(),nn.Linear(96,6))
        self.warp=nn.Sequential(nn.Linear(32+len(TYPES)+4,96),nn.SiLU(),nn.Linear(96,4*5*5))
        nn.init.zeros_(self.warp[-1].weight)
        nn.init.zeros_(self.warp[-1].bias)

    def mixture(self,y):
        # El prior de familias combina evidencia de ambos tipos.
        return (y@self.freq[1:].clamp_min(.01).log()/y.sum(1,keepdim=True).clamp_min(1)).softmax(1)

    def choose(self,z,y):
        draw=.5*(1+torch.erf(z[:,0]/math.sqrt(2)))
        return (self.mixture(y).cumsum(1)<draw[:,None]).sum(1).clamp_max(3)

    def render(self,z,y,family,hard=False):
        n=len(z)
        fam=F.one_hot(family,4).float()
        cond=torch.cat([z[:,:32],y,fam],1)
        p=(.85*torch.tanh(self.geometry(cond))+F.pad(z[:,1:32],(0,14))*.84).sigmoid()
        weight=y/y.sum(1,keepdim=True).clamp_min(1)
        pigment=torch.einsum("nt,tfkc,nf->nkc",weight,self.pigments[1:],fam)
        color=(torch.logit(pigment.clamp(.04,.96))+.65*torch.tanh(self.shades(cond).reshape(n,2,3))
               +.65*z[:,20:26].reshape(n,2,3)).sigmoid()
        v=lambda i:p[:,i:i+1,None,None]
        bird,quad,snake,fin=[fam[:,i:i+1,None,None] for i in range(4)]
        rear=p.new_tensor([0.,1.])[None,:,None,None]
        face=2*rear-1
        side=face*(.42+.32*v(1))
        warp=F.interpolate(self.warp(cond).reshape(n,4,5,5).tanh(),(64,64),mode="bicubic",align_corners=False)*.095
        x=self.axis[None,None,None,:]+warp[:,:2]
        ygrid=self.axis[None,None,:,None]+warp[:,2:]
        W=(.27+.19*v(2))*(.98+.30*quad+.17*fin-.28*snake)
        H=(.23+.17*v(3))*(1+.23*bird+.14*snake-.17*quad)
        by=.08+.11*v(4)
        bx=-side*(.06+.05*v(5))
        hw=(.19+.12*v(6))*(1+.18*snake-.10*bird)
        hh=(.17+.13*v(7))*(1+.10*bird-.25*fin)
        hx=bx+side*W*(.72+.35*quad+.25*fin)
        hy=by-H*(.73+.35*snake+.16*bird-.28*fin)
        parts=[]
        def add(cx,cy,rx,ry,angle,kind,role,visible=1,mark=0):
            zero=0*face
            parts.append(tuple((a+zero).expand(n,2,1,1) for a in (cx,cy,rx,ry,angle,kind,visible,mark))+(role,))
        # Cola alargada, abanico de plumas, cadena serpentina o aleta caudal.
        for j in (3,2,1):
            t=j/3
            add(bx-side*(W+.15+t*(.23+.20*v(8))),by+(.03-.21*v(9))*t+snake*(.14-.24*t)*torch.sin(4*v(10)+t*3),
                (.13+.09*v(11))*(1-.18*t),(.083+.075*v(12))*(1-.36*t),
                face*(v(9)-.5),.1,0,quad+snake)
        for sign in (-1,1):
            add(bx-side*(W+.29),by+.05+sign*.095,.095+.07*v(13),.18+.095*v(14),
                face*(.72+sign*.56),.75,1,fin+bird)
        # Patas delanteras y traseras, patas de ave o ninguna pata.
        for k in (-1,1):
            lx=bx+k*W*(.55+.18*quad)+rear*.02
            if k==-1:
                add(lx+.092,by+H*.70,.082,.18,face*.22,.05,1,quad)
            add(lx,by+H*.82,(.064+.058*v(15))*(1-.35*bird),(.10+.14*v(16))*(1+.22*bird),
                k*(.18+.29*v(17)),.03,0,quad+bird)
            add(lx+face*.038,by+H*.82+.085+.105*v(16),.088+.065*v(18),.048+.053*v(19),
                face*.12,.12,1,quad+bird)
        add(bx,by,W,H,face*(v(20)-.5)*.43,.04,0,1,1)
        # El ala de perfil pasa al centro de la espalda y abre hacia el otro lado.
        for k in (-1,1):
            wingx=bx+k*W*(.30+.42*rear)
            add(wingx,by-.03,(.09+.12*v(21))*(1+.25*bird),.20+.18*v(22),
                k*(.67+.58*v(23)),.85,1,(bird+fin)*(1 if k==1 else .38+.62*rear),1)
        # Cresta dorsal serial del reptil y del pez.
        for j in (-1,0,1):
            add(bx+j*W*.50,by-H*.88-.025,.05+.05*v(24),.085+.07*v(25),
                face*(.25+.3*v(26)),.85,1,snake+fin)
        for j in (-1,1):
            add(hx+j*hw*.61,hy-hh*.85,.056+.058*v(27),.095+.14*v(28),
                j*(.19+.53*v(29)),.84,1,quad+bird*.40)
        add(hx,hy,hw,hh,face*(v(30)-.5)*.50,.10,0)
        add(hx+face*hw*.86,hy+hh*.20,(.10+.076*v(31))*(1+.40*bird),
            .054+.054*v(32),face*(.05+.24*bird),.45+.47*bird,1,1-.76*rear)
        cx,cy,rx,ry,angle,point,opacity,mark=[torch.stack([part[i] for part in parts],2) for i in range(8)]
        dx=x[:,:,None]-cx
        dy=ygrid[:,:,None]-cy
        ax=(dx*angle.cos()+dy*angle.sin())/rx
        ay=(-dx*angle.sin()+dy*angle.cos())/ry
        dist=((ax/(1+.58*point*ay.clamp(-1,1))).square()+ay.square()+1.e-5).sqrt()
        sd=(1-dist)*torch.minimum(rx,ry)
        def thr(s): return (s>0).float() if hard else (90*s).sigmoid()
        mask=thr(sd)*opacity
        inner=thr(sd-.028)
        shade=thr((.26*ax+.57*ay-.08)*.4)
        light=thr((-.32*ax-.53*ay-.35)*.4)
        base=color[:,[item[-1] for item in parts]][:,None,:,:,None,None]
        pig=base*(1-.34*shade[:,:,:,None])+(1-base)*.26*light[:,:,:,None]
        marking=thr(((ax*(3.3+2*v(33)[:,:,None])+ay*.9+v(34)[:,:,None]*3).cos()-.38)*.15)*mark*thr((.68-ay)*.3)
        accent=color[:,1][:,None,None,:,None,None]*.63
        pig=pig*(1-.67*marking[:,:,:,None])+accent*.67*marking[:,:,:,None]
        line=color[:,0][:,None,None,:,None,None]*.19+.025
        surface=line*(1-inner[:,:,:,None])+pig*inner[:,:,:,None]
        future=(1-mask).flip(2).cumprod(2).flip(2)
        visible=mask*torch.cat([future[:,:,1:],torch.ones_like(future[:,:,:1])],2)
        rgb=future[:,:,0,None]+(surface*visible[:,:,:,None]).sum(2)
        ex=hx+face*hw*.25
        ey=hy-hh*.04
        eye=thr((1-((x-ex)/(.038+.020*v(35))).square()-((ygrid-ey)/(.050+.018*v(36))).square())*.048)*(1-rear)
        gleam=thr((.00030-(x-ex+.014).square()-(ygrid-ey+.018).square())*33)*eye
        rgb=rgb*(1-eye[:,:,None])+line[:,:,0]*eye[:,:,None]
        rgb=rgb*(1-gleam[:,:,None])+gleam[:,:,None]
        return rgb.clamp(0,1)*2-1,1-future[:,:,0]

    def forward(self,z,y):
        rgb,_=self.render(z,y,self.choose(z,y),True)
        return F.pad(rgb.flatten(1,2),(16,16,16,16),value=1)
