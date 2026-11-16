"""Generador condicional de pares con volúmenes articulados y oclusión por profundidad."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from prepare import TYPES


class ReliefGenerator(nn.Module):
    def __init__(self,lineage=False,engraved=False,elemental=False,prisma=False,insignias=False,joined=False):
        super().__init__()
        self.register_buffer("axis",torch.linspace(-1,1,64))
        self.register_buffer("freq",torch.ones(len(TYPES)+1,8)/8)
        # Tronco ancho/alto/profundo, altura de cabeza, tamaño cabeza, largo de cola.
        self.register_buffer("builds",torch.tensor([
            [.26,.35,.27,-.36,.225,.43], [.34,.245,.43,-.21,.22,.55],
            [.205,.235,.29,-.30,.22,.78], [.35,.24,.43,-.08,.235,.49],
            [.275,.32,.25,-.39,.28,.28], [.345,.255,.38,-.13,.21,.23],
            [.28,.28,.35,-.37,.21,.64], [.39,.23,.405,-.18,.19,.42]]))
        # Los momentos de la paleta se ajustan sobre TRAIN; inicio neutro idéntico por tipo.
        self.register_buffer("colors",torch.full((len(TYPES)+1,8,2,3),.50))
        self.lineage=lineage
        self.engraved=engraved
        self.elemental=elemental
        self.prisma=prisma
        self.insignias=insignias
        self.joined=joined
        self.illustrated=False
        self.expressive=False
        self.posed=False
        self.anatomy=False
        if lineage:
            self.register_buffer("topo",torch.full((len(TYPES),len(TYPES),8),1/8))
            self.register_buffer("ink",torch.full((len(TYPES),len(TYPES),8,24),1/24))
            self.register_buffer("tones",torch.full((24,2,3),.5))
        if engraved:
            self.register_buffer("surface_mean",torch.zeros(8,208))
            self.register_buffer("surface_basis",torch.zeros(8,16,208))
        n=32+len(TYPES)+8
        self.body=nn.Sequential(nn.Linear(n,128),nn.SiLU(),nn.Linear(128,52))
        self.paint=nn.Sequential(nn.Linear(n,96),nn.SiLU(),nn.Linear(96,9))
        self.displace=nn.Sequential(nn.Linear(n,96),nn.SiLU(),nn.Linear(96,2*5*5))
        nn.init.zeros_(self.displace[-1].weight)
        nn.init.zeros_(self.displace[-1].bias)

    def choose(self,z,y):
        if self.lineage:
            i,j=self.type_ids(y)
            return self.gumbel(self.topo[i,j],z[:,:8],1.10)
        # Afinidad respecto a la tasa global: las topologías comunes no deben acaparar cada tipo.
        logit=(y@self.freq[1:].clamp_min(.002).log()/y.sum(1,keepdim=True).clamp_min(1)
               -.72*self.freq[:1].clamp_min(.002).log())
        prob=(2.4*logit).softmax(1)
        return (prob.cumsum(1)<torch.special.ndtr(z[:,:1])).sum(1).clamp_max(7)

    @staticmethod
    def type_ids(y):
        axis=torch.arange(y.shape[1],device=y.device)
        return (torch.where(y>0,axis,y.shape[1]).amin(1),torch.where(y>0,axis,0).amax(1))

    @staticmethod
    def gumbel(prob,z,strength):
        uniform=torch.special.ndtr(z).clamp(.00001,.99999)
        return (strength*prob.clamp_min(.00001).log()-(-uniform.log()).log()).argmax(1)

    def render(self,z,y,selected,hard=False,colors=None):
        n=len(y)
        fam=F.one_hot(selected,8).float()
        cond=torch.cat([z[:,:32],y,fam],1)
        raw=self.body(cond).tanh()*.72+F.pad(z[:,1:32],(0,21))*.83
        p=raw.sigmoid()
        def v(i): return p[:,i,None]
        bird,quad,snake,fish,biped,bug,dragon,shell=fam.T[:,:,None]
        tem=self.builds[selected]
        w,h,d,neck,head,longtail=[tem[:,i:i+1] for i in range(6)]
        w=w*(.72+.63*v(1))
        h=h*(.73+.55*v(2))
        d=d*(.78+.51*v(3))
        head=head*(.72+.57*v(4))
        longtail=longtail*(.73+.57*v(5))
        by=.13+.13*(v(6)-.4)
        hy=by+neck*(.75+.50*v(7))
        hz=d*(.66+.42*v(8))
        if self.lineage:
            head=head*(1.08+.10*(bird+biped))
            w=w*(1+.18*shell-.16*snake)
            h=h*(1-.18*fish-.14*bug)
            longtail=longtail*(1+.25*snake)
        if self.engraved:
            style=self.surface_mean[selected]+(self.surface_basis[selected]*z[:,12:28,None]).sum(1)
            dimensions=(style[:,200:]/2+.5).clamp(.12,1.12)
            rel=(dimensions[:,:2].mean(1,keepdim=True)/dimensions[:,2:4].mean(1,keepdim=True)).clamp(.55,1.65)
            w=w*rel.sqrt()*(1+.15*shell)
            h=h/rel.sqrt()
            d=d*(.78+.46*v(12))*(1+.20*fish)
            head=head*(.68+.65*v(25))*(1+.14*(biped+bug))
            hy=hy-.09*bird-.06*dragon+(.5-v(27))*.12
            hz=hz+.12*quad+.10*dragon-.05*biped
            longtail=longtail*(.73+.6*v(22))*(1+.27*snake)
        if self.elemental:
            # Vary the body and the head independently within each trained family.
            aspect = torch.exp(.43 * z[:, 23:24].tanh())
            w, h, d = w * aspect, h / aspect.sqrt(), d * aspect.sqrt()
            head = head * torch.exp(.33 * z[:, 17:18].tanh())
            longtail = longtail * torch.exp(.44 * z[:, 19:20].tanh())
            hy = hy - .12 * z[:, 26:27].tanh()
        if self.prisma:
            spirit=y[:,TYPES.index("ghost"):TYPES.index("ghost")+1]
            aquatic=y[:,TYPES.index("water"):TYPES.index("water")+1]
            insect=y[:,TYPES.index("bug"):TYPES.index("bug")+1]
            drake=y[:,TYPES.index("dragon"):TYPES.index("dragon")+1]
            h=h*(1-.22*aquatic+.12*spirit)
            d=d*(1+.26*insect+.24*aquatic)
            hy=hy-.13*drake
            head=head*(1+.15*spirit)
            longtail=longtail*(1+.30*drake+.22*spirit)
            if self.insignias:
                # Latent alternatives include a slender neck or a squatter, wide head.
                tall=(2*v(4)-1)
                hy=hy-(.14*drake+.15*spirit)*tall
                head=head*(1+.22*tall*(spirit+insect).clamp_max(1))
        hx=(v(9)-.5)*.15
        if self.anatomy:
            fighting=y[:,TYPES.index("fighting"):TYPES.index("fighting")+1]
            h=h*(1+.20*fighting-.13*aquatic)
            head=head*(1+.14*fighting+.12*drake)
            hy=hy-.12*fighting-.13*drake
            hz=hz+.11*fighting+.13*drake
        pitch=.19+.09*v(10)
        yaw=torch.cat([.42+.36*v(11),2.31+.46*v(11)],1)
        if self.illustrated:
            yaw=torch.cat([.22+.20*v(11),2.62+.23*v(11)],1)
            pitch=.09+.04*v(10)
        if self.posed:
            slender=((h/w-.72)*1.4).clamp(0,1)
            stride=(bird+biped*slender)*(1-aquatic)*(1-spirit)
            lean=stride*(.31+.61*torch.special.ndtr(z[:,45:46]))
            lead=torch.where(z[:,46:47]>=0,1.,-1.)
            reach=.21+.22*torch.special.ndtr(z[:,47:48])
            head=head*(1+.43*stride)
            yaw=yaw+stride*torch.cat([.44+.16*v(11),-.42-.12*v(11)],1)
        co,si=yaw.cos(),yaw.sin()
        coords=self.axis*(.84 if self.engraved else .90 if self.lineage else 1.)
        deform=F.interpolate(self.displace(cond).reshape(n,2,5,5).tanh(),(64,64),mode="bicubic",align_corners=False)*.048
        X=coords[None,None,None,:]+deform[:,:1]
        Y=coords[None,None,:,None]+deform[:,1:2]
        patches=[]
        facets=[]
        def proj(x,yc,zp):
            if self.posed:
                zp=zp-lean*(yc-by)
            dep=x*si+zp*co
            return x*co-zp*si,yc+pitch*dep,dep
        def record(xc,yc,rx,ry,ang,depth,role,point,opacity,motif):
            params=[xc,yc,rx,ry,ang,depth,point,opacity,motif]
            patches.append(([torch.broadcast_to(t+co*0,(n,2)) for t in params],role))
        def oval(x,yc,zp,rx,ry,rz,role,opacity=1,motif=0):
            xx,yy,dep=proj(x,yc,zp)
            ww=((rx*co).square()+(rz*si).square()+1e-6).sqrt()
            angle=-torch.atan(lean*si) if self.posed else 0
            record(xx,yy,ww,ry,angle,dep,role,0,opacity,motif)
        def limb(a,b,r,role,opacity=1,taper=.36,motif=0):
            x0,y0,z0=proj(*a); x1,y1,z1=proj(*b)
            dx,dy=x1-x0,y1-y0
            length=(dx.square()+dy.square()+1e-6).sqrt()
            record((x0+x1)/2,(y0+y1)/2,r,length*.50+r*.47,
                   torch.atan2(-dx,dy+1.e-5),(z0+z1)*.5,role,taper,opacity,motif)
        def triangle(a,b,c,role,opacity):
            vertices=[proj(*p) for p in (a,b,c)]
            xs,ys,ds=[torch.stack(t) for t in zip(*vertices)]
            x0,x1=xs.amin(0),xs.amax(0)
            y0,y1=ys.amin(0),ys.amax(0)
            facets.append((len(patches),vertices))
            record((x0+x1)/2,(y0+y1)/2,(x1-x0).clamp_min(.005)/2,
                   (y1-y0).clamp_min(.005)/2,0,ds.mean(0),role,0,opacity,0)
        # Cada extremo vive en coordenadas del individuo; las dos cámaras solo cambian su proyección.
        def tail(t):
            bend=(.47*snake*torch.sin(t*5.7+v(9))*t) if self.lineage else 0
            return ((.22+.58*v(12))*torch.sin(t*(1.3+1.7*v(13)))*t+bend,
                    by+t*(.12-.55*v(14))+.12*math.sin(t*3.9)*snake,
                    -d*.60-t*longtail)
        tail_visibility=(1-.75*bug)*(1-.75*fish)
        if self.joined:
            # A caudal fin or stinger needs a solid connection even on thin insect and fish bodies.
            extensions=aquatic+spirit+y[:,TYPES.index("poison"):TYPES.index("poison")+1]
            if self.anatomy:
                extensions=extensions+y[:,TYPES.index("electric"):TYPES.index("electric")+1]
            tail_visibility=torch.maximum(tail_visibility,extensions.clamp_max(1))
        for i in range(3):
            limb(tail(i/3),tail((i+1)/3),(.12+.07*v(15))*(1-.18*i),int(i==2),
                 tail_visibility,.40,i==2)
        for side in (-1,1):
            root=tail(.70); tip=tail(1.)
            limb(root,(tip[0]+side*.16,tip[1]+side*.18,tip[2]-.13),.115,1,fish+bird*.66,.80,1)
        if self.joined:
            # Project a continuous neck between the two volume centres; long-necked
            # samples can otherwise separate when viewed from the second camera.
            limb((0,by-h*.22,d*.26),(hx,hy+head*.12,hz-head*.10),
                 torch.minimum(w,head)*.67,0,1,.18,1)
        oval(0,by,0,w,h,d,0,1,1)
        oval(0,by-.06,-d*.25,w*1.09,h*1.09,d*.91,1,shell+bug*.86,1)
        oval(0,by+h*.07,d*.55,w*.78,h*.83,d*.55,2,biped+bird*.7+dragon*.75)
        for side in (-1,1):
            for end in (-1,1):
                gait=.07*(v(16)-.5)*side*end
                lx=side*w*(.65+.20*v(17))
                lz=end*d*.67
                ly=by+h*(.85+.3*v(18))+(.14+.13*v(19))*(1-.58*bug)+gait
                exists=(quad+bug+dragon+shell) if end==-1 else (quad+bug+dragon+shell+bird+biped)
                if self.prisma:
                    exists=exists*(1-spirit)*(1-.55*aquatic)
                foot_scale=1+.54*(biped+shell)+.13*quad if self.illustrated else 1.
                hip=(lx*.83,by+h*.28,lz)
                foot_y,foot_z=ly,lz+.05
                shin=(.057+.041*v(20))*foot_scale
                if self.posed:
                    swing=side*lead
                    lift=stride*(1+swing)*.5
                    foot_y=ly-.21*lift
                    foot_z=foot_z+stride*reach*swing
                    knee=(lx*1.07,(hip[1]+foot_y)*.5-.075*lift,(lz+foot_z)*.5+.19*stride)
                    bent=stride>.49
                    limb(hip,knee,shin*1.10,0,exists*bent,.12)
                    limb(knee,(lx*1.16,foot_y,foot_z),shin*.84,0,exists*bent,.20)
                    straight=exists*(~bent)
                else:
                    straight=exists
                limb(hip,(lx*1.16,foot_y,foot_z),shin,0,straight,.13)
                toes_z=foot_z+.04 if self.posed else lz+.09
                oval(lx*1.16,foot_y,toes_z,(.079+.045*v(21))*foot_scale,(.045+.04*v(22))*foot_scale,.105*foot_scale,1,exists)
            # Brazo, aleta o ala con tres primarias; el ala próxima oculta parte del torso.
            sx=side*w*.78
            root=(sx,by-h*.52,d*.19)
            width=.19+.18*v(23)+.19*dragon+.10*fish
            tip=(sx+side*width,by+.08-.35*v(24)-.12*dragon,-d*.68)
            if self.lineage:
                tip=(tip[0]+side*(.09*bird+.17*dragon),tip[1]-.14*dragon,tip[2]-.07*bird)
            arm_wing=bird+dragon+fish+(0 if self.illustrated else .65)*biped
            limb(root,tip,.096+.055*v(25),1,arm_wing,.60,1)
            if self.illustrated:
                # Bend the forelimbs forward on bipeds. Keep fins and flight feathers on other bodies.
                palm=(side*w*.77,by+h*.09,d*.92)
                limb(root,palm,.096+.028*v(25),0,biped*(1-spirit),.15)
                oval(*palm,.109,.099,.101,1,biped*(1-spirit))
            for i in (0,1,2):
                limb((sx+side*width*.48,by-h*.28,-d*.15),
                     (tip[0]-side*.058*i,tip[1]+.065*i,tip[2]-.08*i),
                     .048+.023*v(26),2 if i==1 else 1,bird+dragon,.72)
            # Aletas, placas dorsales o cresta de insecto.
            for i in (0,1):
                limb((side*w*.30,by-h*.76,-d*(i*.64)),(side*w*(.36+.28*v(27)),by-h-.10-.12*v(28),-d*(i*.64+.35)),
                     .067+.035*v(29),1,snake+fish+shell,.79)
        head_w=head*(1.02+.21*(fish+bug)-.13*dragon) if self.engraved else head*1.02
        head_h=head*(.91+.20*v(30))*(1-.22*(snake+fish)+.15*(bird+biped)) if self.engraved else head*(.91+.20*v(30))
        head_d=head*(.88+.28*(quad+dragon)-.14*biped) if self.engraved else head*.88
        if self.illustrated:
            head_w=head_w*1.12
            head_h=head_h*.87
        head_slot=len(patches)
        oval(hx,hy,hz,head_w,head_h,head_d,0,1,.40 if self.engraved else 0)
        if self.lineage:
            for side in (-1,1):
                # Seis patas y antenas acodadas, o cuernos y membranas en el animal alado.
                limb((side*w*.75,by,.015),(side*(w+.16),by+h*.92,-.025),.042,1,bug,.35)
                root=(hx+side*head*.48,hy-head*.64,hz+head*.16)
                mid=(root[0]+side*.085,root[1]-.11,root[2]+.05)
                tip=(mid[0]+side*.065,mid[1]-.08,mid[2]+.08)
                limb(root,mid,.032,1,bug,.24)
                limb(mid,tip,.025,1,bug,.18)
                oval(*tip,.043,.039,.034,2,bug)
                limb((hx+side*head*.43,hy-head*.68,hz),(hx+side*head*.92,hy-head*1.52,hz-head*.75),
                     .077,2,dragon,.85)
                limb((side*w*.6,by-h*.51,0),(side*(w+.28),by-.12,-d*.90),
                     .141,1,dragon,.72,1)
                # Placa de mejilla bajo el ojo, visible de tres cuartos.
                oval(hx+side*head*.76,hy+head*.26,hz+head*.49,.076,.050,.053,1,
                     quad+biped+dragon+bird)
        common_horns=quad+biped+bug+dragon+bird*(.15 if self.lineage else .52)
        if self.insignias:
            common_horns=common_horns*(1-spirit)*(1-insect)
        for side in (-1,1):
            limb((hx+side*head*.57,hy-head*.62,hz-.035),
                 (hx+side*head*(.96+.35*v(0)),hy-head*(1.14+.73*v(12)),hz-head*.47),
                 .055+.031*v(17),1,common_horns,.77)
        muzzle=1-.75*bird
        if self.engraved:
            muzzle=(quad+biped+dragon+snake+shell)
            if self.illustrated:
                oval(hx,hy+head_h*.28,hz+head_d*.77,head_w*.56,head_h*.42,head_d*.43,2,muzzle)
                oval(hx,hy+head_h*.20,hz+head_d*1.18,head_w*.15,head_h*.125,head_d*.14,4,muzzle)
            else:
                oval(hx,hy+head*.22,hz+head_d*.76,head_w*(.58+.18*v(19)),head_h*.37,head_d*(.60+.36*v(21)),2,muzzle)
        else:
            oval(hx,hy+head*.22,hz+head*.67,head*(.60+.18*v(19)),head*.37,head*(.53+.32*v(21)),2,muzzle)
        limb((hx,hy+head*.10,hz+head*.54),(hx,hy+head*.33,hz+head*1.83),.092,1,bird+dragon*.50,.87)
        if self.engraved:
            # Nariz, mandíbula y penacho dejan perfiles de cabeza distintos según la topología.
            oval(hx,hy+head*.18,hz+head_d*1.28,head*.24,head*.15,head*.17,4,quad)
            limb((hx,hy+head_h*.38,hz+head_d*.43),(hx,hy+head_h*.4,hz+head_d*1.34),
                 .066,1,fish+snake,.56)
            for k in range(3):
                limb((hx+head*(k-1)*.3,hy-head_h*.55,hz-head_d*.15),
                     (hx+head*(k-1)*.51,hy-head_h*(1.45+.22*v(24)-abs(k-1)*.30),hz-head_d*(.60+.19*k)),
                     .053+.021*v(26),1,bird+snake*.72+dragon*.76,.8)
        for side in (-1,1):
            ex=hx+side*head*.56; ey=hy-head*.06; ez=hz+head*.71
            if self.engraved:
                ex=hx+side*head_w*.61
                ey=hy-head_h*.115
                ez=hz+head_d*.82
                if not self.expressive:
                    limb((ex-side*.043,ey-.068,ez+.025),(ex+side*.055,ey-.087-.02*v(22),ez-.01),
                         .020,4,quad+dragon+biped,.35)
            visible=(.56*side*si+.83*co).clamp(0,1)
            visible=(visible*3.).clamp_max(1.)
            eye_scale=(1.36+.24*bug-.25*shell) if self.engraved else 1.26 if self.lineage else 1.
            if self.insignias:
                eye_scale=eye_scale*(1+.32*spirit+.14*insect)
            if self.illustrated:
                # Bring both eyes forward on the face; the muzzle used to hide the far eye.
                ex=hx+side*head_w*.48
                ey=hy-head_h*.21
                ez=hz+head_d*.96
                eye_scale=1.90+.18*v(11)
            if self.expressive:
                # Link the gaze to proportions and give individuals within a family variation.
                sleepy=((shell+.85*biped)*(1-insect)*(1-drake)+.24*z[:,39:40].tanh()).clamp(0,1)
                alert=((.80*bird+.70*drake+.46*snake)*(1-insect)+.18*z[:,40:41].tanh()).clamp(0,1)
                curious=(insect+.32*bug+.22*z[:,41:42].tanh()).clamp(0,1)
                spacing=.44+.04*drake+.04*z[:,42:43].tanh()
                xx,yy,depth=proj(hx+side*spacing*head_w,hy-head_h*.25,hz+head_d*.65)
                rx=(head_w*(.38+.07*curious)).clamp(.075,.185)
                ry=(head_h*(.40-.08*sleepy+.18*curious)).clamp(.074,.21)
                angle=-side*(.57*alert+.09*z[:,43:44].tanh())*co
                record(xx,yy,rx,ry,angle,depth+head_d*.85,3,0,visible,0)
                continue
            oval(ex,ey,ez,(.055+.022*v(23))*eye_scale,(.061+.029*v(21))*eye_scale,.044,3,visible)
            oval(ex+side*.008,ey+.002,ez+.034,(.028+.012*v(23))*eye_scale,.049*eye_scale,.023,4,visible)
            oval(ex+side*.008-.008,ey-.022,ez+.045,.016,.019,.012,5,visible)
            if self.illustrated:
                # Place the brow above the eye on the front-facing surface.
                limb((ex-side*.060,ey-.061,ez+.057),(ex+side*.052,ey-.083,ez+.054),
                     .015,4,visible,.3)
        if self.illustrated:
            fire=y[:,TYPES.index("fire"):TYPES.index("fire")+1]
            curl_slot=len(patches)
            # A caudal coil grows from the rear of compact bodies and is occluded by the trunk in front.
            curl=fire*(biped+shell)
            oval(w*.19,by+h*.22,-d*.91,w*.65,h*.65,w*.34,1,curl)
            if fire.any():
                # Seat the flame on whichever surface is higher; both cameras use that point.
                on_head=hy-head_h < by-h
                top=torch.where(on_head,hy-head_h,by-h)
                crest_z=torch.where(on_head,hz,-d*.18)
                crest_x=torch.where(on_head,hx,torch.zeros_like(hx))
                crest_h=.23+.18*v(5)
                if self.posed or self.anatomy:
                    # Both cameras see the same flame attachment point.
                    tail_flame=(z[:,48:49]>.15)&(tail_visibility>.49)
                    tx,ty,tz=tail(1.)
                    crest_x=torch.where(tail_flame,tx,crest_x)
                    crest_z=torch.where(tail_flame,tz,crest_z)
                    top=torch.where(tail_flame,ty,top)
                    crest_h=crest_h*(.78+.52*torch.special.ndtr(z[:,49:50]))
                oval(crest_x,top-.80*crest_h,crest_z,.132+.067*v(3),crest_h,.13,8,fire)
        if self.elemental:
            from sprite_adornments import add_traits
            add_traits(oval, limb, y, p, hx, hy, hz, head, w, h, d, by, tail, self.prisma)
            if self.insignias:
                from sprite_adornments import signature_traits
                signature_traits(oval,limb,y,p,hx,hy,hz,head,w,h,d,by,tail)
        if self.anatomy:
            from sprite_anatomy import add_anatomy
            add_anatomy(oval,limb,triangle,y,z,hx,hy,hz,head,w,h,d,by,tail)
        vals=torch.stack([torch.stack(part[0],-1) for part in patches],2)
        role=torch.tensor([part[1] for part in patches],device=y.device)
        # Z-buffer por pieza, de lejos a cerca para cada cámara; orden estable al empatar.
        order=vals[:,:,:,5].argsort(dim=2,stable=True)
        vals=vals.gather(2,order[:,:,:,None].expand_as(vals))
        cx,cy,rx,ry,angle,depth,point,opacity,motif=[a[:,:,:,None,None] for a in vals.unbind(-1)]
        if self.prisma:
            # Keep the two cameras at a common scale; contain long fins and horns in frame.
            bx=(rx.square()*angle.cos().square()+ry.square()*angle.sin().square()).sqrt()*1.12
            bybox=(rx.square()*angle.sin().square()+ry.square()*angle.cos().square()).sqrt()*1.12
            visible=opacity>.49
            def bounds(c,r):
                lo=torch.where(visible,c-r,torch.inf).amin(2)
                hi=torch.where(visible,c+r,-torch.inf).amax(2)
                return (lo+hi)*.5,(hi-lo)*.5
            mx,ex=bounds(cx,bx)
            my,ey=bounds(cy,bybox)
            scale=torch.maximum(ex,ey).amax(1,keepdim=True).clamp_min(.67)/.755
            X=X*scale+mx
            Y=Y*scale+my
        dx=X[:,:,None]-cx; dy=Y[:,:,None]-cy
        ax=(dx*angle.cos()+dy*angle.sin())/rx
        ay=(-dx*angle.sin()+dy*angle.cos())/ry
        dist=((ax/(1-point*.62*ay.clamp(-1,1))).square()+ay.square()+1e-5).sqrt()
        sdf=(1-dist)*torch.minimum(rx,ry)
        if facets:
            inverse=order.argsort(dim=2)
            sample=torch.arange(n,device=y.device)[:,None]
            view=torch.arange(2,device=y.device)[None,:]
            for slot,vertices in facets:
                edges=[]
                for i,(x0,y0,_) in enumerate(vertices):
                    x1,y1,_=vertices[(i+1)%3]
                    dx_edge,dy_edge=(x1-x0)[:,:,None,None],(y1-y0)[:,:,None,None]
                    cross=dx_edge*(Y-y0[:,:,None,None])-dy_edge*(X-x0[:,:,None,None])
                    edges.append(cross/(dx_edge.square()+dy_edge.square()+1.e-6).sqrt())
                edges=torch.stack(edges)
                sdf[sample,view,inverse[:,:,slot]]=torch.maximum(edges.amin(0),-edges.amax(0))
        def cut(x): return (x>0).float() if hard else (100*x).sigmoid()
        if self.engraved:
            opacity=(opacity>.49).float() if hard else (opacity>0).float()*torch.sigmoid(12*(opacity-.49))
        mask=cut(sdf)*opacity
        inner=cut(sdf-.026)
        # Normales en los ejes del individuo para mantener bandas a la misma altura al girar.
        nz=(1-ax.square()-ay.square()).clamp_min(.00001).sqrt()
        wx=ax*co[:,:,None,None,None]+nz*si[:,:,None,None,None]
        wz=-ax*si[:,:,None,None,None]+nz*co[:,:,None,None,None]
        def scalar(i): return p[:,i,None,None,None,None]
        striped=(wz*(7+5*scalar(31))+ay*2+2*scalar(32)).cos()
        spotted=(wx*(6+4*scalar(32))).cos()*(ay*(6+3*scalar(33))).cos()
        style=(z[:,5,None,None,None,None]>0).float()
        mark=cut((style*striped+(1-style)*spotted-(.48+.22*scalar(34)))*.23)*motif*cut((.65-ay)*.15)
        light=cut((-.52*ax-.66*ay+.25*nz-.38)*.24)
        dark=cut((.53*ay+.35*ax-.23)*.24)
        condcol=torch.einsum("nt,tkvc,nk->nvc",y/y.sum(1,keepdim=True).clamp_min(1),self.colors[1:],fam)
        if self.lineage:
            i,j=self.type_ids(y)
            pigment_id=self.gumbel(self.ink[i,j,selected],z[:,8:32],1.20)
            condcol=self.tones[pigment_id]
        offset=self.paint(cond).reshape(n,3,3).tanh()
        spread=.25 if self.lineage else .57
        base=(torch.logit(condcol.clamp(.06,.94))+offset[:,:2]*.8+spread*z[:,20:26].reshape(n,2,3)).sigmoid()
        main,accent=base.unbind(1)
        if self.insignias:
            # Sample an accent from each constituent's TRAIN palette. Use the
            # more contrasting one so that the secondary pigment stays legible.
            i,j=self.type_ids(y)
            def accent_from(t,noise):
                pick=self.gumbel(self.ink[t,t,selected],noise,1.4)
                return self.tones[pick,0]
            first=accent_from(i,z[:,:24].roll(7,1))
            second=accent_from(j,z[:,:24].roll(-5,1))
            dark_or_light=torch.where((first-main).square().mean(1,keepdim=True)>(second-main).square().mean(1,keepdim=True),first,second)
            accent=.25*accent+.75*dark_or_light
        if colors is not None:
            main,accent=colors.unbind(1)
        pale=(.56*main+.36+.085*offset[:,2]).clamp(0,1)
        line=.11*main+.032
        palette=torch.stack([main,accent,pale,torch.ones_like(main)*.97,line,torch.ones_like(main)],1)
        if self.insignias:
            luminous=(accent*.52+.44).clamp(0,1)
            deep=(accent*.37+line*.63).clamp(0,1)
            palette=torch.cat([palette,luminous[:,None],deep[:,None]],1)
        if self.illustrated:
            ember=main.new_tensor([.98,.29,.065]).expand_as(main)
            gold=main.new_tensor([1.,.79,.20]).expand_as(main)
            palette=torch.cat([palette,ember[:,None],gold[:,None]],1)
            if self.anatomy:
                # Wing membranes share pigment between the front and back. Highlights
                # vary with the emitted colour or the venom marking.
                teal=main.new_tensor([.12,.76,.67])
                lime=main.new_tensor([.71,.88,.17])
                electric=y[:,TYPES.index("electric"):TYPES.index("electric")+1]
                poison=y[:,TYPES.index("poison"):TYPES.index("poison")+1]
                membrane=(.53*main+.47*accent).lerp(teal,.42*aquatic)
                bright=(.55*accent+.40).lerp(lime,.87*poison).lerp(gold,.93*electric)
                palette=torch.cat([palette,membrane[:,None],bright[:,None]],1)
        roles=role[order]
        pigment=palette[torch.arange(n,device=y.device)[:,None,None],roles][:,:,:,:,None,None]
        accent=accent[:,None,None,:,None,None]
        pig=pigment*(1-.82*mark[:,:,:,None])+accent*.82*mark[:,:,:,None]
        if self.engraved:
            # Un mismo coeficiente latente controla la pareja de mapas de luminosidad de TRAIN.
            surf=self.surface_mean[selected]+(self.surface_basis[selected]*z[:,12:28,None]).sum(1)
            field=F.interpolate(surf[:,:200].reshape(n,2,10,10),(64,64),mode="bicubic",align_corners=False)
            level=field[:,:,None]/field.square().mean((1,2,3),keepdim=True).sqrt().clamp_min(.12)[:,:,None]
            eligible=(roles<=2)[:,:,:,None,None]
            patch=cut((level-.48)*.11)*eligible
            edge=cut((.14-(level-.48).abs())*.14)*eligible
            low=cut((-.72-level)*.11)*eligible
            pig=pig*(1-.52*patch[:,:,:,None])+(.5+.48*pig)*.52*patch[:,:,:,None]
            pig=pig*(1-.35*low[:,:,:,None])+.35*accent*low[:,:,:,None]
            pig=pig*(1-.23*edge[:,:,:,None])
        pig=pig*(1-.33*dark[:,:,:,None])+(1-pig)*.30*light[:,:,:,None]
        if self.illustrated:
            # Light across the silhouette; individual limbs should not each look like a bead.
            organic=(roles<=2)[:,:,:,None,None,None]
            detail=pigment*.76+pig*.24
            global_y=((Y-my)/ey.clamp_min(.12))[:,:,None]
            shadow=(.34*ay+.46*global_y+.18*ax>.22).to(pig.dtype)[:,:,:,None]
            lit=(-.23*ax-.34*ay-.35*global_y>.44).to(pig.dtype)[:,:,:,None]
            coloured=detail*(1-.25*shadow)+(1-detail)*.15*lit
            pig=torch.where(organic,coloured,pigment)
            face=(order==head_slot)[:,:,:,None,None] & (wz>.14)
            cheeks=face & (ay>.03-.26*wx.abs()) & (ay<.59)
            cream=(.54*main+.40)[:,None,None,:,None,None]
            pig=torch.where(cheeks[:,:,:,None],.34*pig+.66*cream,pig)
            # Pigment follows the single curved tail surface instead of outlining separate disks.
            coil=(order==curl_slot)[:,:,:,None,None]
            radius=(ax.square()+ay.square()+1.e-5).sqrt()
            phase=torch.atan2(ay,ax)+9.5*radius
            spiral=coil & (phase.cos()>.63) & (radius<.84)
            pig=torch.where(spiral[:,:,:,None],.25*pig+.75*line[:,None,None,:,None,None],pig)
            # A few connected facets follow the helmet from forehead to nape.
            helmet=(order==head_slot)[:,:,:,None,None] & ((shell+biped)[:,0,None,None,None,None]>.5)
            ridge=torch.cos(3.4*torch.atan2(wx,wz)+.55*ay)
            brow=ay+.42+.13*ridge
            crown=((brow.abs()<.055) | ((ridge.abs()<.11)&(brow<0))) & helmet
            warm=fire[:,0,None,None,None,None]*crown
            pigment_gold=(.25*ember+.75*gold)[:,None,None,:,None,None]
            pig=pig*(1-warm[:,:,:,None]) + pigment_gold*warm[:,:,:,None]
            # Use a tapered, curved silhouette and a light core inside each flame.
            flame=(roles==8)[:,:,:,None,None]
            t=((1-ay)*.5).clamp(0,1)
            bend=.30*torch.sin(t*6.3)-.35*t
            midline=ax-bend
            width=(1.52*(1-t).pow(.75)*torch.sin(3.14159*t).clamp_min(0).sqrt()).clamp_min(.01)
            lobe_t=((t-.10)/.66).clamp(0,1)
            lobe_width=.49*(1-lobe_t)*torch.sin(3.14159*lobe_t).clamp_min(0).sqrt()
            lobe=lobe_width-(ax+.49+.31*lobe_t).abs()
            flame_mask=cut(torch.maximum(width-midline.abs(),lobe)*.1)*((ay.abs()<1).float())
            mask=torch.where(flame,flame_mask*opacity,mask)
            core=((midline+.05).abs()<width*.47)&(t<.67)
            flame_colour=torch.where(core[:,:,:,None],gold[:,None,None,:,None,None],ember[:,None,None,:,None,None])
            pig=torch.where(flame[:,:,:,None],flame_colour,pig)
            if self.expressive:
                eye=(roles==3)[:,:,:,None,None]
                lid=(-.95+1.26*sleepy+.17*alert)[:,:,None,None,None]
                exposed=ay>lid
                mask=torch.where(eye,mask*exposed,mask)
                gaze=(.20*z[:,44].tanh())[:,None,None,None,None]
                pupil=((ax-gaze)/(.45+.15*curious[:,:,None,None,None])).square()+((ay+.02)/.78).square()<1
                glint=((ax-gaze+.21).square()+(ay+.29).square())<.13
                iris=(line+.10*main)[:,None,None,:,None,None]
                white=pigment.new_full((),.97)
                eye_rgb=torch.where((pupil & ~glint)[:,:,:,None],iris,white)
                rim=(dist>.94)|(ay<lid+.12)
                pig=torch.where(eye[:,:,:,None],torch.where(rim[:,:,:,None],iris,eye_rgb),pig)
        unlined=(roles==4)|(roles==5)
        edged=inner+(1-inner)*unlined[:,:,:,None,None]
        surface=line[:,None,None,:,None,None]*(1-edged[:,:,:,None])+pig*edged[:,:,:,None]
        future=(1-mask).flip(2).cumprod(2).flip(2)
        amount=mask*torch.cat([future[:,:,1:],torch.ones_like(future[:,:,:1])],2)
        if self.illustrated:
            # Ink the outside once. Use coloured, much lighter seams on internal overlaps.
            seams=.68*pig+.32*line[:,None,None,:,None,None]
            surface=seams*(1-edged[:,:,:,None])+pig*edged[:,:,:,None]
        rgb=future[:,:,0,None]+(surface*amount[:,:,:,None]).sum(2)
        if self.illustrated:
            alpha=1-future[:,:,0]
            erosion=1-F.max_pool2d(1-alpha,3,1,1)
            contour=(alpha-erosion).clamp(0,1)[:,:,None]
            rgb=rgb*(1-contour)+line[:,None,:,None,None]*contour
        return rgb.clamp(0,1)*2-1,1-future[:,:,0]

    def forward(self,z,y):
        rgb,_=self.render(z,y,self.choose(z,y),True)
        return F.pad(rgb.flatten(1,2),(16,16,16,16),value=1)
