"""Folded surfaces and articulated appendages anchored to one creature in 3-D."""
import torch
from prepare import TYPES


def add_anatomy(oval, limb, triangle, y, z, hx, hy, hz, head, w, h, d, by, tail):
    def tag(name):
        i=TYPES.index(name)
        return y[:,i:i+1]
    water,dragon,ghost=tag("water"),tag("dragon"),tag("ghost")
    fight,electric,bug=tag("fighting"),tag("electric"),tag("bug")
    grass,ice,poison=tag("grass"),tag("ice"),tag("poison")
    spread=.75+.65*torch.special.ndtr(z[:,52:53])
    rise=.65+.70*torch.special.ndtr(z[:,53:54])
    if dragon.any() or bug.any() or grass.any() or ice.any():
        # Root, wrist and rays form an actual membrane, visible from both cameras.
        wings=torch.maximum(dragon*(1-.8*water),bug)
        for s in (-1,1):
            root=(s*w*.55,by-h*.55,-d*.22)
            wrist=(s*(w+.20*spread),by-h-.39*rise,-d*.46)
            tip=(s*(w+.57*spread),by-h-.56*rise,-d*.99)
            mid=(s*(w+.36*spread),by-h*.15,-d*1.42)
            fold=(s*(w+.13*spread),by-h*.51,-d)
            far=(s*(w+.04),by+h*.06,-d*1.67)
            triangle(root,wrist,mid,10,wings)
            triangle(wrist,tip,mid,10,wings)
            triangle(root,fold,far,10,wings)
            limb(root,wrist,.045,0,wings,.23)
            limb(wrist,tip,.037,1,wings,.76)
            limb(wrist,mid,.022,2,wings,.56)
            jewel=(s*(w+.19*spread),by-h*.82-.05*rise,-d*.88)
            oval(*jewel,.066,.060,.05,11,bug*poison)
            oval(s*(w+.31*spread),by-h*.81-.12*rise,-d*.87,.074,.043,.055,11,bug*poison)
            if dragon.any():
                cheek=(hx+s*head*.47,hy+head*.12,hz+head*.52)
                jaw=(hx+s*head*.25,hy+head*.41,hz+head*1.36)
                triangle(cheek,(hx+s*head*.70,hy+head*.31,hz+head*.92),jaw,2,dragon)
            if grass.any() or ice.any():
                base=(s*w*.49,by-h*.63,-d*.26)
                end=(s*(w+.25*spread),by-h-.34*rise,-d*.75)
                outer=(s*(w+.28*spread),by-h*.67,-d*.63)
                inner=(s*w*.36,by-h-.12*rise,-d*.57)
                leaves=torch.maximum(grass,ice)
                triangle(base,outer,end,10,leaves)
                triangle(base,end,inner,2,leaves)
                limb(base,end,.025,1,leaves,.65)
    if fight.any():
        lead=torch.where(z[:,54:55]>0,1.,-1.)
        for s in (-1,1):
            ahead=.5+.5*s*lead
            shoulder=(s*w*.70,by-h*.48,d*.36)
            elbow=(s*(w+.13*spread),by+h*.33,d*.44)
            fist=(s*(w+.04+.15*ahead),by-h*(.42+.50*ahead),d+.16+.15*ahead)
            limb(shoulder,elbow,.12,0,fight,.22)
            limb(elbow,fist,.13,2,fight,.18)
            oval(*fist,.14*spread,.155*spread,.16,1,fight)
            oval(fist[0],fist[1]+.052,fist[2]+.077,.12,.042,.105,2,fight)
    if electric.any():
        a=tail(.79)
        b=(a[0]+.34*spread,a[1]-.16*rise,a[2]-.14)
        c=(b[0]-.15,b[1]-.04,b[2]-.10)
        e=(c[0]+.34*spread,c[1]-.45*rise,c[2]-.16)
        triangle(a,(b[0]+.09,b[1]+.09,b[2]),c,11,electric)
        triangle(c,b,e,11,electric)
        limb(a,b,.062,11,electric,.25)
        limb(b,c,.063,11,electric,.20)
        limb(c,e,.049,11,electric,.64)
        for s in (-1,1):
            root=(hx+s*head*.50,hy-head*.52,hz)
            end=(hx+s*(head+.15*spread),hy-head-.27*rise,hz-.12)
            corner=(hx+s*(head+.10),hy-head*.67,hz-.035)
            triangle(root,corner,end,11,electric)
    if water.any():
        for s in (-1,1):
            a=tail(.75)
            b=tail(1.)
            end=(b[0]+s*.30*spread,b[1]-.12*rise,b[2]-.19)
            triangle(a,end,(b[0],b[1]+.095,b[2]+.01),10,water)
    if ghost.any():
        for s in (-1,1):
            upper=(s*w*.62,by-h*.77,-d*.44)
            outer=(s*(w+.19*spread),by+h*.22,-d*.73)
            hem=(s*(w+.34*spread),by+h+.16*rise,-d)
            triangle(upper,outer,hem,10,ghost)
            triangle(upper,hem,(s*w*.38,by+h*.63,-d*.81),7,ghost)
