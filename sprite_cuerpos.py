"""Seeded pixel anatomy for the type-conditioned Ficha generator."""
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from prepare import TYPES

# Each type has three possible body plans. Dual types can inherit from either parent.
FAMILIES = {
    "normal": ("mustelid","hare","cub"), "fighting": ("pugilist","cub","raptor"),
    "flying": ("raptor","chick","wyvern"), "poison": ("salamander","moth","wyrm"),
    "ground": ("armadillo","cub","wyrm"), "rock": ("armadillo","golem","raptor"),
    "bug": ("mantis","moth","armadillo"), "ghost": ("wyrm","moth","raptor"),
    "steel": ("golem","mantis","armadillo"), "fire": ("chick","salamander","cub"),
    "water": ("mustelid","wyrm","salamander"), "grass": ("cub","mantis","chick"),
    "electric": ("hare","raptor","mustelid"), "psychic": ("hare","moth","wyrm"),
    "ice": ("mustelid","cub","raptor"), "dragon": ("wyvern","wyrm","salamander"),
    "dark": ("cub","raptor","wyvern"), "fairy": ("moth","hare","chick"),
}
ACCENTS = dict(zip(TYPES,["dfbf96","e07545","87bef0","b375db","c88f59","ba9b79",
                          "a4c94e","9475ca","aac5cf","ff8138","4cbede","80c858",
                          "f4d34b","e685b8","a2e4ec","ad83ce","756789","edaccb"]))


def blend(a,b,t):
    return tuple(round((1-t)*x+t*y) for x,y in zip(a,b))


def curve(points):
    """Closed Catmull-Rom outline sampled on the native pixel grid."""
    pts=np.asarray(points,dtype=float)
    a,b,c,d=(np.roll(pts,k,axis=0) for k in (1,0,-1,-2))
    t=np.arange(5)[None,:,None]/5
    out=(2*b[:,None]+(-a+c)[:,None]*t+(2*a-5*b+4*c-d)[:,None]*t*t+
         (-a+3*b-3*c+d)[:,None]*t**3)*.5
    return [tuple(p) for p in out.reshape(-1,2).round().astype(int)]


def render_body(types,z,card):
    if "bug" in types and set(types)<= {"bug","poison"}:
        from sprite_insecto import insect_views
        return insect_views(z,card,"poison" in types,blend)
    parents=[types[0],types[-1]]
    primary=parents[int(z[10]>0)]
    other=parents[1-int(z[10]>0)]
    styles=FAMILIES[primary]
    if set(types)=={"water","dragon"}:
        styles=("wyrm","wyvern","salamander")
    elif set(types)=={"bug","poison"}:
        styles=("mantis","moth","armadillo")
    slots=12+3*(TYPES.index(primary)%9)
    family=styles[int(np.argmax(z[slots:slots+3]))]
    accent=tuple(bytes.fromhex(ACCENTS[primary]))
    second=tuple(bytes.fromhex(ACCENTS[other]))
    main=blend(tuple(round(255*v) for v in card[3:]),accent,.26)
    accent=blend(accent,(255,251,220),.18)
    pale=blend(main,(255,245,210),.69)
    shadow=blend(main,(36,29,65),.36)
    ink=blend(main,(20,17,37),.82)
    rear_tint=blend(main,second,.35)
    tall=round(3*np.tanh(z[2]))
    pose=round(4*np.tanh(z[6]))
    tilt=round(3*np.tanh(z[9]))
    classmark=set(types)

    def draw(back):
        im=Image.new("RGB",(96,96),"white")
        pen=ImageDraw.Draw(im)
        def patch(pts,color,smooth=True,edge=True):
            line=curve(pts) if smooth else pts
            pen.polygon(line,fill=color)
            if edge:
                pen.line([*line,line[0]], fill=ink, width=1)
        def oval(box,color,edge=True):
            pen.ellipse(box,fill=color,outline=ink if edge else None)
        def stroke(pts,col,width=1):
            pen.line(pts,fill=col,width=width)

        serpent=family=="wyrm"
        airborne=family in ("moth","mantis","wyvern","raptor","chick")
        low=family in ("mustelid","armadillo","salamander","cub")
        heavy=family in ("armadillo","golem","pugilist")
        hx=32 if serpent or low else 40 if airborne else 42
        hy=(37 if low else 31)+tall
        if family=="moth": hx,hy=47,36+tall
        if family=="mantis": hx,hy=41,28+tall
        # The tail articulates at the hip and swings to the other side in the rear projection.
        def tail():
            if "electric" in classmark:
                patch([(55,61),(63,59),(66,48),(72,51),(78,32+pose),(82,36+pose),
                       (74,60),(69,56),(66,66),(58,68)],second if other=="electric" else accent,False)
            elif serpent or "water" in classmark or "dragon" in classmark or low:
                patch([(50,59),(64,62),(71,56),(75,42+pose),(81,36+pose),(81,51),
                       (76,67),(65,74),(49,73)],main)
                if "water" in classmark:
                    patch([(73,49),(69,36+pose),(70,29+pose),(77,36+pose),(81,33+pose),
                           (89,30+pose),(87,40+pose),(79,50)],second if other=="water" else accent)
                elif "fire" in classmark:
                    patch([(73,45+pose),(74,36+pose),(79,30+pose),(78,24+pose),(85,32+pose),
                           (85,40+pose),(79,47+pose)],(246,109,43))
                    oval((77,35+pose,81,43+pose),(255,211,83),False)
                else:
                    stroke([(59,67),(68,65),(74,58)],pale,2)
        wings=family in ("wyvern","moth","raptor") or bool({"flying","fairy"} & classmark)
        def wing(s,far=False):
            def pts(seq):
                return [(48+s*xx,50+yy) for xx,yy in seq]
            if family=="moth" or bool({"bug","fairy"} & classmark):
                patch(pts([(2,1),(10,-17),(29,-25+pose),(35,-20),(32,-4),(22,9),
                           (29,13),(23,22),(14,20),(3,9)]),shadow if far else rear_tint)
                patch(pts([(6,-1),(15,-14),(27,-19),(26,-12),(17,-3)]),accent,True,False)
                oval((48+s*20-3,47,48+s*20+3,53),second if "poison" in classmark else pale)
            elif "dragon" in classmark or family=="wyvern":
                patch(pts([(2,8),(9,-13),(22,-29+pose),(32,-36+pose),(29,-16),(37,0),
                           (27,-3),(21,7),(15,3),(8,13)]),shadow if far else accent,False)
                stroke(pts([(3,8),(16,-13),(31,-33+pose),(26,-3)]),ink)
                stroke(pts([(16,-13),(21,5)]),shadow,2)
            else:
                patch(pts([(2,7),(10,-8),(33,-22+pose),(28,-10),(21,-5),(30,-7),
                           (21,5),(11,12),(4,13)]),main,False)
                stroke(pts([(7,6),(16,0),(26,-11)]),pale,2)
        if wings:
            wing(-1,True)
            wing(1)
        tail()
        if not serpent:
            for leg in (0,1):
                x=(33+leg*22) if low else (39+leg*15)
                y=64+leg*2
                if family=="mantis" or family=="moth":
                    stroke([(x+5,57),(x-4,65+leg*2),(x-8,71)],ink,2)
                else:
                    stride=pose if leg else -pose
                    patch([(x-4,y-6),(x+6,y-4),(x+6+stride,y+7),(x+9+stride,y+10),
                           (x-3+stride,y+11),(x-5+stride,y+8)],shadow if leg==0 else main)
                    stroke([(x-2+stride,y+8),(x-3+stride,y+10)],pale,2)
        if serpent:
            patch([(hx-7,hy+2),(hx+6,hy-1),(43,45),(38,56),(41,65),(54,66),(66,60),
                   (70,63),(61,74),(45,77),(31,73),(27,63),(31,52),(34,43)],main)
            patch([(33,47),(31,61),(35,68),(45,74),(55,72),(44,70),(36,63),(37,51)],pale,True,False)
        elif family=="mantis":
            patch([(39,37),(46,38),(46,49),(56,54),(65,67),(61,72),(51,66),(42,54)],main)
            for dx,dy in ((-1,0),(16,5)):
                patch([(40+dx,44+dy),(30+dx,50+dy),(20+dx,42+dy),(18+dx,31+dy),
                       (25+dx,38+dy),(29+dx,40+dy),(37+dx,37+dy)],accent,False)
        elif family=="moth":
            patch([(43,40),(51,39),(54,54),(51,67),(47,73),(42,63),(40,51)],main)
            for yy in (55,62):
                stroke([(44,yy),(50,yy+1)],accent,2)
        else:
            if family=="salamander":
                contour=[(24,52),(32,45),(44,45),(60,47),(70,51),(76,57),(69,61),(54,61),(42,64),(29,61)]
            elif family=="mustelid":
                contour=[(25,49),(34,40),(40,44),(50,46),(57,43),(69,46),(72,55),(65,64),(53,65),(42,60),(31,58)]
            elif family=="golem":
                contour=[(24,45),(30,36),(44,34),(63,38),(69,47),(62,62),(51,68),(34,63)]
            elif low:
                contour=[(26,48),(34,42),(46,43),(54,39),(66,41),(72,49),(70,61),
                         (61,66),(47,67),(38,62),(29,60)]
            elif heavy:
                contour=[(28,43),(36,38),(55,37),(66,43),(64,55),(58,66),(45,69),(32,62),(26,53)]
            elif family in ("chick","hare"):
                contour=[(35,42),(45,37),(53,40),(58,49),(63,61),(57,68),(42,69),(32,64),(29,56)]
            else:
                contour=[(34,44),(42,37),(51,41),(54,51),(65,56),(62,66),(51,68),(40,63),(32,56)]
            patch(contour,rear_tint if back else main,smooth=family!="golem")
            if not back:
                patch([(32,47),(40,44),(43,50),(48,56),(53,62),(48,65),(38,61),(32,55)],pale,True,False)
            if family=="armadillo" or "steel" in classmark or "rock" in classmark:
                patch([(41,44),(48,37),(59,36),(68,41),(72,52),(67,59),(56,57),(45,53)],shadow)
                stroke([(45,43),(54,44),(61,53),(70,51)],pale,2)
            if family=="hare":
                patch([(53,52),(60,57),(59,66),(65,70),(66,73),(51,73),(47,70),(52,65)],main)
            if family=="mustelid" or family=="cub":
                patch([(hx-9,hy+6),(hx+7,hy+5),(hx+15,hy+11),(hx+11,hy+12),
                       (hx+13,hy+17),(hx+6,hy+15),(hx+2,hy+18),(hx-2,hy+14),
                       (hx-9,hy+14),(hx-6,hy+11)],pale if family=="mustelid" else shadow,False)
            if family=="pugilist" or "fighting" in classmark:
                for x,y in ((26,48+pose),(57,41-pose)):
                    patch([(x+6,y+10),(x+2,y+3),(x-4,y),(x-6,y-6),(x-2,y-10),
                           (x+5,y-9),(x+8,y-4),(x+7,y+1),(x+12,y+6)],main)
                    stroke([(x-3,y),(x+5,y+1)],pale,2)
        if not serpent and family not in ("moth","mantis"):
            # Bridge the shoulder and skull even at the extreme sampled head heights.
            patch([(hx-5,hy+4),(hx+7,hy+1),(hx+10,hy+9),(47,45),(49,51),
                   (40,54),(hx-3,hy+14)],rear_tint if back else main,True,False)
        # Type-linked anatomy, rooted on the neck or spine in both projections.
        for typ in types:
            color=tuple(bytes.fromhex(ACCENTS[typ]))
            if typ in ("grass","ice","rock","steel","dragon","ground"):
                for k in range(3):
                    x,y=(46+6*k,69-k) if serpent else (49+6*k,43+k)
                    tip=(x+5+tilt,y-14+k)
                    patch([(x-3,y),tip,(x+8,y-4),(x+5,y+1)],color,typ=="grass")
                    stroke([(x+1,y-2),tip],pale)
            elif typ=="fire":
                for k in range(3):
                    x,y=(47+5*k,70-k) if serpent else (51+5*k,44+k)
                    patch([(x-4,y),(x-5,y-5),(x,y-11),(x+1,y-17+pose+k),
                           (x+6,y-9),(x+5,y-2)],(239,95,45))
                    stroke([(x,y-3),(x+2,y-9)],(255,211,75),2)
            elif typ=="ghost":
                patch([(46,45),(59,42),(65,46),(69,58),(75,63),(65,61),(60,67),(57,59),(46,55)],color)
            elif typ in ("poison","fairy","psychic"):
                for x,y,s in ((51,47,3),(61,50,2)):
                    oval((x-s,y-s,x+s,y+s),color)
                    pen.point((x-1,y-1),fill=pale)
            elif typ=="water":
                x,y=(49,69) if serpent else (51,44)
                patch([(x,y),(x+5,y-13+pose),(x+10,y-18+pose),(x+10,y-7),(x+14,y-5),(x+10,y+1)],color,False)
            elif typ=="electric":
                patch([(49,45),(56,42),(53,47),(60,46),(50,54),(53,49),(47,50)],color,False,False)
            elif typ=="dark":
                patch([(41,42),(49,35),(55,38),(62,33),(58,43),(66,43),(56,50),(48,49)],shadow,False)
            elif typ=="normal":
                patch([(33,46),(37,41),(48,44),(53,49),(47,48),(44,54),(40,50),(35,52)],pale,False)
        # Horns/ears are behind the skull. The muzzle, eyes and beak are front-only.
        ear_h=15+tilt if family=="hare" else 7+tilt
        horns=("dragon" in classmark or family=="golem")
        feelers=("bug" in classmark or family in ("mantis","moth"))
        for dx in (-6,7):
            s=-1 if dx<0 else 1
            x=hx+dx
            if feelers:
                stroke([(x,hy-6),(x+s*4,hy-18),(x+s*9,hy-20+tilt)],ink,2)
                oval((x+s*9-2,hy-22+tilt,x+s*9+2,hy-18+tilt),accent)
            elif horns or family in ("hare","cub","mustelid","pugilist") or "electric" in classmark:
                tip=(x+s*(7+tilt),hy-ear_h-(9 if horns or "electric" in classmark else 4))
                patch([(x-4,hy-3),tip,(x+4,hy-6),(x+4,hy-1)],pale if horns else main,False)
                stroke([(x,hy-6),tip],accent)
        if family in ("chick","raptor") or "flying" in classmark:
            patch([(hx-5,hy-7),(hx+1,hy-17),(hx+8,hy-21+tilt),(hx+4,hy-13),
                   (hx+13,hy-15),(hx+7,hy-6)],accent,False)
        if family=="mantis":
            head=[(-14,-5),(-9,-10),(4,-10),(12,-7),(8,2),(0,8),(-6,6)]
        elif heavy or family=="wyvern":
            head=[(-13,-5),(-7,-10),(4,-10),(12,-4),(13,3),(7,9),(-3,10),(-14,6),(-17,1)]
        else:
            head=[(-13,-1),(-10,-8),(-3,-12),(6,-10),(11,-5),(12,2),(8,8),(0,10),(-9,7),(-15,4)]
        patch([(hx+x,hy+y) for x,y in head],main)
        if "water" in classmark or "dragon" in classmark:
            patch([(hx+6,hy),(hx+14,hy-5),(hx+20,hy-7),(hx+15,hy),
                   (hx+18,hy+2),(hx+10,hy+5)],accent,False)
        if not back:
            patch([(hx-13,hy+1),(hx-7,hy+2),(hx-3,hy+4),(hx+4,hy+3),(hx+2,hy+7),
                   (hx-4,hy+8),(hx-10,hy+6)],pale,True,False)
            for ex,ey,radius in ((hx-10,hy-3,2),(hx+2,hy-4,3)):
                oval((ex-radius,ey-radius,ex+radius,ey+radius+1),(253,245,212))
                oval((ex-1,ey-1,ex+1,ey+2),ink,False)
                pen.point((ex-1,ey-1),fill="white")
                if heavy or "dark" in classmark:
                    stroke([(ex-radius-1,ey-radius-1),(ex+radius,ey-radius+1)],shadow,2)
            if family in ("raptor","chick"):
                patch([(hx-12,hy),(hx-20,hy+4),(hx-12,hy+6),(hx-8,hy+3)],accent,False)
            else:
                stroke([(hx-12,hy+3),(hx-10,hy+4),(hx-8,hy+3)],ink)
        else:
            patch([(hx-8,hy-5),(hx-4,hy-8),(hx+6,hy-6),(hx+8,hy-3),(hx+1,hy-4)],rear_tint,True,False)
            stroke([(hx-6,hy+3),(hx-2,hy+6),(hx+2,hy+4),(hx+6,hy+6),(hx+9,hy+3)],shadow)
            if wings:
                wing(1)
            elif not heavy and not serpent:
                stroke([(45,49),(51,50),(55,54)],shadow,2)
        return ImageOps.mirror(im) if back else im
    return np.stack([np.asarray(draw(v)) for v in (False,True)])


def paired_bodies(z,y,traits):
    images=[render_body([TYPES[i] for i in np.flatnonzero(tags)],noise,card)
            for noise,tags,card in zip(z.detach().cpu().numpy(),y.detach().cpu().numpy(),traits.detach().cpu().numpy())]
    pixels=torch.from_numpy(np.stack(images)).to(z).permute(0,1,4,2,3)/255
    # Cel shading follows colour regions, with the same light direction in both views.
    foreground=pixels.amin(2,keepdim=True)<.94
    paint=foreground & (pixels.mean(2,keepdim=True)>.20)
    upper=torch.roll(pixels,(2,2),(-2,-1))
    lower=torch.roll(pixels,(-3,-2),(-2,-1))
    lit=(pixels-upper).abs().amax(2,keepdim=True)>.18
    dark=(pixels-lower).abs().amax(2,keepdim=True)>.18
    pixels=torch.where(paint&dark,pixels*.73+.015,torch.where(paint&lit,pixels*.82+.18,pixels))
    return (pixels*2-1).reshape(-1,6,96,96)
