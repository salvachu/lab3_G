"""A pair of foreshortened views for leaf insects, cicadas and shield beetles."""
from PIL import Image, ImageDraw, ImageOps
import numpy as np


def insect_views(noise,card,venom,blend):
    # Pose and anatomy parameters are shared before either view is painted.
    plan=int(np.argmax(noise[12:15]))
    lift=round(4*np.tanh(noise[5]))
    spread=round(5*np.tanh(noise[7]))
    tall=round(3*np.tanh(noise[4]))
    ink=(39,26,48)
    rgb=tuple(round(v*255) for v in card[3:])
    elytron=blend(rgb,[(57,142,113),(95,130,155),(86,136,82)][plan],.70)
    purple=blend(rgb,(108,48,100) if venom else (100,76,65),.65)
    shade=blend(elytron,ink,.43)
    light=blend(elytron,(210,244,179),.51)
    flash=blend((191,231,79) if venom else (255,205,87),rgb,.10)
    dark=blend(purple,ink,.43)
    def view(back):
        image=Image.new("RGB",(96,96),"white")
        p=ImageDraw.Draw(image)
        def part(coords,col,outline=True):
            p.polygon(coords,fill=col)
            if outline: p.line([*coords,coords[0]],fill=ink,width=1)
        def limb(points,colour,width):
            p.line(points,fill=ink,width=width+2,joint="curve")
            p.line(points,fill=colour,width=width,joint="curve")
        hx,hy=(44,30+tall) if plan!=2 else (40,39+tall)
        hip=(48,61)
        def wing(root,end,width,mirror):
            ax,ay=root; bx,by=end
            dx,dy=bx-ax,by-ay
            length=max(1,(dx*dx+dy*dy)**.5)
            nx,ny=-dy/length*width*mirror,dx/length*width*mirror
            def point(t,s): return (round(ax+t*dx+s*nx),round(ay+t*dy+s*ny))
            poly=[point(0,0),point(.18,-.62),point(.45,-.64),point(.75,-.44),point(1,0),
                  point(.77,.12),point(.73,.38),point(.58,.31),point(.48,.61),point(.24,.62),point(.08,.46)]
            part(poly,elytron)
            part([point(.04,0),point(.28,.1),point(.76,.02),point(1,0),point(.74,.30),point(.51,.52),point(.28,.56)],shade,False)
            p.line([point(0,0),point(.48,.04),point(.88,0)],fill=dark,width=1)
            for t in (.30,.51,.70):
                p.line([point(t,.02),point(t+.13,-.33)],fill=light,width=1)
            for t,size in ((.40,3),(.62,2)):
                cx,cy=point(t,-.28)
                p.ellipse((cx-size,cy-2,cx+size,cy+1),fill=flash)
        # Rear legs have angular knees and shared chitin cuffs.
        for x,k,foot in ((43,28,17),(52,64,75),(48,58,62)):
            yy=58+int(x==52)*5
            knee=(k,yy-3+lift)
            toes=(foot,75+int(x==52)*3)
            limb([(x,yy),knee,(foot-2,toes[1]-3),toes],dark,3)
            part([knee,(k+3,yy-6+lift),(foot,toes[1]-5),(foot-4,toes[1]-6)],elytron)
            p.line([toes,(foot+4,toes[1])],fill=flash,width=2)
        part([(40,42),(51,42),(57,50),(58,61),(53,68),(47,71),(39,66),(36,58)],purple)
        part([(47,45),(53,48),(54,59),(49,65),(42,65),(46,59)],dark,False)
        for yy in (53,58,63):
            p.line([(41,yy),(47,yy+2),(53,yy)],fill=purple if back else ink)
        width=(16,20,22)[plan]
        # Elytra close along the back or open laterally like a leaf in flight.
        left=(17-spread,65+lift) if plan!=2 else (27-spread,70)
        right=(77+spread,61-lift) if plan!=2 else (67+spread,71)
        wing((45,41),left,width,-1)
        if not back: wing((50,42),right,width,1)
        def forearm(root,elbow,tip):
            x,y=root; k,l=elbow; a,b=tip
            limb([root,elbow],purple,4)
            part([(k-2,l+1),(k,l-4),(k+4,l-3),(a+4,b-7),(a,b),(a-1,b-6)],elytron)
            p.line([(k+1,l-2),(a+2,b-7)],fill=light,width=1)
        if not back:
            forearm((40,43),(29,45+lift),(23,56+lift))
            forearm((50,44),(58,40-lift),(65,54-lift))
        # Crown and eyes share a slanted brow. Only the near eye peeks past the rear rim.
        fore=[(hx-13,hy-2),(hx-7,hy-7),(hx+7,hy-8),(hx+12,hy-3),(hx+9,hy+5),(hx,hy+10),(hx-7,hy+6)]
        part(fore,elytron)
        brow=[(hx-10,hy),(hx-15,hy-15),(hx-4,hy-11),(hx,hy-20-lift),
              (hx+6,hy-12),(hx+14,hy-14),(hx+10,hy-3),(hx+2,hy+3)]
        part(brow,purple)
        part([brow[1],brow[2],brow[3],(hx+1,hy-6),(hx+2,hy+2)],blend(purple,(245,170,195),.35),False)
        if not back:
            for offset in (-7,5):
                ex,ey=hx+offset,hy+2
                part([(ex-4,ey-2),(ex+4,ey),(ex+1,ey+4),(ex-2,ey+3)],(254,209,73))
                p.line([(ex,ey),(ex,ey+2)],fill=ink)
            p.line([(hx-2,hy+7),(hx+1,hy+7)],fill=ink)
        for side in (-1,1):
            root=(hx+side*5,hy-9)
            tip=(hx+side*(13+spread),max(3,hy-25+side*lift))
            limb([root,(hx+side*9,hy-17),tip],dark,1)
            p.ellipse((tip[0]-2,tip[1]-2,tip[0]+1,tip[1]+1),fill=flash,outline=ink)
        if back:
            wing((47,41),right,width+2,1)
            p.line([(46,42),(48,44),(49,47)],fill=light,width=2)
        return np.asarray(ImageOps.mirror(image) if back else image)
    return np.stack([view(False),view(True)])
