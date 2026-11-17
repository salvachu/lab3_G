"""Continue observed back markings and shading across a paired sprite splice."""
import numpy as np
from PIL import Image, ImageFilter


def markings(ink, interior, seam, y):
    """Extract separated bright markings above the join, excluding edge highlights."""
    eligible=(ink>=7)&interior&(y<seam-1)
    result=[]
    for row,col in np.argwhere(eligible):
        if not eligible[row,col]:
            continue
        pending=[(row,col)]
        eligible[row,col]=False
        pixels=[]
        while pending:
            r,c=pending.pop()
            pixels.append((r,c))
            for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if 0<=rr<len(ink) and 0<=cc<ink.shape[1] and eligible[rr,cc]:
                    pending.append((rr,cc))
                    eligible[rr,cc]=False
        coords=np.asarray(pixels)
        height,width=np.ptp(coords,axis=0)+1
        if 3<=len(coords)<=38 and height<=7 and width<=9:
            result.append(coords)
    return result


def finish_back(ink,seam,lower):
    """Work in palette indices. Infer the torso extent independently for each sprite."""
    mask=ink>0
    rows,cols=np.where(mask)
    if len(rows)<30:
        return ink
    y,x=np.indices(mask.shape)
    top,bottom=rows.min(),rows.max()
    left,right=cols.min(),cols.max()
    cx,cy=(left+right)/2,(top+bottom)/2
    rx,ry=(right-left+1)/2,(bottom-top+1)/2
    ellipse=((x-cx)/rx)**2+((y-cy)/ry)**2<=1
    interior=np.array(Image.fromarray(mask.astype("uint8")).filter(ImageFilter.MinFilter(5)))>0
    # A round surface with one detailed profile and an almost blank opposite
    # flank can show pieces of a face at the back. Continue the quiet flank.
    overlap=(mask&ellipse).sum()/(mask|ellipse).sum()
    rounded=False
    if overlap>.86 and .70<rx/ry<1.35:
        sides=[]
        for side in (-1,1):
            flank=(side*(x-cx)>.50*rx)&(np.abs(y-cy)<.45*ry)&mask
            changes=(ink!=np.roll(ink,1,axis=0))|(ink!=np.roll(ink,1,axis=1))
            sides.append((changes&flank).sum()/max(1,flank.sum()))
        if max(sides)>1.5*min(sides)+.035:
            sign=1 if sides[1]>sides[0] else -1
            mirror=np.clip(np.rint(2*cx-x).astype(int),0,ink.shape[1]-1)
            ink=np.where((sign*(x-cx)>.38*rx)&(np.abs(y-cy)<.57*ry),ink[y,mirror],ink)
            mask=ink>0
            rounded=True

    # Spread a straight change of tone over the curved central torso. Feet,
    # hands, exterior ink, and side ornaments retain their own outlines.
    height=max(1,bottom-top)
    torso=np.clip(1-((x-cx)/max(1,.9*rx))**2,0,1)
    distance=y-seam
    curve=.16*height*torso
    shift=curve*np.clip(1-np.abs(distance-2)/(.32*height),0,1)
    sample=np.clip(np.rint(y-shift).astype(int),0,len(ink)-1)
    inside=interior&interior[sample,x]&(ink>=2)&(ink[sample,x]>=2)
    result=np.where(inside,ink[sample,x],ink).copy()
    # Remove isolated horizontal strips by interpolating the base shade on
    # the dorsal surface, while retaining local contrast around real markings.
    above=np.clip(seam-5,0,len(ink)-1)
    below=np.clip(seam+10,0,len(ink)-1)
    t=np.clip((distance+5)/15,0,1)
    shade=np.rint((1-t)*ink[above,x]+t*ink[below,x]).astype(int)
    smooth=inside&(ink[above,x]>=2)&(ink[below,x]>=2)&(distance>=-5)&(distance<=10)
    flat=np.abs(ink[above,x].astype(int)-ink[below,x].astype(int))<=1
    result=np.where(smooth&flat&(result<7),shade,result)

    if rounded:
        # Remove thin horizontal remnants on an otherwise uniform shell.
        clean=Image.fromarray(ink.astype("uint8"))
        median=np.asarray(clean.filter(ImageFilter.MedianFilter(7)))
        shell=np.where(interior&(np.abs(distance)<10),median,ink)
        same=(ink[np.clip(seam-9,0,len(ink)-1),x]==ink[np.clip(seam+11,0,len(ink)-1),x])
        return np.where(interior&same&(np.abs(distance)<8),ink[np.clip(seam-9,0,len(ink)-1),x],shell).astype(ink.dtype)
    spots=markings(ink,interior,seam,y)
    if len(spots)>=3:
        result=ink.copy()
        # Continue the underlying skin as well as its pattern.
        upper_skin=ink[interior&(distance<0)&(distance>=-13)&(ink<7)]
        lower_skin=ink[interior&(distance>0)&(distance<=15)&(ink<7)]
        if upper_skin.size and lower_skin.size:
            surface=np.bincount(upper_skin).argmax()
            base=np.bincount(lower_skin).argmax()
            result[(ink==base)&interior&(distance>=0)]=surface
        # Repeat the observed motifs at their native scale with staggered spacing.
        # They stay inside the main lower surface, clear of feet and outline.
        centres=np.array([p.mean(0) for p in spots])
        distances=np.linalg.norm(centres[:,None]-centres[None,:],axis=2)
        np.fill_diagonal(distances,np.inf)
        spacing=int(np.clip(np.median(distances.min(1)),7,13))
        allowed=interior&mask&(distance>=2)&(y<bottom-3)&(ink>=2)
        for row_id,row in enumerate(range(int(np.median(seam))+4,bottom-3,spacing)):
            for col_id,col in enumerate(range(int(left+4+(row_id%2)*spacing//2),right,spacing)):
                motif=spots[(row_id+2*col_id)%len(spots)]
                yy,xx=(motif-np.rint(motif.mean(0)).astype(int)+[row,col]).T
                if yy.min()>=0 and xx.min()>=0 and yy.max()<len(ink) and xx.max()<ink.shape[1] and allowed[yy,xx].all():
                    result[yy,xx]=ink[motif[:,0],motif[:,1]]
    else:
        shoulder=ink[mask&(distance>=-10)&(distance<=-2)]
        waist=ink[mask&(distance>=1)&(distance<=8)]
        if not shoulder.size or not waist.size:
            return ink
        dominant=np.bincount(shoulder).argmax()
        if (shoulder==dominant).mean()<.70 or np.bincount(waist).argmax()==dominant:
            return ink
        # Let the lower donor's shoulder and back shading reach up into a
        # blank torso. Match the shade across the join without flattening it.
        window=interior&(lower>0)&(distance>=-12)&(distance<=8)
        near=window&(distance<=-2)
        if near.sum()>16:
            offset=np.median(ink[near].astype(float)-lower[near])
            tone=np.clip(lower+offset,1,8)
            strength=np.clip((distance+13)/9,0,1)*np.clip((11-distance)/8,0,1)
            strength*=np.clip(torso*1.5,0,1)
            shaded=np.rint(result+strength*(tone-result))
            # Intermediate ranks can be a completely different hue. Draw the
            # transition only in pigments already common on this animal's back.
            ranks,counts=np.unique(ink[interior&(np.abs(distance)<16)],return_counts=True)
            active=ranks[counts>=max(2,.055*counts.sum())]
            if len(active):
                shaded=active[np.abs(shaded[...,None]-active).argmin(-1)]
            result=np.where(window,shaded,result)
    return result.astype(ink.dtype)
