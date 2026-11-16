"""Type-linked accents placed once in the shared 3-D space of the creature."""
from prepare import TYPES


def add_traits(oval, limb, types, params, hx, hy, hz, head, w, h, d, by, tail, bold=False):
    def has(name):
        i = TYPES.index(name)
        return types[:, i:i + 1]

    def param(i):
        return params[:, i:i + 1]

    water, dragon, ghost = has("water"), has("dragon"), has("ghost")
    bug, poison = has("bug"), has("poison")
    length = .63 + .85 * param(11)
    spread = .65 + .80 * param(12)
    if bold:
        length=length*1.42
        spread=spread*1.48
    # Branch geometry uses the same latent for both views. Depth determines occlusion.
    for side in (-1, 1):
        if water.any():
            root = (hx + side * head * .68, hy + .02, hz - head * .20)
            for k in range(3):
                height = (.13 - .095 * k) * length
                tip = (root[0] + side * (.14 + .05 * k) * spread,
                       root[1] - height, root[2] - .19 * length)
                limb(root, tip, .067 - .011 * k, 1 if k != 1 else 2, water, .85)
        if dragon.any():
            root = (hx + side * head * .57, hy - head * .67, hz - head * .12)
            mid = (root[0] + side * .09 * spread, root[1] - .17 * length, root[2] - .09)
            tip = (mid[0] + side * .035, mid[1] - .13 * length, mid[2] - .14 * length)
            limb(root, mid, .065, 1, dragon, .43)
            limb(mid, tip, .042, 2, dragon, .93)
        if bug.any():
            # Elbowed antennae and two-tone wing covers on the back.
            root = (hx + side * head * .47, hy - head * .64, hz + head * .43)
            mid = (root[0] + side * .11, root[1] - .19 * length, root[2] + .07)
            end = (mid[0] + side * .13 * spread, mid[1] - .075, mid[2] + .06)
            limb(root, mid, .032, 1, bug, .22)
            limb(mid, end, .027, 1, bug, .32)
            oval(*end, .046, .044, .048, 2, bug)
            oval(side * w * .47, by - h * .38, -d * .56,
                 w * .60, h * .80, d * .57, 1, bug, 1)
        if ghost.any():
            # Tapered wisps curl back; a pale core stays attached to each tip.
            root = (side * w * .63, by - h * .38, -d * .57)
            mid = (root[0] + side * .14, root[1] - .18 * length, root[2] - .10)
            tip = (mid[0] - side * .09 * spread, mid[1] - .17 * length, mid[2] - .12)
            limb(root, mid, .115, 1, ghost, .56, 1)
            limb(mid, tip, .077, 1, ghost, .90)
            limb(mid, (tip[0], tip[1] + .035, tip[2] + .025), .035, 2, ghost, .75)
    if poison.any():
        # Visible from the rear and the side: paired venom sacs and a stinger.
        for side in (-1, 1):
            oval(side * w * .74, by - h * .32, -d * .50,
                 .094 * spread, .11 * length, .10, 1, poison, 1)
        root = tail(.78)
        tip = tail(1.)
        end = (tip[0] + .055, tip[1] - .15 * length, tip[2] - .17 * length)
        limb(root, tip, .095, 1, poison, .50)
        limb(tip, end, .043, 2, poison, .94)
    if not bold:
        return
    if water.any():
        # The broad caudal lobes remain visible even on an otherwise draconic body.
        root=tail(.77)
        x,cy,cz=tail(1.)
        for direction in (-1,1):
            tip=(x+.09*spread,cy+direction*.32*length,cz-.17*length)
            limb(root,tip,.118,1,water,.91,1)
    if ghost.any():
        # A curling shroud under the torso keeps its projected path in both views.
        for side in (-1,1):
            a=(side*w*.60,by+h*.10,-d*.30)
            b=(side*(w+.08*spread),by+h*.96,-d*.57)
            c=(side*(w+.28*spread),by+h*.64,-d*.91)
            limb(a,b,.115,0,ghost,.31,1)
            limb(b,c,.082,1,ghost,.83,1)
            oval(c[0]+side*.06,c[1]-.07,c[2]-.02,.041,.065,.041,2,ghost)
    if bug.any():
        for side in (-1,1):
            root=(side*w*.22,by-h*.78,-d*.25)
            tip=(side*(w+.34*spread),by-h*.95-.14*length,-d-.09)
            limb(root,tip,.17,1,bug,.71,1)
            limb(root,tip,.042,2,bug,.89)


def signature_traits(oval,limb,y,p,hx,hy,hz,head,w,h,d,by,tail):
    def tag(name):
        return y[:,TYPES.index(name):TYPES.index(name)+1]
    ghost,water,poison=tag("ghost"),tag("water"),tag("poison")
    crest=.70+.80*p[:,4:5]
    flare=.72+.65*p[:,18:19]
    if ghost.any():
        # Two tear-shaped spirit lights flank the shoulders as they turn in 3-D.
        for side in (-1,1):
            size=(.09+.08*p[:,16:17])*(1 if side==1 else .76)
            x=side*(w+.22*flare)
            yy=by-.20-(.17+.13*side)*crest
            zz=-d*.30
            oval(x,yy,zz,size,size*1.24,size*.72,7,ghost)
            limb((x,yy,zz),(x+side*.07,yy-size*2.8,zz-.05),size*.72,1,ghost,.94)
            oval(x,yy+size*.1,zz+size*.60,size*.59,size*.85,size*.44,6,ghost)
        # A mantle across the upper back reads differently from the frontal face.
        oval(0,hy+head*.31,hz-head*.73,head*1.11,head*.62,head*.64,7,ghost,1)
    if water.any():
        # A connected dorsal sail, with staggered rays.
        for k in range(3):
            zpos=d*(.16-.35*k)
            root=(0,by-h*.70,zpos)
            tip=(.035,by-h-(.25-.050*k)*crest,zpos-.17*flare)
            limb(root,tip,.106-.014*k,1,water,.84)
            limb(root,tip,.031,6,water,.95)
    if poison.any():
        for side in (-1,1):
            x=side*(w+.01)
            yy=by-h*.12
            zz=-d*.58
            oval(x,yy,zz,.096*flare,.127*crest,.11,7,poison)
            oval(x+side*.06,yy-.024,zz+.035,.045,.057,.041,6,poison)
