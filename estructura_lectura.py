"""Transfiere lecturas únicamente para recortes idénticos y muestra los que faltan sin clave."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from estructura_eval import CFG, ROOT
from prepare import CASES


def native(folder,case,i,j):
    with Image.open(folder/f"ciego_{case}.png") as sheet:
        x,y=335+j*200,74+i*104
        return sheet.crop((x,y,x+192,y+96))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase")
    p.add_argument("--anterior",action="append",default=[])
    p.add_argument("--anotaciones",type=Path)
    a=p.parse_args()
    folder=ROOT/a.phase
    def seeds(phase):
        return CFG["reserva" if phase.startswith("final") else "desarrollo"]
    def digest(im):
        return hashlib.sha256(im.tobytes()).hexdigest()
    known={}
    known_turn={}
    for prev in a.anterior:
        read=json.loads((ROOT/prev/"lectura_ciega.json").read_text(encoding="utf8"))
        turn_file=ROOT/prev/"giro_ciego.json"
        turns=json.loads(turn_file.read_text(encoding="utf8")) if turn_file.exists() else {}
        for case,cols in read.items():
            for code,(bodies,features) in cols.items():
                for i,s in enumerate(seeds(prev)):
                    im=native(ROOT/prev,case,i,ord(code)-65)
                    key=(case,s,digest(im))
                    known[key]=(int(s in bodies),features.get(str(s),""))
                    if code in turns.get(case,{}):
                        known_turn[key]=s in turns[case][code]
    manual=json.loads(a.anotaciones.read_text(encoding="utf8")) if a.anotaciones else {}
    records={}
    pose={}
    missing=[]
    for case in CASES:
        records[case]={}
        pose[case]={}
        with Image.open(folder/f"ciego_{case}.png") as im:
            cols=(im.width-335)//200
        for j in range(cols):
            code=chr(65+j)
            ref=case+" "+code
            bodies,features=[],{}
            turned=[]
            strip=Image.new("RGB",(100+192*8,96),"white")
            ImageDraw.Draw(strip).text((2,39),ref,fill="black")
            absent=False
            for i,s in enumerate(seeds(a.phase)):
                im=native(folder,case,i,j)
                key=(case,s,digest(im))
                if key in known:
                    body,feature=known[key]
                elif ref in manual:
                    bx,fx=manual[ref][:2]
                    body=str(i+1) in bx
                    feature=fx[i]
                else:
                    absent=True
                    strip.paste(im,(100+i*192,0))
                    body,feature=False,""
                if body:
                    bodies.append(s)
                if feature:
                    features[str(s)]=feature
                if known_turn.get(key) or (ref in manual and len(manual[ref])==3 and str(i+1) in manual[ref][2]):
                    turned.append(s)
            if absent:
                missing.append(strip)
            records[case][code]=[bodies,features]
            pose[case][code]=turned
    if missing:
        for k in range(0,len(missing),8):
            part=missing[k:k+8]
            sheet=Image.new("RGB",(1636,24+96*len(part)),"white")
            d=ImageDraw.Draw(sheet)
            for i,s in enumerate(seeds(a.phase)):
                d.text((110+192*i,3),str(s),fill="black")
            for i,row in enumerate(part):
                sheet.paste(row,(0,24+96*i))
            sheet.save(folder/f"pendientes_{k//8}.png")
        return
    (folder/"lectura_ciega.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf8")
    if known_turn or any(len(v)==3 for v in manual.values()):
        (folder/"giro_ciego.json").write_text(json.dumps(pose,ensure_ascii=False,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
