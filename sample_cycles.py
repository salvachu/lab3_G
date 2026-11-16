import argparse
import json
import random
from pathlib import Path

import torch

from generate import render
from model import Generator
from prepare import CASES, SEEDS
from type_generator import TypeGenerator


def load_generator(path,device):
    state=torch.load(path,map_location="cpu",weights_only=True)
    relief_stages=("paired_relief","paired_lineage","paired_engraved","paired_elemental","paired_prisma","paired_insignias","paired_insignias_neck")
    if state.get("architecture")=="paired_trazo":
        from sprite_trazo import TrazoGenerator
        model=TrazoGenerator()
    elif state.get("architecture")=="paired_ficha":
        from sprite_ficha import FichaGenerator
        model=FichaGenerator(detailed=state.get("detailed",False),all_bodies=state.get("all_bodies",False))
    elif state.get("architecture")=="paired_profiles":
        from sprite_profiles import ProfileGenerator
        model=ProfileGenerator(learned_palette=state.get("learned_palette",False))
    elif state.get("architecture")=="paired_spatial_latent":
        from sprite_latent import LatentGenerator
        model=LatentGenerator()
    elif state.get("architecture")=="paired_learned_detail":
        from sprite_detail import DetailedGenerator
        model=DetailedGenerator()
    elif state.get("architecture") in relief_stages:
        from sprite_relieve import ReliefGenerator
        stage=relief_stages.index(state["architecture"])
        model=ReliefGenerator(lineage=stage>=1,engraved=stage>=2,elemental=stage>=3,
                              prisma=stage>=4,insignias=stage>=5,joined=stage>=6)
    elif state.get("architecture")=="paired_sketch":
        from sprite_boceto import SketchGenerator
        model=SketchGenerator()
    elif state.get("architecture")=="paired_families":
        from sprite_families import FamilyGenerator
        model=FamilyGenerator()
    elif state.get("architecture")=="paired_parts":
        from sprite_parts import PartsGenerator
        model=PartsGenerator()
    elif state.get("architecture")=="joint_diffusion":
        from sprite_joint import JointGenerator
        model=JointGenerator()
        model.correct_clip=state.get("correct_clip",False)
    elif state.get("architecture")=="sprite_edges":
        from sprite_edges import EdgeGenerator
        model=EdgeGenerator(state["groups"],state["temperature"])
    elif state.get("architecture")=="sprite_ink":
        from sprite_ink import InkGenerator
        model=InkGenerator(state["groups"],state["temperature"])
    elif state.get("architecture")=="sprite_contours":
        from sprite_contours import ContourGenerator
        model=ContourGenerator(state["groups"],state["temperature"])
    elif state.get("architecture")=="sprite_patches":
        from sprite_patches import PatchGenerator
        model=PatchGenerator(state["groups"],state["temperature"])
    elif state.get("architecture")=="sprite_palette":
        from sprite_palette import PaletteGenerator
        model=PaletteGenerator(state["groups"],state["colors"],state["temperature"])
        model.tokenizer.decision=state.get("palette_decision","mode")
    elif state.get("architecture")=="sprite_plans":
        from sprite_plans import PlanGenerator
        model=PlanGenerator(state["groups"],state["temperature"])
    elif state.get("architecture")=="sprite_tokens":
        from sprite_tokens import TokenGenerator
        model=TokenGenerator(state["temperature"])
    else:
        model=TypeGenerator() if state.get("type_modulation",False) else Generator()
    model.load_state_dict(state["generator"])
    if state.get("rank_tones",False):
        model.rank_tones=True
    return model.to(device).eval()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint",type=Path)
    p.add_argument("--types",nargs="+",default=CASES)
    sampling=p.add_mutually_exclusive_group()
    sampling.add_argument("--seeds",type=int,nargs="+")
    sampling.add_argument("--azar",type=int,metavar="N",help="Sortear N semillas distintas para esta generación")
    p.add_argument("--output",type=Path,default=Path("muestras.png"))
    p.add_argument("--acabado",action="store_true",help="Acabado ilustrado de Insignias con cuello continuo")
    p.add_argument("--miradas",action="store_true",help="Variar la mirada según el cuerpo y la semilla en Detalle +1.200")
    p.add_argument("--postura",action="store_true",help="Activar miradas y variar la zancada de los cuerpos corredores")
    p.add_argument("--ficha",type=Path,help="JSON con los campos visuales que se quieren fijar al generar")
    a=p.parse_args()
    if a.azar is not None and not 1<=a.azar<=2**31:
        p.error("--azar requiere una cantidad positiva de semillas de 31 bits")
    seeds=random.SystemRandom().sample(range(2**31),a.azar) if a.azar is not None else (a.seeds or SEEDS)
    model=load_generator(a.checkpoint,"cuda" if torch.cuda.is_available() else "cpu")
    if a.ficha:
        from sprite_ficha import FIELDS, FichaGenerator
        if not isinstance(model,FichaGenerator):
            p.error("--ficha requiere un checkpoint de Ficha")
        values=json.loads(a.ficha.read_text(encoding="utf-8-sig"))
        ranges=dict(zip(FIELDS,[(.30,2.7),(-.85,.85),(.1,1),(0,1),(0,1),(0,1)]))
        if not isinstance(values,dict) or any(k not in ranges or type(v) not in (int,float)
                or not ranges[k][0]<=v<=ranges[k][1] for k,v in values.items()):
            p.error(f"Campos y rangos de la ficha: {ranges}")
        model.overrides=values
    if a.acabado:
        if not getattr(model,"joined",False):
            p.error("--acabado requiere un checkpoint de Insignias con cuello continuo")
        model.illustrated=True
    if a.miradas or a.postura:
        base=getattr(model,"base",None)
        if not getattr(base,"illustrated",False):
            p.error("--miradas y --postura requieren el checkpoint de Detalle +1.200")
        base.expressive=True
        base.posed=a.postura
    if a.azar is not None:
        print("Semillas:",*seeds,flush=True)
    render(model,a.types,seeds,a.output)
