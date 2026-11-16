"""Comparación de muestreo con los pesos y las máscaras de Tinta · 12.500."""
from pathlib import Path

import torch

from noche_plan import status
from sample_cycles import load_generator


def main():
    base=Path("runs/noche_tinta_12500/step_12500.pt")
    dest=Path("runs/noche_tonos_12500/step_12500.pt")
    cmd=".\\.venv\\Scripts\\python noche_tonos.py"
    decision="Comparar con Tinta los mismos contornos y elegir finalistas antes de abrir la reserva."
    original=torch.load(base,map_location="cpu",weights_only=True)
    if original["architecture"]!="sprite_ink" or original["step"]!=12500:
        raise ValueError("Se requiere Tinta al paso 12.500")
    status(dest.parent.name,base,cmd,decision,"Inicio de la prueba de cuantiles desde Tinta · 12.500.")
    dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        tmp=dest.with_suffix(".tmp")
        torch.save(dict(original,rank_tones=True),tmp)
        tmp.replace(dest)
    model=load_generator(dest,"cpu")
    if not model.rank_tones or any(not torch.equal(v,original["generator"][k]) for k,v in model.state_dict().items()):
        raise ValueError("La comparación debe conservar todos los pesos")
    status("ninguna",dest,cmd,decision,"Cierre de Tonos · 12.500. Se verificaron pesos idénticos y cero actualizaciones adicionales.")


if __name__=="__main__":
    main()
