"""Control de los pesos neuronales manteniendo idénticas las estadísticas de TRAIN."""
import argparse
from pathlib import Path

import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint",type=Path)
    a=p.parse_args()
    final=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    start=torch.load(a.checkpoint.parent/"inicio.pt",map_location="cpu",weights_only=True)
    assert final["architecture"]==start["architecture"]=="paired_relief"
    for key in ("colors","freq"):
        start["generator"][key]=final["generator"][key]
    start.update(data=final["data"],estadisticas_train=True)
    torch.save(start,a.checkpoint.parent/"calibrado_00000.pt")


if __name__=="__main__":
    main()
