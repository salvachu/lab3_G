"""Serve a local front/back sprite generator using the bundled inference weights."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit

import torch
from PIL import Image
from model import LATENT
from prepare import TYPES
from sample_cycles import load_generator

ROOT=Path(__file__).resolve().parent
NOMBRES="Normal Lucha Volador Veneno Tierra Roca Bicho Fantasma Acero Fuego Agua Planta Eléctrico Psíquico Hielo Dragón Siniestro Hada".split()
WEIGHTS={"ficha":ROOT/"pesos"/"ficha.pth","cuerpos":ROOT/"pesos"/"ficha_todos.pth","trazo":ROOT/"pesos"/"trazo.pth"}
WEIGHTS["diversa"]=ROOT/"pesos"/"diversa.pth"
for name in ("detalle", "perfiles", "pigmentos", "empalme", "jerarquica", "empalme_union", "jerarquica_nueva"):
    WEIGHTS[name] = ROOT/"pesos"/f"{name}.pth"


@torch.inference_mode()
def generate(net, types, seed, device):
    if not isinstance(types,list) or len(types) not in (1,2) or any(t not in TYPES for t in types) or len(set(types))!=len(types):
        raise ValueError("Elige uno o dos tipos distintos.")
    if seed is None:
        seed=secrets.randbelow(2**31)
    if type(seed) is not int or not 0<=seed<2**31:
        raise ValueError("Usa una semilla entera entre 0 y 2147483647.")
    z=torch.randn(1,LATENT,generator=torch.Generator().manual_seed(seed)).to(device)
    y=z.new_tensor([[t in types for t in TYPES]])
    rgb=((net(z,y)[0].reshape(2,3,96,96).clamp(-1,1)+1)*127.5).round().byte()
    images=[]
    for pixels in rgb.permute(0,2,3,1).cpu().numpy():
        stream=BytesIO()
        Image.fromarray(pixels).save(stream,format="PNG")
        images.append("data:image/png;base64,"+base64.b64encode(stream.getvalue()).decode("ascii"))
    return {"seed":seed,"types":types,"front":images[0],"back":images[1]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puerto",type=int,default=7860)
    parser.add_argument("--cpu",action="store_true")
    args=parser.parse_args()
    device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.set_num_threads(min(4,torch.get_num_threads()))
    nets={name:load_generator(path,device) for name,path in WEIGHTS.items()}
    lock=threading.Lock()
    assets=ROOT/"interfaz"

    class Handler(BaseHTTPRequestHandler):
        def send(self,status,mime,data):
            self.send_response(status)
            self.send_header("Content-Type",mime)
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def json_response(self,status,data):
            self.send(status,"application/json; charset=utf-8",json.dumps(data,ensure_ascii=False).encode())

        def do_GET(self):
            path=urlsplit(self.path).path
            if path=="/api/tipos":
                self.json_response(200,dict(zip(TYPES,NOMBRES)))
                return
            files={"/":("index.html","text/html; charset=utf-8"),"/app.css":("app.css","text/css"),
                   "/app.js":("app.js","text/javascript; charset=utf-8")}
            if path in files:
                filename,mime=files[path]
                self.send(200,mime,(assets/filename).read_bytes())
            else:
                self.send_error(404)

        def do_POST(self):
            if urlsplit(self.path).path!="/api/generar":
                self.send_error(404)
                return
            try:
                size=int(self.headers.get("Content-Length","0"))
                if not 0<size<=4096:
                    raise ValueError("Solicitud inválida.")
                data=json.loads(self.rfile.read(size))
                if not isinstance(data,dict):
                    raise ValueError("Solicitud inválida.")
                model=data.get("model","ficha")
                if model not in nets:
                    raise ValueError("Modelo desconocido.")
                with lock:
                    result=generate(nets[model],data.get("types"),data.get("seed"),device)
                    result["model"]=model
            except (ValueError,TypeError) as exc:
                self.json_response(400,{"error":str(exc)})
                return
            self.json_response(200,result)

    with ThreadingHTTPServer(("127.0.0.1",args.puerto),Handler) as server:
        print(f"Abre http://127.0.0.1:{server.server_port} en tu navegador. Modelo en {device}.",flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__=="__main__":
    main()
