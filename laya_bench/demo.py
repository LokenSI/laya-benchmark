"""Local document intake proof of concept. No external systems are contacted."""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

from .common import ROOT,read_json
from .industry import asset_tags,keyword_route,question,validate_input


class Intake:
    def __init__(self,device="cuda"):
        import torch
        from .adapter import LayaAdapter
        torch.set_num_threads(8)
        self.lock=threading.Lock()
        self.summary=read_json(ROOT/"results/industry/summary.json")
        self.models={name:LayaAdapter(name,device=device,offline=True) for name in ["laya","laya-multilingual"]}

    def predict(self,payload):
        from .adapter import decode
        text,task,language=validate_input(payload)
        model="laya" if language=="en" else "laya-multilingual"
        key=f"{model}/{task}/{language}"
        q=question(task,language,self.summary["selected"][key])
        with self.lock:
            adapter=self.models[model]
            audit=adapter.audit(text,q)["decision"]
            if audit["state_truncated"]:
                raise ValueError("This excerpt is too long for the evaluated workflow. Paste a shorter passage.")
            answers,elapsed=adapter.predict([text],q)
        labels=list(q["decision"]["criteria"])
        pred,p=decode(answers[0]["answers"]["decision"],labels)
        measured=self.summary["results"][key]
        return {"suggestion":pred,"scores":dict(zip(labels,p)),"model":model,"language":language,"task":task,
            "asset_tags":asset_tags(text),"rules_suggestion":keyword_route(text,task),"inference_ms":round(elapsed*1000,1),
            "evidence":{"correct":measured["correct"],"n":measured["n"],"accuracy":measured["accuracy"],"accuracy_wilson95":measured["accuracy_wilson95"],"source":"Assistant-authored bilingual feasibility test; no customer data or expert validation."},
            "review_required":True,"original_text":text,"model_revision":adapter.metadata["revision"]}


def serve(port,device):
    intake=Intake(device)
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,body,kind="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type",kind)
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/","/index.html"):
                self.send(200,(ROOT/"web/intake.html").read_bytes(),"text/html; charset=utf-8")
            elif self.path=="/report":
                self.send(200,(ROOT/"results/assessment/report.html").read_bytes(),"text/html; charset=utf-8")
            elif self.path=="/health":
                self.send(200,b'{"status":"ready","local_only":true}')
            else:
                self.send(404,b'{"error":"Not found"}')

        def do_POST(self):
            if self.path!="/classify":
                self.send(404,b'{"error":"Not found"}')
                return
            origin=self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{port}",f"http://localhost:{port}"):
                self.send(403,b'{"error":"This local service accepts requests from its own page only."}')
                return
            try:
                size=int(self.headers.get("Content-Length","0"))
                if not 0<size<=24000:
                    raise ValueError("Request size is invalid.")
                data=json.loads(self.rfile.read(size))
                self.send(200,json.dumps(intake.predict(data),ensure_ascii=False).encode())
            except (ValueError,UnicodeDecodeError) as exc:
                self.send(400,json.dumps({"error":str(exc)}).encode())
            except Exception:
                self.send(500,b'{"error":"Local inference failed. Check the server log before retrying."}')
                import traceback
                traceback.print_exc()
    server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    print(f"Document intake desk ready at http://127.0.0.1:{port}",flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        for adapter in intake.models.values():
            adapter.close()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--port",type=int,default=8766)
    p.add_argument("--device",default="cuda")
    args=p.parse_args()
    serve(args.port,args.device)
