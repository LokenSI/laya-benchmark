"""Nev's native calibrated readout over an owned loopback llama.cpp process."""
import atexit
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from .common import ROOT,read_json

class Nev:
    def __init__(self,meta):
        root=Path(meta['path']);sys.path.insert(0,str(root/'python'))
        from nev.backend import LlamaServer
        from nev.gguf import read_calibration
        runtime=ROOT/'.cache/llama-runtime'
        manifest=read_json(runtime/'manifest.json')
        executable=next(runtime.rglob('llama-server.exe'))
        gguf=next(root.glob('*Q8_0.gguf'))
        self.cal=read_calibration(str(gguf));assert self.cal is not None
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        self.url=f'http://127.0.0.1:{port}'
        self.log=(ROOT/'results/alternatives/logs/nev-llama-server.log').open('a',encoding='utf-8')
        self.proc=subprocess.Popen([str(executable),'-m',str(gguf),'--host','127.0.0.1','--port',str(port),'-ngl','99','-c','8192','-np','1','--no-context-shift'],stdout=self.log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        atexit.register(self.close)
        for _ in range(150):
            if self.proc.poll() is not None:raise RuntimeError('Nev llama-server exited; inspect its log')
            try:
                with urllib.request.urlopen(self.url+'/health',timeout=2) as response:
                    if response.status==200:break
            except OSError:time.sleep(2)
        else:raise RuntimeError('Nev llama-server did not become ready')
        self.backend=LlamaServer(self.url,n_probs=40,timeout=120)
        self.metadata={'precision':'Q8_0 GGUF','interface':'native Nev single-mode serving, one permutation, embedded calibration','runtime':manifest,'max_tokens':8192,'native_state_character_limit':20000,'option_limit':26,'memory_warning':'GPU work is in an external Vulkan process; Torch allocator peak does not measure Nev VRAM','probability_warning':'Native llama-server top-40 readout applies publisher missing-letter floor'}

    def batch(self,rows):
        from nev.serve import parse_request,collect_runs,plan_runs,read_answers
        from .alternatives_adapters import typed_questions,decode_typed,answer_probabilities
        out=[]
        for r in rows:
            qs,mapping=typed_questions(r['questions'])
            ex,p=parse_request({'state':r['state'],'questions':qs})
            runs,_=collect_runs(self.backend,ex,plan_runs(ex.questions,p,'single'),'single')
            result=decode_typed(answer_probabilities(read_answers(ex,runs,self.cal)),r['questions'],mapping)
            result['state_character_truncated']=len(ex.state)>20000
            out.append(result)
        return out

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
        if not self.log.closed:self.log.close()
