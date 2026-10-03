"""Refresh this experiment's report until its existing supervisors exit."""
import argparse
import subprocess
import time
import psutil
from .common import ROOT,read_json

def main(pids):
    processes=[]
    for pid in pids:
        try:processes.append(psutil.Process(pid))
        except psutil.NoSuchProcess:pass
    previous=None
    while True:
        alive=any(p.is_running() for p in processes)
        statuses=[]
        for name in ['queue','queue-followup','priority']:
            path=ROOT/f'results/alternatives/{name}.json'
            try:
                state=read_json(path);statuses.append((name,[(j['model'],j.get('fixture'),j['status']) for j in state['jobs']]))
            except (OSError,ValueError):pass
        try:
            live=read_json(ROOT/'results/jev_live/progress.json')
            statuses.append(('api',live['completed']//1000,live['status']))
        except (OSError,ValueError):pass
        if statuses!=previous or not alive:
            subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-m','laya_bench.alternatives_report','--charts'],cwd=ROOT,check=False)
            if (ROOT/'results/jev_live/protocol.json').exists():
                subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-m','laya_bench.jev_live_report','--charts'],cwd=ROOT,check=False)
            previous=statuses
        if not alive:break
        time.sleep(15)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('pids',type=int,nargs='+');args=parser.parse_args();main(args.pids)
