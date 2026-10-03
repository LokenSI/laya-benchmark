"""Sequential GPU supervisor. Progress survives terminal/session interruption."""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime,timezone
from .common import ROOT,write_json,read_json

def run(models,state_name='queue',after_pid=None):
    out=ROOT/'results/alternatives';logs=out/'logs';logs.mkdir(exist_ok=True)
    if after_pid:
        import psutil
        try:
            previous=psutil.Process(after_pid)
            print('Waiting for prior GPU supervisor',after_pid,flush=True)
            previous.wait()
        except psutil.NoSuchProcess:pass
    state={'pid':os.getpid(),'started':datetime.now(timezone.utc).isoformat(),'models':models,'jobs':[]}
    def save():write_json(out/f'{state_name}.json',state)
    save()
    for name in models:
        if not (out/f'models/{name}.json').exists():
            state['jobs'].append({'model':name,'status':'checkpoint_not_ready'});save();continue
        env='.venv-julia' if name=='julia' else '.venv' if name in ['gliner-decide','laya','laya-multilingual','clm-int8'] else '.venv-decision'
        python=ROOT/env/'Scripts/python.exe'
        fixtures=['jev_fresh','jev_verified','alternatives','alternatives_claims','alternatives_typed']
        if name=='intern-decision-4b':fixtures.append('intern_claims')
        for fixture in fixtures:
            log=logs/f'{name}-{fixture}.log'
            job={'model':name,'fixture':fixture,'status':'running','started':datetime.now(timezone.utc).isoformat(),'log':str(log)}
            state['jobs'].append(job);save()
            with log.open('a',encoding='utf-8') as f:
                f.write('\n=== Resumable invocation '+job['started']+' ===\n');f.flush()
                proc=subprocess.Popen([str(python),'-X','utf8','-m','laya_bench.alternatives_run',name,'--fixture',fixture,'--batch-size','8'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                job['worker_pid']=proc.pid;save()
                code=proc.wait()
            job['exit_code']=code;job['status']='finished' if code==0 else 'needs_repair';job['finished']=datetime.now(timezone.utc).isoformat();save()
            # Refresh the reviewable report after each job. Reporting failures
            # do not stop unrelated GPU jobs or become model failures.
            reporter=ROOT/'.venv/Scripts/python.exe'
            with (logs/'report-refresh.log').open('a',encoding='utf-8') as report_log:
                subprocess.run([str(reporter),'-X','utf8','-m','laya_bench.alternatives_report','--charts'],cwd=ROOT,stdout=report_log,stderr=subprocess.STDOUT)
            if code:break
    state['finished']=datetime.now(timezone.utc).isoformat();save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('models',nargs='+');p.add_argument('--state-name',default='queue');p.add_argument('--after-pid',type=int);a=p.parse_args();run(a.models,a.state_name,a.after_pid)
