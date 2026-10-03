"""Insert new paid-comparison cases at an owned GPU job boundary, then resume."""
import argparse
import subprocess
import time
import psutil
from .common import ROOT,read_json,write_json
from .jev_api import now

def run(supervisor_pid,models=None,fixtures_override=None,state_name='priority',queue_state='queue',suite_override=None):
    parent=psutil.Process(supervisor_pid)
    assert 'laya_bench.alternatives_queue' in parent.cmdline()
    out=ROOT/'results/alternatives';state={'started':now(),'supervisor_pid':supervisor_pid,'jobs':[]}
    parent.suspend()
    try:
        state['status']='waiting_for_current_worker';write_json(out/f'{state_name}.json',state)
        queue=read_json(out/f'{queue_state}.json');assert queue['pid']==supervisor_pid
        running=[j for j in queue['jobs'] if j['status']=='running']
        for job in running:
            try:
                worker=psutil.Process(job['worker_pid'])
                assert 'laya_bench.alternatives_run' in worker.cmdline()
                print('Finishing current GPU job:',job['model'],job['fixture'],flush=True)
                worker.wait()
            except psutil.NoSuchProcess:pass
        # No GPU worker remains. Only this supervisor owns the device until resume.
        state['status']='running';write_json(out/f'{state_name}.json',state)
        models=models or ['laya','laya-multilingual','decider-08b','decider-2b','decider-4b','von','gliner-decide']
        for name in models:
            env='.venv' if name in ['laya','laya-multilingual','gliner-decide'] else '.venv-decision'
            fixtures=[('jev_fresh',None),('jev_verified',None)]
            if name in ['laya','laya-multilingual']:
                fixtures.extend([('alternatives','laya_claim/'),('alternatives','industry/')])
            if fixtures_override:fixtures=[(f,suite_override) for f in fixtures_override]
            for fixture,suite in fixtures:
                job={'model':name,'fixture':fixture,'suite':suite,'started':now(),'status':'running'}
                state['jobs'].append(job);write_json(out/f'{state_name}.json',state)
                cmd=[str(ROOT/env/'Scripts/python.exe'),'-X','utf8','-m','laya_bench.alternatives_run',name,'--fixture',fixture,'--batch-size','8']
                if suite:cmd.extend(['--suite',suite])
                with (out/f'logs/priority-{name}-{fixture}.log').open('a',encoding='utf-8') as log:
                    p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    job['worker_pid']=p.pid;write_json(out/f'{state_name}.json',state)
                    code=p.wait()
                job.update(status='finished' if code==0 else 'needs_repair',exit_code=code,finished=now())
                write_json(out/f'{state_name}.json',state)
                if code:break
        state['status']='finished';state['finished']=now();write_json(out/f'{state_name}.json',state)
    finally:
        if parent.is_running():parent.resume()
        print('Resumed original GPU supervisor',supervisor_pid,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('supervisor_pid',type=int);p.add_argument('--models',nargs='+');p.add_argument('--fixtures',nargs='+');p.add_argument('--state-name',default='priority');p.add_argument('--queue-state',default='queue');p.add_argument('--suite');p.add_argument('--after-pid',type=int);a=p.parse_args()
    if a.after_pid:
        try:psutil.Process(a.after_pid).wait()
        except psutil.NoSuchProcess:pass
    run(a.supervisor_pid,a.models,a.fixtures,a.state_name,a.queue_state,a.suite)
