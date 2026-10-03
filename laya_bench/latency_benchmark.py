"""Warm, serial latency on a frozen short-message workload; no throughput claims."""
import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from .common import ROOT,write_json,read_json,digest
from .alternatives_jev import read_rows
from .jev_api import now

OUT=ROOT/'results/latency'
MODELS=['laya','laya-multilingual','decider-08b','decider-2b','decider-4b','von','gliner-decide']

def prepare():
    OUT.mkdir(exist_ok=True)
    source=ROOT/'data/prepared/jev_fresh.jsonl';rows=read_rows(source);selected=[]
    for suite in ['fresh/news','fresh/emotion','fresh/spam','fresh/phishing']:
        candidates=[r for r in rows if r['suite']==suite and r['split']=='test' and len(json.dumps(r['state'],ensure_ascii=False))<=512]
        candidates.sort(key=lambda r:hashlib.sha256(('latency-20261001/'+r['id']).encode()).hexdigest())
        assert len(candidates)>=20,(suite,len(candidates))
        selected.extend(candidates[:20])
    selected.sort(key=lambda r:hashlib.sha256(('latency-order/'+r['id']).encode()).hexdigest())
    path=ROOT/'data/prepared/latency.jsonl';body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)
    if path.exists():assert path.read_text(encoding='utf-8')==body
    else:path.write_text(body,encoding='utf-8')
    write_json(OUT/'protocol.json',{'created':now(),'fixture_sha256':digest(path),'source_sha256':digest(source),
        'cases':80,'passes':3,'warmup_calls':8,'models':MODELS+['jev-1.13.0'],
        'selection':'20 test messages per public task, serialized state at most 512 Unicode characters. Salted-ID selection and order independent of correctness. Narrow short-message workload, not long-document latency.',
        'timing':'One outstanding request; local batch size one, loaded model, eight warm-up calls excluded, GPU synchronized before and after adapter call. API uses one persistent HTTPS Session with eight warm-up calls excluded. HTTP wall time includes transport, excludes ledger bookkeeping. No retries in timed calls. Three passes retained separately.',
        'hardware':'Local RTX 5070 Ti 16 GB; 78% PyTorch allocation cap; Windows; recorded venv, checkpoint and precision. Hosted hardware unknown. Local timings include benchmark adapter tokenization and output processing.',
        'limits':'Serial response time, not requests/second or vendor-H100 speed replication. API network location and backend cache behavior are uncontrolled. All model errors and native truncation counts reported; partial runs never ranked. Repeated passes are not independent accuracy cases.'})
    print('Frozen latency workload: 80 cases x 3 passes',flush=True)

def local(name):
    import torch
    from .alternatives_adapters import create
    from .alternatives_run import validate_answer,expected_failure
    torch.set_num_threads(8);torch.manual_seed(20261001);torch.cuda.set_per_process_memory_fraction(.78)
    torch.backends.cuda.matmul.allow_tf32=False
    protocol=read_json(OUT/'protocol.json');path=ROOT/'data/prepared/latency.jsonl';assert digest(path)==protocol['fixture_sha256']
    rows=read_rows(path);dest=OUT/f'{name}.jsonl'
    assert not dest.exists(),'Preserve timing runs: use a new named experiment to rerun'
    t=time.perf_counter();adapter=create(name);loaded=time.perf_counter()-t
    write_json(OUT/f'{name}-metadata.json',{'adapter':adapter.metadata,'load_seconds':loaded,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'started':now()})
    try:
        for r in rows[:8]:adapter.batch([r])
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with dest.open('a',encoding='utf-8') as f:
            for repeat in range(3):
                for r in rows:
                    torch.cuda.synchronize();t=time.perf_counter();error=None
                    try:
                        a=adapter.batch([r])[0];torch.cuda.synchronize();elapsed=time.perf_counter()-t
                        validate_answer(r,a)
                    except Exception as e:
                        if not expected_failure(e,torch):raise
                        elapsed=time.perf_counter()-t;error=type(e).__name__;a={'pred':{}};torch.cuda.empty_cache()
                    result={'id':r['id'],'pass':repeat+1,'input_sha256':r['input_sha256'],'seconds':elapsed,
                            'error':error,'pred':a['pred'],'truncated':any(v.get('state_truncated') or v.get('instruction_truncated') or v.get('options_truncated') for v in a.get('truncation_audit',{}).values())}
                    f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
                print(name,'pass',repeat+1,'complete',flush=True)
        write_json(OUT/f'{name}-memory.json',{'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30})
    finally:
        if hasattr(adapter,'close'):adapter.close()

def api():
    import requests
    from .jev_api import Budget,Client,decode,append
    rows=read_rows(ROOT/'data/prepared/latency.jsonl');dest=OUT/'jev-1.13.0.jsonl'
    assert not dest.exists(),'Preserve timing runs'
    with requests.Session() as session:
        budget=Budget();client=Client(budget,session=session)
        try:
            for i,r in enumerate(rows[:8]+rows*3):
                warm=i<8;repeat=0 if warm else (i-8)//len(rows)+1
                payload={'model':'jev-1.13.0','state':r['state'],'questions':r['questions']}
                raw=client.request(payload,f'latency/{repeat}/{r["id"]}')
                error=raw.get('error');pred={}
                if not error:
                    assert raw['response']['model']=='jev-1.13.0'
                    pred=decode(r['questions'],raw['response'])['pred']
                result={'id':r['id'],'pass':repeat,'input_sha256':r['input_sha256'],'seconds':raw['seconds'],'error':error,'pred':pred,'truncated':False,'raw':raw,'request':payload}
                append(OUT/('api-warmup.jsonl' if warm else dest.name),result)
                if error in ['HTTP 401','HTTP 402','HTTP 403']:raise RuntimeError('Account response; stopped')
            write_json(OUT/'jev-1.13.0-metadata.json',{'mode':'Persistent HTTPS session, sequential calls, no timed retries','completed':now()})
        finally:budget.close()

def queue(supervisor_pid,after_pid=None):
    import psutil
    if after_pid:
        try:
            p=psutil.Process(after_pid);assert 'laya_bench.alternatives_priority' in p.cmdline();p.wait()
        except psutil.NoSuchProcess:pass
    parent=psutil.Process(supervisor_pid);assert 'laya_bench.alternatives_queue' in parent.cmdline()
    state={'started':now(),'supervisor_pid':supervisor_pid,'status':'waiting_for_current_worker','jobs':[]}
    parent.suspend();write_json(OUT/'queue.json',state)
    try:
        existing=read_json(ROOT/'results/alternatives/queue.json');assert existing['pid']==supervisor_pid
        for job in existing['jobs']:
            if job['status']!='running':continue
            try:
                p=psutil.Process(job['worker_pid']);assert 'laya_bench.alternatives_run' in p.cmdline();p.wait()
            except psutil.NoSuchProcess:pass
        state['status']='running';write_json(OUT/'queue.json',state)
        for name in MODELS:
            env='.venv' if name in ['laya','laya-multilingual','gliner-decide'] else '.venv-decision'
            job={'model':name,'started':now(),'status':'running'};state['jobs'].append(job);write_json(OUT/'queue.json',state)
            with (OUT/f'{name}.log').open('a',encoding='utf-8') as log:
                p=subprocess.Popen([str(ROOT/env/'Scripts/python.exe'),'-X','utf8','-m','laya_bench.latency_benchmark','local','--model',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                job['worker_pid']=p.pid;write_json(OUT/'queue.json',state);code=p.wait()
            job.update(status='complete' if code==0 else 'needs_repair',exit_code=code,finished=now());write_json(OUT/'queue.json',state)
        state.update(status='complete',finished=now());write_json(OUT/'queue.json',state)
    finally:
        if parent.is_running():parent.resume()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','api','local','queue']);p.add_argument('--model');p.add_argument('--supervisor-pid',type=int);p.add_argument('--after-pid',type=int);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='api':api()
    elif a.action=='local':local(a.model)
    else:queue(a.supervisor_pid,a.after_pid)
