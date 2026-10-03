"""Task-matched serial timing and device-memory experiment for expanded plots."""
import argparse
import atexit
import ctypes
import hashlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone

from .common import ROOT, read_json, write_json, digest
from .alternatives_jev import read_rows

OUT = ROOT / 'results/tradeoff_expansion'
JEV = 'jev-1.13.0'
SUITES = ['fresh/news', 'fresh/emotion', 'fresh/spam', 'fresh/phishing',
          'massive_en', 'massive_nb', 'public_jevbench', 'kev_claim/devtools-v1',
          'industry/routing/en', 'industry/routing/nb', 'industry/documents/en', 'industry/documents/nb']
TIERS = ['jevbench_public/easy', 'jevbench_public/original', 'jevbench_public/hard']


def now(): return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    write_json(temp, obj)
    temp.replace(path)


def prepare():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'data/prepared/jev_live.jsonl'
    rows = read_rows(source)
    original = read_json(ROOT / 'results/jev_live/comparison.json')
    protocol = {'created': now(), 'source_sha256': digest(source), 'passes': 3,
                'selection': 'Salted ID order independent of predictions. Existing 20 short messages/task for fresh tasks. 20 paired language IDs. 20 public JevBench cases stratified 4 easy, 6 original, 10 hard. 12 each for developer tools and industrial tasks. Full original inputs, no new truncation.',
                'timing': 'Batch one, three serial passes, eight warm-ups per task excluded, GPU synchronized. Median over successful answered calls; failures, abstentions and native truncation retained. Incomplete blocks are never plotted.',
                'memory': 'NVML device-used bytes sampled every 20 ms during each warmed task block, minus the median idle baseline measured before model load. Includes runtime and external native processes; excludes initial model-loading peak. An incremental whole-device estimate, not process-isolated VRAM. Desktop activity can affect it.',
                'cache': 'CLM: candidate/action embeddings precomputed for this fixed workload; every request state re-encoded. Winnow: prefix reuse off. Other adapters retain their native runtime behavior, recorded in metadata.',
                'hardware': 'RTX 5070 Ti 16 GB, Windows, existing pinned Python venvs, one GPU model at a time; PyTorch memory allocation cap 78%. Jev uses sequential persistent HTTPS; hosted memory unknown.',
                'groups': {}, 'models': {}}
    previous = {s: [] for s in SUITES[:4]}
    for r in read_rows(ROOT / 'data/prepared/latency.jsonl'): previous[r['suite']].append(r['id'])
    rank = lambda r: hashlib.sha256(('tradeoff-expanded-20261001/' + r['id']).encode()).hexdigest()
    selected = []
    for suite in SUITES:
        subset = [r for r in rows if (r['suite'] in TIERS if suite == 'public_jevbench' else r['suite'] == suite)
                  and r.get('split') != 'development']
        if suite in previous:
            pick = [r for r in subset if r['id'] in previous[suite]]
        elif suite == 'public_jevbench':
            pick = [r for s,n in zip(TIERS,[4,6,10]) for r in sorted([x for x in subset if x['suite']==s],key=rank)[:n]]
        elif suite in ['massive_en', 'massive_nb']:
            # The same source-family ranking gives corresponding EN/NB examples.
            pick = sorted(subset, key=lambda r: hashlib.sha256(('language-pair/'+str(r.get('family') or r['id'].split('/',1)[1])).encode()).hexdigest())[:20]
        else: pick = sorted(subset, key=rank)[:12]
        assert len(pick) == (20 if suite in SUITES[:7] else 12)
        protocol['groups'][suite] = {'accuracy_n': len(subset), 'timing_cases': len(pick)}
        selected += [{**r, 'timing_suite': suite} for r in sorted(pick, key=rank)]
    for model, record in original['models'].items():
        groups = []
        for suite in SUITES:
            src = TIERS if suite == 'public_jevbench' else [suite]
            if all(record['suites'].get(s,{}).get('complete') for s in src): groups.append(suite)
        protocol['models'][model] = groups
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected)
    dest = OUT / 'fixture.jsonl'
    if dest.exists():
        assert dest.read_text(encoding='utf-8') == payload, 'Frozen timing fixture changed'
        return
    dest.write_text(payload, encoding='utf-8')
    protocol['fixture_sha256'] = digest(dest)
    write_json(OUT / 'accuracy-snapshot.json', original)
    write_json(OUT / 'protocol.json', protocol)
    print('Frozen',len(selected),'timing cases for',len(protocol['models']),'models',flush=True)


class DeviceMemory:
    class Info(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]
    def __init__(self):
        self.lib = ctypes.WinDLL('nvml.dll')
        assert self.lib.nvmlInit_v2() == 0
        self.handle = ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) == 0
        self.lock = threading.Lock()
        self.values = []
        self.stop = threading.Event()
        self.thread = None
    def read(self):
        obj = self.Info()
        assert self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(obj)) == 0
        return obj.used / 2**30
    def begin(self):
        self.values = [self.read()]
        self.stop.clear()
        def loop():
            while not self.stop.wait(.02):
                with self.lock: self.values.append(self.read())
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
    def end(self):
        self.stop.set()
        self.thread.join()
        self.values.append(self.read())
        return {'device_peak_gib':max(self.values), 'device_min_gib':min(self.values), 'samples':len(self.values)}


def local(name):
    import numpy as np
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(300,repeat=True)
    protocol = read_json(OUT / 'protocol.json')
    assert digest(OUT / 'fixture.jsonl') == protocol['fixture_sha256']
    groups = protocol['models'][name]
    directory = OUT / name
    directory.mkdir(exist_ok=True)
    state_path = directory / 'summary.json'
    state = read_json(state_path) if state_path.exists() else {'model':name,'groups':{},'started':now()}
    todo = [s for s in groups if not state['groups'].get(s,{}).get('complete')]
    if not todo: print(name,'already complete',flush=True); return
    meter = DeviceMemory()
    base = []
    for _ in range(10): base.append(meter.read()); time.sleep(.1)
    baseline = float(np.median(base))
    assert baseline < 4, f'GPU not idle before model load: {baseline:.2f} GiB'
    import torch
    from .alternatives_adapters import create, typed_questions
    from .alternatives_run import validate_answer, expected_failure
    torch.set_num_threads(8)
    torch.manual_seed(20261001)
    torch.cuda.set_per_process_memory_fraction(.78)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    adapter = create(name)
    if hasattr(adapter,'close'): atexit.register(adapter.close)
    state.update(adapter=adapter.metadata, load_seconds=time.perf_counter()-started,
                 idle_baseline_gib=baseline, idle_range_gib=[min(base),max(base)],
                 fixture_sha256=protocol['fixture_sha256'], memory_method=protocol['memory'])
    rows = read_rows(OUT / 'fixture.jsonl')
    if name == 'clm-int8':
        from .clm_run import schema
        # Never persist measurement-only cache changes into the accuracy cache.
        atexit.unregister(adapter.encoder.save)
        adapter.encoder.save = lambda: None
        adapter.encoder.cache = {}
        actions = set()
        for r in rows:
            if r['timing_suite'] not in groups: continue
            q,_ = typed_questions(r['questions'])
            pairs = schema.build_pairs(r['state'],q)
            actions.update(t for pair in pairs.values() for t in pair[2])
        adapter.encoder.embed(sorted(actions), batch_tokens=1024)
        state['adapter']['timing_cache_policy'] = 'Candidate embeddings warm; state embeddings deleted before every request, including repeated passes. No writes to the shared cache.'
    save(state_path,state)
    try:
        for suite in todo:
            cases = [r for r in rows if r['timing_suite']==suite]
            dest = directory / (suite.replace('/','--')+'.jsonl')
            if dest.exists():
                dest.rename(dest.with_name(dest.name+'.interrupted-'+str(time.time_ns())))
            def query(r):
                if name == 'clm-int8':
                    q,_=typed_questions(r['questions'])
                    for pair in schema.build_pairs(r['state'],q).values():
                        adapter.encoder.cache.pop(hashlib.sha256(pair[0].encode()).hexdigest(),None)
                return adapter.batch([r])[0]
            torch.cuda.empty_cache()
            warm_errors=[]
            for r in cases[:8]:
                try: query(r)
                except Exception as e:
                    if not expected_failure(e,torch): raise
                    warm_errors.append(type(e).__name__)
                    torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            meter.begin()
            measurements=[]
            with dest.open('w',encoding='utf-8') as f:
                for repeat in range(1,4):
                    for r in cases:
                        torch.cuda.synchronize()
                        began=time.perf_counter()
                        error=None
                        try:
                            before_truncated = getattr(getattr(adapter,'encoder',None),'truncated',0)
                            a=query(r)
                            torch.cuda.synchronize()
                            elapsed=time.perf_counter()-began
                            validate_answer(r,a)
                        except Exception as e:
                            if not expected_failure(e,torch): raise
                            elapsed=time.perf_counter()-began
                            error=type(e).__name__+': '+str(e)[:350]
                            a={'pred':{},'probabilities':{}}
                            torch.cuda.empty_cache()
                        abstained=any(a.get('abstained',{}).values())
                        truncated=any(v.get('state_truncated') or v.get('options_truncated') or v.get('instruction_truncated') for v in a.get('truncation_audit',{}).values())
                        truncated=bool(truncated or a.get('state_character_truncated') or getattr(getattr(adapter,'encoder',None),'truncated',0)>before_truncated)
                        record={'id':r['id'],'suite':suite,'pass':repeat,'input_sha256':r['input_sha256'],
                                'seconds':elapsed,'error':error,'abstained':abstained,'truncated':truncated,
                                'pred':a['pred']}
                        measurements.append(record)
                        f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
            memory=meter.end()
            values=[r['seconds']*1000 for r in measurements if not r['error'] and not r['abstained']]
            block={'complete':True,'calls':len(measurements),'unique_cases':len(cases),
                   'answered':len(values),'failures':sum(bool(r['error']) for r in measurements),
                   'abstentions':sum(r['abstained'] for r in measurements),
                   'truncated_calls':sum(r['truncated'] for r in measurements),
                   'median_ms':float(np.median(values)) if values else None,
                   'p95_ms':float(np.quantile(values,.95)) if values else None,
                   **memory,'added_device_peak_gib':max(0,memory['device_peak_gib']-baseline),
                   'torch_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                   'torch_peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                   'warmup_errors':warm_errors,'finished':now(),'predictions_sha256':digest(dest)}
            state['groups'][suite]=block
            save(state_path,state)
            print(name,suite,'median',round(block['median_ms'] or 0,1),'ms','VRAM',round(block['added_device_peak_gib'],2),'GiB',f'{len(values)}/{len(measurements)} answered',flush=True)
        state['finished']=now();save(state_path,state)
    finally:
        if hasattr(adapter,'close'):adapter.close()


def api():
    import numpy as np
    import requests
    from .jev_api import Budget, Client, decode
    directory=OUT/JEV;directory.mkdir(exist_ok=True)
    state_path=directory/'summary.json'
    state=read_json(state_path) if state_path.exists() else {'model':JEV,'started':now(),'groups':{}}
    rows=read_rows(OUT/'fixture.jsonl')
    with requests.Session() as session:
        budget=Budget();client=Client(budget,session)
        try:
            for suite in SUITES:
                if state['groups'].get(suite,{}).get('complete'):continue
                cases=[r for r in rows if r['timing_suite']==suite]
                measurements=[]
                path=directory/(suite.replace('/','--')+'.jsonl')
                if path.exists():path.rename(path.with_name(path.name+'.interrupted-'+str(time.time_ns())))
                with path.open('w',encoding='utf-8') as f:
                    for i,r in enumerate(cases[:8]+cases*3):
                        repeat=0 if i<8 else 1+(i-8)//len(cases)
                        result=client.request({'model':JEV,'state':r['state'],'questions':r['questions']},f'tradeoff-expansion/{suite}/{repeat}/{r["id"]}')
                        error=result.get('error');a={'pred':{}}
                        if not error:
                            assert result['response']['model']==JEV
                            a=decode(r['questions'],result['response'])
                        if repeat:
                            v={'id':r['id'],'suite':suite,'pass':repeat,'input_sha256':r['input_sha256'],
                               'seconds':result['seconds'],'error':error,'abstained':False,'truncated':False,'pred':a['pred']}
                            measurements.append(v);f.write(json.dumps(v)+'\n');f.flush()
                        if error in ['HTTP 401','HTTP 402','HTTP 403']:raise RuntimeError('API account response: '+error)
                ms=[v['seconds']*1000 for v in measurements if not v['error']]
                state['groups'][suite]={'complete':True,'calls':len(measurements),'unique_cases':len(cases),
                    'answered':len(ms),'failures':len(measurements)-len(ms),'abstentions':0,'truncated_calls':0,
                    'median_ms':float(np.median(ms)) if ms else None,'p95_ms':float(np.quantile(ms,.95)) if ms else None,
                    'added_device_peak_gib':None,'finished':now(),'predictions_sha256':digest(path)}
                save(state_path,state);print(JEV,suite,'complete',flush=True)
        finally:budget.close()


def queue(supervisor_pid):
    import psutil
    parent=psutil.Process(supervisor_pid)
    assert 'laya_bench.alternatives_queue' in parent.cmdline()
    state={'pid':os.getpid(),'supervisor_pid':supervisor_pid,'started':now(),'status':'waiting_for_worker','jobs':[]}
    parent.suspend()
    try:
        q=read_json(ROOT/'results/alternatives/queue-followup.json')
        assert q['pid']==supervisor_pid
        save(OUT/'queue.json',state)
        for job in q['jobs']:
            if job['status']!='running':continue
            try:
                p=psutil.Process(job['worker_pid'])
                assert 'laya_bench.alternatives_run' in p.cmdline()
                print('Waiting for',job['model'],job['fixture'],flush=True)
                p.wait()
            except psutil.NoSuchProcess:pass
        protocol=read_json(OUT/'protocol.json')
        # Run the requested additions and compact references first.
        first=['laya','laya-multilingual','decider-4b','decision-4b-v12','winnow-12b-q8','cygnet-12b-nf4','jev-omni-12b-nf4','clm-int8',JEV]
        order=first+[m for m in protocol['models'] if m not in first]
        state['status']='running';save(OUT/'queue.json',state)
        for name in order:
            env='.venv-julia' if name=='julia' else '.venv' if name in ['gliner-decide','laya','laya-multilingual','clm-int8',JEV] else '.venv-decision'
            job={'model':name,'started':now(),'status':'running'}
            state['jobs'].append(job);save(OUT/'queue.json',state)
            with (OUT/(name+'.log')).open('a',encoding='utf-8') as log:
                cmd=[str(ROOT/env/'Scripts/python.exe'),'-X','utf8','-m','laya_bench.tradeoff_expansion','api' if name==JEV else 'local','--model',name]
                process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                job['worker_pid']=process.pid;save(OUT/'queue.json',state)
                code=process.wait()
            job.update(exit_code=code,status='complete' if code==0 else 'needs_repair',finished=now());save(OUT/'queue.json',state)
            print(name,job['status'],flush=True)
        state.update(status='finished',finished=now());save(OUT/'queue.json',state)
    finally:
        if parent.is_running():parent.resume()
        print('Resumed benchmark supervisor',supervisor_pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','local','api','queue']);p.add_argument('--model');p.add_argument('--supervisor-pid',type=int);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='local':local(a.model)
    elif a.action=='api':api()
    else:queue(a.supervisor_pid)
