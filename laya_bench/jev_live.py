"""Resumable, budgeted live Jev comparison on the exact shared requests."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
from .common import ROOT, read_json, write_json, digest, fingerprint
from .alternatives_jev import read_rows
from .alternatives_adapters import typed_questions, decode_typed
from .jev_api import OUT, Budget, Client, decode, append, now

FIXTURES=['jev_fresh','jev_verified','alternatives_claims','alternatives_typed','alternatives']

def prepare():
    sources={}; rows=[]
    for name in FIXTURES:
        path=ROOT/f'data/prepared/{name}.jsonl'
        source=read_rows(path);sources[name]={'sha256':digest(path),'n':len(source)}
        if name=='alternatives_claims': source.sort(key=lambda r:(not r['suite'].startswith('jevbench_public/'),r['id']))
        if name=='alternatives': source.sort(key=lambda r:(not r['suite'].startswith(('industry/','laya_claim/')),r['id']))
        rows.extend({**r,'source_fixture':name} for r in source)
    assert len(rows)==len({r['id'] for r in rows})
    dest=ROOT/'data/prepared/jev_live.jsonl'
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if dest.exists(): assert dest.read_text(encoding='utf-8')==payload
    else: dest.write_text(payload,encoding='utf-8')
    protocol={'created':now(),'fixture_sha256':digest(dest),'n':len(rows),'sources':sources,
              'model':read_json(OUT/'model-pin.json'),'planned_concurrency':4,
              'selection':'All rows from the five frozen common fixtures, including weak tasks and all failed local cases. Whole task groups, never only local successes. Strong-task selection is retrospective; fresh subset is separately frozen and reported.',
              'requests':'Exact state, questions, option descriptions and order. Native boolean expansion for multi-label tasks matches other typed adapters; GLiNER native multi-label is identified separately. Raw payload and response retained without credentials.',
              'scoring':'Shared argmax, ordered ties (lexical ties for public JevBench native score). Native selected labels also retained. Probability normalization only inside publisher tolerance 0.02; raw vectors retained. Errors stay in denominator. Per-suite metrics; no pooled leaderboard.',
              'latency':'Concurrent client wall time includes internet transport and connection setup; not a hardware-normalized latency benchmark or proof of advertised speed ratios.',
              'cost':'Billable input token totals times the dated published $42/billion price; output tokens free. $25 ledger cap with conservative pre-call reservation; unconfirmed attempts retain reservation. Account balance is not exposed by the API.',
              'historic_comparison':'Fresh calls to resolved Jev 1.13.0; separate from published historical Jev 1.13.0 outcomes. Stable and preview aliases resolved to the same version in probes; not counted as distinct models.'}
    if (OUT/'protocol.json').exists():
        prior=read_json(OUT/'protocol.json');assert prior['fixture_sha256']==protocol['fixture_sha256']
    else: write_json(OUT/'protocol.json',protocol)
    print('Frozen live API comparison:',len(rows),'requests',flush=True)

def evaluate(row,client,model):
    qs,mapping=typed_questions(row['questions'])
    payload={'model':model,'state':row['state'],'questions':qs}
    history=[]
    for retry in range(3):
        raw=client.request(payload,row['id']);history.append(raw)
        if raw.get('status_code') not in [429,500,502,503,504] and raw.get('error') not in ['ConnectTimeout','ReadTimeout','ConnectionError']: break
        if retry<2: time.sleep(2**retry)
    base={k:row[k] for k in ['id','suite','input_sha256','gold','source_fixture']}
    result={**base,'model':model,'request_ordered_sha256':hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode()).hexdigest(),
            'request':payload,'attempts':history,'seconds_per_item_in_batch':sum(x['seconds'] for x in history),'batch_size':1,
            'timestamp':now()}
    if 'response' in raw:
        body=raw['response']
        if body['model']!=model: raise RuntimeError('Resolved model changed; refusing a mixed-version benchmark')
        try:
            answer=decode(qs,body)
            decoded=decode_typed(answer['probabilities'],row['questions'],mapping)
            # Preserve the explicit ordered tie rule from the decoder.
            for key in row['questions']:
                if key not in mapping: decoded['pred'][key]=answer['pred'][key]
            result.update(decoded);result['renormalized_questions']=answer['renormalized_questions']
        except (AssertionError,KeyError,TypeError,ValueError) as e:
            result.update(pred={},probabilities={},error='Invalid response: '+type(e).__name__)
    else: result.update(pred={},probabilities={},error=raw['error'])
    result['correct']=set(result['pred'])==set(row['gold']) and all(set(result['pred'][k])==set(g) for k,g in row['gold'].items())
    return result

def run(workers=4,limit=None):
    path=ROOT/'data/prepared/jev_live.jsonl';protocol=read_json(OUT/'protocol.json')
    assert digest(path)==protocol['fixture_sha256']
    rows=read_rows(path);dest=OUT/'predictions.jsonl'
    completed=read_rows(dest);byid={r['id']:r for r in completed};assert len(byid)==len(completed)
    for r in rows:
        if r['id'] in byid: assert byid[r['id']]['input_sha256']==r['input_sha256']
    todo=[r for r in rows if r['id'] not in byid]
    if limit:todo=todo[:limit]
    budget=Budget();client=Client(budget);model=protocol['model']['model']
    started=time.perf_counter();n=len(completed);errors=sum(bool(r.get('error')) for r in completed)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            iterator=iter(todo);pending={}
            def submit():
                row=next(iterator,None)
                if row is not None: pending[pool.submit(evaluate,row,client,model)]=row
            for _ in range(workers): submit()
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    row=pending.pop(future);result=future.result()
                    append(dest,result);n+=1;errors+=bool(result.get('error'))
                    write_json(OUT/'progress.json',{'completed':n,'expected':len(rows),'errors':errors,
                        'elapsed_seconds_this_run':time.perf_counter()-started,'last_suite':row['suite'],'workers':workers,
                        'status':'complete' if n==len(rows) else 'running',**budget.summary()})
                    if n%100==0 or n==len(rows):print(model,n,'/',len(rows),'errors',errors,'estimated USD',round(budget.summary()['estimated_cost_usd'],6),row['suite'],flush=True)
                    if result.get('error') in ['HTTP 401','HTTP 402','HTTP 403']:
                        raise RuntimeError('Authentication or credit response; stopped to preserve budget')
                    submit()
    finally:
        write_json(OUT/'spend.json',budget.summary());budget.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);a=p.parse_args()
    prepare() if a.action=='prepare' else run(a.workers,a.limit)
