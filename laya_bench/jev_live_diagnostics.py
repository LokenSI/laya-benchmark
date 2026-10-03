"""Outcome-independent repeat and option-order probes, outside accuracy suites."""
import argparse
from collections import defaultdict
import hashlib
import json
import psutil
from .common import ROOT,read_json,write_json,fingerprint,digest
from .alternatives_jev import read_rows
from .jev_api import OUT,Budget,Client,append,now
from .jev_live import evaluate

SUITES=['fresh/news','fresh/emotion','fresh/spam','fresh/phishing','industry/routing/en','industry/routing/nb',
        'industry/documents/en','industry/documents/nb','jev_verified/agnews','jev_verified/emotiondair',
        'jev_verified/banking77','jevbench_public/easy','jevbench_public/original','jevbench_public/hard']

def prepare():
    rows=read_rows(ROOT/'data/prepared/jev_live.jsonl');samples=[]
    for suite in SUITES:
        candidates=[r for r in rows if r['suite']==suite and r.get('split')!='development']
        candidates.sort(key=lambda r:hashlib.sha256(('repeat-probe-20261001/'+r['id']).encode()).hexdigest())
        for r in candidates[:8]:
            for variant in ['repeat1','repeat2','reversed_choices']:
                qs={k:dict(q) for k,q in r['questions'].items()}
                if variant=='reversed_choices':
                    if not any(q['type']=='choice' and len(q['criteria'])>1 for q in qs.values()):continue
                    for q in qs.values():
                        if q['type']=='choice':q['criteria']=dict(reversed(list(q['criteria'].items())))
                row={**r,'id':'diagnostic/'+variant+'/'+r['id'],'reference_id':r['id'],'variant':variant,
                     'questions':qs,'source_fixture':'jev_diagnostics'}
                row['input_sha256']=fingerprint({'state':row['state'],'questions':qs});samples.append(row)
    path=ROOT/'data/prepared/jev_diagnostics.jsonl';body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in samples)
    if path.exists():assert path.read_text(encoding='utf-8')==body
    else:path.write_text(body,encoding='utf-8')
    write_json(OUT/'diagnostics-protocol.json',{'created':now(),'fixture_sha256':digest(path),'n':len(samples),
                'selection':'Eight lowest salted-ID hashes per preselected task, independent of correctness or confidence. Two exact repeats and one reversed-choice request where applicable. Main experiment cases reused, not independent accuracy evidence.',
                'timing':'Robustness follow-up defined after the main run began. Sequential API calls; no inference hardware comparison.',
                'gold':'Unchanged; reversing option order preserves label descriptions and identifiers. Score-rubric order is never reversed.'})
    print('Frozen diagnostic requests:',len(samples),flush=True)

def run(after_pid=None):
    if after_pid:
        try:
            proc=psutil.Process(after_pid);assert 'laya_bench.jev_live' in ' '.join(proc.cmdline());proc.wait()
        except psutil.NoSuchProcess:pass
    source=ROOT/'data/prepared/jev_diagnostics.jsonl';assert digest(source)==read_json(OUT/'diagnostics-protocol.json')['fixture_sha256']
    rows=read_rows(source);dest=OUT/'diagnostic-predictions.jsonl';done={r['id'] for r in read_rows(dest)}
    # All paid attempts, including diagnostics and integration probes, share the cap.
    budget=Budget();client=Client(budget);model=read_json(OUT/'model-pin.json')['model']
    try:
        for r in rows:
            if r['id'] in done:continue
            value=evaluate(r,client,model);append(dest,value)
            if value.get('error') in ['HTTP 401','HTTP 402','HTTP 403']:raise RuntimeError('Account response; diagnostics stopped')
    finally:budget.close()
    report()

def report():
    original={r['id']:r for r in read_rows(OUT/'predictions.jsonl')}
    rows={r['id']:r for r in read_rows(ROOT/'data/prepared/jev_diagnostics.jsonl')}
    groups=defaultdict(list)
    for p in read_rows(OUT/'diagnostic-predictions.jsonl'):
        r=rows[p['id']];base=original.get(r['reference_id'])
        if base is None:continue
        valid=not p.get('error') and not base.get('error')
        diff=max((abs(v-base['probabilities'].get(k,{}).get(label,0)) for k,ps in p.get('probabilities',{}).items() for label,v in ps.items()),default=0)
        groups[r['variant']].append({'id':r['reference_id'],'valid':valid,'same_decision':valid and p['pred']==base['pred'],
                                      'max_probability_change':diff if valid else None,'seconds':p['seconds_per_item_in_batch']})
    result={'updated':now(),'expected':len(rows),'completed':sum(map(len,groups.values())),'by_variant':{}}
    for variant,items in groups.items():
        valid=[x for x in items if x['valid']]
        result['by_variant'][variant]={'n':len(items),'valid':len(valid),'same_decision':sum(x['same_decision'] for x in items),
                                     'changed_decisions':[x['id'] for x in items if x['valid'] and not x['same_decision']],
                                     'max_probability_change':max((x['max_probability_change'] for x in valid),default=None)}
    write_json(OUT/'diagnostics.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','report']);p.add_argument('--after-pid',type=int);a=p.parse_args()
    {'prepare':prepare,'run':lambda:run(a.after_pid),'report':report}[a.action]()
