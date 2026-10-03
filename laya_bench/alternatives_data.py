"""Freeze the October comparison before running any new model on it."""
import json
from collections import Counter
from .common import ROOT, read_json, write_json, digest, fingerprint
from .questions import question
from .industry import question as industry_question
from .jev_native import short_name, INSTRUCTIONS

OUT=ROOT/'data/prepared/alternatives.jsonl'

def build():
    records=[]
    sources={}
    def add(suite,id,state,questions,gold,**meta):
        row=dict(id=f'{suite}/{id}',suite=suite,state=state,questions=questions,gold=gold,**meta)
        row['input_sha256']=fingerprint({'state':state,'questions':questions})
        records.append(row)
    p=ROOT/'data/prepared/industry.json';sources[str(p.relative_to(ROOT))]=digest(p)
    for c in read_json(p)['cases']:
        if c['split']!='test':continue
        q=industry_question(c['task'],c['language'],'plain')
        suite=f'industry/{c["task"]}/{c["language"]}'
        add(suite,c['id'],c['text'],q,{'decision':[c['gold']]},family=c['family'],sector=c['sector'],split='synthetic_holdout')
        rev={k:{**v,'criteria':dict(reversed(list(v['criteria'].items())))} for k,v in q.items()}
        add(suite+'/reversed',c['id'],c['text'],rev,{'decision':[c['gold']]},family=c['family'],split='diagnostic')
    p=ROOT/'data/prepared/jev_manifest.jsonl';sources[str(p.relative_to(ROOT))]=digest(p)
    for line in p.read_text(encoding='utf-8').splitlines():
        c=json.loads(line); names=[short_name(h) for h in c['labels']]
        assert len(names)==len(set(names))
        q={'decision':{'type':'choice','instructions':INSTRUCTIONS[c['dataset']], 'criteria':dict(zip(names,names))}}
        add('jev_native/'+c['dataset'],c['example_id'],{'text':c['text']},q,{'decision':[names[c['target_index']]]},split='public_test')
    meta=read_json(ROOT/'results/alternatives/data/fastino--fast-decisions.json')
    for p in sorted(__import__('pathlib').Path(meta['path']).glob('*.jsonl')):
        sources[f'fast_decisions/{p.name}']=digest(p)
        for i,line in enumerate(p.read_text(encoding='utf-8').splitlines()):
            c=json.loads(line);qs={};gold={}
            for h in c['output']['classifications']:
                qs[h['task']]={'type':'multilabel' if h['multi_label'] else 'choice','instructions':h['task'], 'criteria':{k:k for k in h['labels']}}
                gold[h['task']]=h['true_label']
            add('fast_decisions_dev/'+p.stem,str(i),c['input'],qs,gold,split='publisher_development')
    p=ROOT/'data/prepared/claims.json';sources[str(p.relative_to(ROOT))]=digest(p)
    for name,suite in read_json(p)['suites'].items():
        if suite['phase']!='replication':continue
        for c in suite['cases']:
            add('laya_claim/'+name,c['id'],c['state'],{'decision':c['question']},{'decision':[c['gold']]},split='publisher_replication')
    p=ROOT/'data/prepared/full.json';sources[str(p.relative_to(ROOT))]=digest(p)
    for name,suite in read_json(p)['suites'].items():
        for c in suite['splits']['test']:
            add(name,c['id'],c['text'],question(suite),{'decision':[c['label']]},split='public_test',family=c['group'])
    assert len({r['id'] for r in records})==len(records)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)
    if OUT.exists() and OUT.read_text(encoding='utf-8')!=payload:
        raise RuntimeError('Frozen fixture differs; create a new version rather than overwriting it')
    OUT.write_text(payload,encoding='utf-8')
    write_json(ROOT/'results/alternatives/protocol.json',dict(
        version=1,fixture_sha256=digest(OUT),n=len(records),suites=dict(Counter(r['suite'] for r in records)),sources=sources,
        selection='All held-out examples in existing full/public fixtures, all 1700 Fast Decisions development rows. Full candidate lists; no gold-informed shortlist.',
        prompts='Existing task-native prompts, industry plain prompt fixed for every model before inference. Reversed industry options are a diagnostic, never used to select a prompt.',
        metrics='Exact match of all heads per record, head accuracy, per-suite macro F1 for single fixed heads, Wilson 95% intervals. Failures remain in denominators. No single pooled leaderboard across unrelated tasks.',
        multi_label='Native GLiNER multi-label head. Typed choice-only models receive one boolean question per candidate. Select p(true)>=0.5; no gold-informed top-k.',
        timing='Synchronized GPU batch elapsed time is throughput, not interactive latency. Separate warmed batch-one timings. Record precision, model revision, runtime and peak allocated VRAM.',
        claim_replication='Only exact public fixtures and matching metric/protocol support a replication verdict. Private held-out results remain unverified. Different precision/runtime is explicitly qualified.',
        limits='Industry labels are assistant-authored, not SME-validated. Public vendor development data may be training/calibration-exposed. Norwegian includes Bokmal intent and NoReC review sentiment, not general Norwegian competence.'
    ))
    print(f'Frozen {len(records)} cases at {OUT}',flush=True)

if __name__=='__main__':build()
