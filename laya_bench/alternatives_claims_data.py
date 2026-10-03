"""Public exact claim fixtures, kept separate from the pre-frozen core."""
import json
import tomllib
from collections import Counter
from .common import ROOT,read_json,write_json,digest,fingerprint

R=ROOT/'.cache/alternatives_research'
def key(v):return str(v).lower() if isinstance(v,bool) else str(v)

def build():
    rows=[];sources={}
    def add(suite,id,state,questions,gold,**meta):
        row=dict(id=f'{suite}/{id}',suite=suite,state=state,questions=questions,gold=gold,**meta)
        row['input_sha256']=fingerprint({'state':state,'questions':questions});rows.append(row)
    for tier in ['easy','original','hard']:
        p=R/f'code--fstandhartinger--jevbench/datasets/public/{tier}.jsonl';sources[str(p.relative_to(ROOT))]=digest(p)
        for line in p.read_text(encoding='utf-8').splitlines():
            c=json.loads(line)
            assert c['expected'] is not None
            q=c['question'];expected=key(c['expected'])
            if q['type']=='noul':expected={'yes':'true','no':'false'}.get(expected,expected)
            add('jevbench_public/'+tier,c['id'],c['state'],{'decision':q},{'decision':[expected]},native_task=c,split='public_test')
    p=R/'code--jabr--classifier-benchmark/cases/v2.toml';sources[str(p.relative_to(ROOT))]=digest(p)
    with p.open('rb') as f:tasks=tomllib.load(f)['task']
    for t in tasks:
        for i,c in enumerate(t['cases']):
            add('jabr_v2/'+t['id'],str(i),c['state'],{'decision':{'type':t['type'],**t['question']}},{'decision':[key(c['expected'])]},split='public_synthetic_test')
    for suite in ['documents-v1','hard-v1','devtools-v1']:
        p=R/f'code--jaredpalmer--kev/evals/{suite}/test.jsonl';sources[str(p.relative_to(ROOT))]=digest(p)
        manifest=read_json(p.parent/'manifest.json')
        assert digest(p)==manifest['files']['test.jsonl']['sha256']
        for i,line in enumerate(p.read_text(encoding='utf-8').splitlines()):
            c=json.loads(line);gold={k:[key(q['label'])] for k,q in c['questions'].items()}
            qs={k:{field:v for field,v in q.items() if field in ['type','instructions','criteria']} for k,q in c['questions'].items()}
            add('kev_claim/'+suite,str(i),c['state'],qs,gold,split='publisher_locked_test',family=c.get('_meta',{}).get('group_id'))
    assert len(rows)==len({r['id'] for r in rows})
    out=ROOT/'data/prepared/alternatives_claims.jsonl'
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if out.exists() and out.read_text(encoding='utf-8')!=payload:raise RuntimeError('Frozen claim fixture differs')
    out.write_text(payload,encoding='utf-8')
    write_json(ROOT/'results/alternatives/alternatives_claims-protocol.json',{'fixture_sha256':digest(out),'n':len(rows),'heads':sum(len(r['gold']) for r in rows),'suites':dict(Counter(r['suite'] for r in rows)),'sources':sources,'selection':'All 231 public JevBench items, all 866 Jabr v2 cases, complete locked Kev documents/hard/devtools test files. No question labels, source tags or expected answers sent to models. Native claim scoring additionally reports per-question accuracy; JevBench uses lexical tie break and distribution validity checks. Public fixtures cannot reproduce sealed leaderboards.'})
    print('Frozen public claim records',len(rows),flush=True)

if __name__=='__main__':build()
