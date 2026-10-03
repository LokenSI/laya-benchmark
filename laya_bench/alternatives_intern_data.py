"""Hash-verified publisher claim bundle, separate from the frozen comparison."""
import json
from collections import Counter
import requests
from .common import ROOT,read_json,write_json,digest,fingerprint

REV='3572c8a68b5df5dafe02d0e093989ba8ec0183bc'
BASE=f'https://raw.githubusercontent.com/InternLM/Intern-Decision/{REV}/benchmarks/accuracy-v1/'

def build():
    root=ROOT/'.cache/alternatives_research/intern-accuracy-v1';root.mkdir(exist_ok=True)
    def download(name):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            response=requests.get(BASE+name,timeout=90);response.raise_for_status();p.write_bytes(response.content)
        return p
    manifest=read_json(download('manifest.json'));records=[]
    for suite,info in manifest['datasets'].items():
        path=download(info['path']);assert digest(path)==manifest['files'][info['path']]
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==info['rows']
        for row_number,c in enumerate(rows):
            assert not c.get('images')
            qs=c.get('questions') or {'decision':c['question']}
            targets=c.get('targets') or {'decision':{'label':c['expected']}}
            gold={}
            for k,t in targets.items():
                label=str(t['label'])
                if qs[k]['type']=='noul':label={'yes':'true','no':'false'}.get(label,label)
                gold[k]=[label]
            records.append({'id':f'intern_claim/{suite}/{row_number}/{c["id"]}','source_id':c['id'],'suite':'intern_claim/'+suite,'state':c['state'],'questions':qs,'gold':gold,'native_targets':targets,'split':'publisher_public_test','input_sha256':fingerprint({'state':c['state'],'questions':qs})})
    assert len(records)==manifest['rows']
    assert sum(len(r['gold']) for r in records)==manifest['decisions']
    for name,rows in [('intern_claims',records),('alternatives_typed',[r for r in records if r['suite'].endswith('/typed_decisions-test')])]:
        path=ROOT/f'data/prepared/{name}.jsonl';payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
        if path.exists():assert path.read_text(encoding='utf-8')==payload
        else:path.write_text(payload,encoding='utf-8')
        write_json(ROOT/f'results/alternatives/{name}-protocol.json',{'fixture_sha256':digest(path),'n':len(rows),'heads':sum(len(r['gold']) for r in rows),'suites':dict(Counter(r['suite'] for r in rows)),'source':BASE,'revision':REV,'manifest_sha256':digest(root/'manifest.json'),'scoring':'All public test records, per-question hard-label accuracy; preserve supplied Noul criteria. Targets never enter inference requests. Typed labels are teacher agreement, not independently verified human truth. Other publishers may use different preprocessing.'})
        print(name,len(rows),flush=True)

if __name__=='__main__':build()
