"""Verified Jev references, exact pilot requests, and paired public-case comparisons."""
import argparse
from collections import Counter,defaultdict
import hashlib
import html
import json
import sys
from .common import ROOT,read_json,write_json,digest,fingerprint

OUT=ROOT/'results/alternatives'
CACHE=ROOT/'.cache/alternatives_research/jev-verified'
PUBLIC=ROOT/'.cache/alternatives_research/code--fstandhartinger--jevbench'
PILOT=ROOT/'.cache/jev'
PUBLIC_REV='bb05a335bc809e61b20c0f745d25499a82b326fc'
PILOT_REV='0d610cc53e79bcbec691312b0c4adb4a0e371642'
PUBLIC_URL=f'https://github.com/fstandhartinger/jevbench/blob/{PUBLIC_REV}/results/v1.2/jevbench-v1.2-per-task.json'
PILOT_URL=f'https://github.com/AbdelStark/jev-benchmarks/blob/{PILOT_REV}/results/reports/btzsc-pilot-v1.md'

def read_rows(path):
    if not path.exists():return []
    text=path.read_text(encoding='utf-8');parts=text.splitlines()
    if text and not text.endswith('\n'):parts=parts[:-1]
    return [json.loads(line) for line in parts]

def verify_blob(path,tree,relative):
    entry=next(e for e in tree['tree'] if e['path']==relative)
    body=path.read_bytes()
    assert hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==entry['sha'],relative
    return digest(path)

def prepare():
    import yaml
    tree=read_json(PUBLIC/'tree.json');assert tree['sha']==PUBLIC_REV
    hashes={}
    for relative in ['results/v1.2/jevbench-v1.2-per-task.json','datasets/manifest.json']:
        hashes[relative]=verify_blob(CACHE/relative,tree,relative)
    manifest=read_json(CACHE/'datasets/manifest.json')
    public=[]
    for name in ['easy','original','hard']:
        path=PUBLIC/f'datasets/public/{name}.jsonl'
        expected=next(s for s in manifest['splits'] if s['name']==name)
        assert digest(path)==expected['sha256']
        cases=read_rows(path);assert len(cases)==expected['n']
        public.extend(cases);hashes[f'datasets/public/{name}.jsonl']=digest(path)
    reference=read_json(CACHE/'results/v1.2/jevbench-v1.2-per-task.json')
    jev=reference['systems']['jev-1.13.0']['public_tasks']
    assert len(public)==231==len(jev)==len({r['id'] for r in public})
    assert set(jev)=={r['id'] for r in public}=={r['id'] for r in reference['tasks']}
    assert all(v[0] in ['c','w','f'] for v in jev.values())
    # Verify already-running cases against the publisher's state AND option order.
    running={r['native_task']['id']:r for r in read_rows(ROOT/'data/prepared/alternatives_claims.jsonl') if 'native_task' in r}
    for case in public:
        ours=running[case['id']]
        for field in ['state','question']:
            actual=ours['state'] if field=='state' else ours['questions']['decision']
            assert json.dumps(actual,ensure_ascii=False)==json.dumps(case[field],ensure_ascii=False)
        assert ours['native_task']==case
    ptree=read_json(PILOT/'tree.json');assert ptree['sha']==PILOT_REV
    for relative in ['configs/pilot-v1.yaml','results/reports/btzsc-pilot-v1.json','src/jev_benchmarks/adapters/jev.py','src/jev_benchmarks/metrics.py','src/jev_benchmarks/runner.py']:
        hashes['pilot/'+relative]=verify_blob(PILOT/relative,ptree,relative)
    published=read_json(PILOT/'results/reports/btzsc-pilot-v1.json')
    path=ROOT/'data/prepared/jev_manifest.jsonl'
    assert digest(path)==published['artifacts']['manifest_sha256']
    config=yaml.safe_load((PILOT/'configs/pilot-v1.yaml').read_text())
    question=config['models']['jev']['question'];records=[]
    for case in read_rows(path):
        assert hashlib.sha256(case['text'].encode()).hexdigest()==case['text_sha256']
        keys=[f'label_{i:03d}' for i in range(len(case['labels']))]
        state={'text':case['text']}
        qs={'label':{'type':'choice','instructions':question,'criteria':dict(zip(keys,case['labels']))}}
        record={'id':'jev_verified/'+case['example_id'],'source_id':case['example_id'],'suite':'jev_verified/'+case['dataset'],'state':state,'questions':qs,'gold':{'label':[keys[case['target_index']]]},'split':'published_jev_pilot','input_sha256':fingerprint({'state':state,'questions':qs})}
        record['ordered_input_sha256']=hashlib.sha256(json.dumps({'state':state,'questions':qs},ensure_ascii=False).encode()).hexdigest()
        records.append(record)
    assert len(records)==300==len({r['id'] for r in records})
    dest=ROOT/'data/prepared/jev_verified.jsonl';payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)
    if dest.exists():assert dest.read_text(encoding='utf-8')==payload
    else:dest.write_text(payload,encoding='utf-8')
    write_json(OUT/'jev_verified-protocol.json',{'fixture_sha256':digest(dest),'n':300,'heads':300,'suites':dict(Counter(r['suite'] for r in records)),'source_manifest_sha256':digest(path),'source':PILOT_URL,'revision':PILOT_REV,'reference_model':published['models']['jev'],'selection':'Exact published 300-example manifest: 100 each AG News, DAIR Emotion and Banking77/BTZSC. Same state, instructions, option descriptions, identifiers and order; all 72 banking options retained. No prompt selection or shortlist.','scoring':'Publisher argmax in option order and probability validation/normalization; publisher metric code for accuracy, full-label macro F1, Brier, NLL and ECE. Failed requests remain incorrect.','limits':'Historical published Jev 1.13.0 API run, not a fresh call or current Jev claim. No public per-case pilot outputs, so no paired pilot significance test. Timing is not a hardware-normalized comparison. Public data exposure unknown. Verified means source integrity and matching inputs, not independent relabelling.'})
    counts={name:sum(jev[r['id']][0]=='c' for r in public if r['id'].startswith(prefix)) for name,prefix in [('easy','easy-'),('original','original-'),('hard','hard-')]}
    verification={'reference_model':'jev-1.13.0','public_revision':PUBLIC_REV,'public_artifact_revision':reference['revision'],'public_source':PUBLIC_URL,'public_case_count':231,'public_jev_correct':sum(v[0]=='c' for v in jev.values()),'public_jev_correct_by_tier':counts,'public_cases_match_existing_fixture':True,'pilot_revision':PILOT_REV,'pilot_source':PILOT_URL,'pilot_case_count':300,'pilot_manifest_matches_published_hash':True,'pilot_published_accuracy':{name:v['accuracy'] for name,v in published['results']['jev'].items()},'verified_source_sha256':hashes,'limitations':['Reference outcomes are independently published observations, not a new Jev API run.','Public and sealed JevBench scores must not be conflated.','Public outcomes permit paired correctness comparisons, but not paired calibration without Jev probability vectors.','Pilot per-case predictions are not published; aggregate differences only.']}
    write_json(OUT/'jev_verification.json',verification)
    print('Verified 231 public Jev outcomes; froze 300 exact pilot requests.',flush=True)
    return verification

def paired_summary(cases,local,reference):
    """Paired bootstrap, keeping paraphrase groups together and tiers stratified."""
    import numpy as np
    assert len(cases)==len(local)==len(reference)>0
    local=np.asarray(local,dtype=int);reference=np.asarray(reference,dtype=int)
    grouped=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(cases):grouped[r['suite']][r['native_task'].get('group') or r['id']].append(i)
    rng=np.random.default_rng(20261001);num=np.zeros(2000);den=np.zeros(2000)
    for groups in grouped.values():
        chunks=list(groups.values());sums=np.array([(local-reference)[ix].sum() for ix in chunks]);sizes=np.array([len(ix) for ix in chunks])
        draws=rng.integers(0,len(chunks),size=(2000,len(chunks)));num+=sums[draws].sum(1);den+=sizes[draws].sum(1)
    return {'n':len(cases),'local_correct':int(local.sum()),'jev_correct':int(reference.sum()),'local_accuracy':float(local.mean()),'jev_accuracy':float(reference.mean()),'difference_pp':float((local-reference).mean()*100),'paired_cluster_bootstrap95_pp':list(map(float,np.quantile(100*num/den,[.025,.975]))),'local_only_correct':int(((local==1)&(reference==0)).sum()),'jev_only_correct':int(((local==0)&(reference==1)).sum()),'both_correct':int(((local==1)&(reference==1)).sum()),'both_wrong':int(((local==0)&(reference==0)).sum()),'independent_groups':sum(len(g) for g in grouped.values()),'interval_note':'Descriptive paired bootstrap, 2000 resamples, stratified by public tier and clustered by provided paraphrase group; not adjusted for multiple model comparisons.'}

def pilot_metrics(cases,predictions):
    sys.path.insert(0,str(PILOT/'src'))
    from jev_benchmarks.models import Prediction
    from jev_benchmarks.metrics import score_predictions
    # This import uses only the publisher's lightweight validation, not an API call.
    from jev_benchmarks.runner import _validate_prediction
    rows=[]
    for case,pred in zip(cases,predictions):
        keys=list(case['questions']['label']['criteria']);probs=pred.get('probabilities',{}).get('label',{})
        error=pred.get('error') or ('Native abstention' if pred.get('abstained',{}).get('label') else None);values=tuple(probs[k] for k in keys) if set(probs)==set(keys) else ()
        if not values:error=error or 'Missing option probabilities'
        index=max(range(len(values)),key=values.__getitem__) if values and not error else -1
        row=Prediction('btzsc-pilot-v1','local',pred['model'],pred['model'],case['suite'].split('/')[-1],case['source_id'],keys.index(case['gold']['label'][0]),index,tuple(keys),values,0.,error=error)
        if not error:
            try:row=_validate_prediction(row)
            except ValueError as e:
                from dataclasses import replace
                row=replace(row,predicted_index=-1,error=str(e))
        rows.append(row)
    result=score_predictions(rows,ece_bins=10,error_budget=.05)
    for k in list(result):
        if k.startswith('latency_') or k=='input_tokens_total':result.pop(k)
    result['accuracy']=sum(r.predicted_index==r.target_index for r in rows)/len(rows)
    result['coverage_at_error_budget_note']='Threshold fitted on this test slice: retrospective diagnostic, not an automation guarantee.'
    return result

def compare(models):
    """Called by the main reporter; reuse only predictions with matching hashes."""
    sys.path.insert(0,str(PUBLIC))
    from jevbench.tasks import Task
    from jevbench.scoring import score_task
    verification=read_json(OUT/'jev_verification.json')
    reference=read_json(CACHE/'results/v1.2/jevbench-v1.2-per-task.json')['systems']['jev-1.13.0']['public_tasks']
    public=[r for r in read_rows(ROOT/'data/prepared/alternatives_claims.jsonl') if 'native_task' in r]
    pilot=read_rows(ROOT/'data/prepared/jev_verified.jsonl')
    output={'verification':verification,'models':{},'public_case_count':len(public),'pilot_case_count':len(pilot)}
    pairs=[]
    for name,model in models.items():
        byid={r['id']:r for r in read_rows(OUT/f'alternatives_claims/{name}/predictions.jsonl')}
        matched=[r for r in public if r['id'] in byid];entry={'label':model['label'],'public_attempted':len(matched),'public_complete':len(matched)==len(public),'public':{},'pilot':{}}
        scored={}
        for row in matched:
            pred=byid[row['id']];assert pred['input_sha256']==row['input_sha256']
            probs={} if pred.get('abstained',{}).get('decision') else pred.get('probabilities',{}).get('decision',{})
            if row['questions']['decision']['type']=='noul':probs={{'true':'yes','false':'no'}[k]:v for k,v in probs.items()}
            score=score_task(probs,Task.from_dict(row['native_task']));scored[row['id']]=score
            pairs.append({'model':name,'case_id':row['native_task']['id'],'tier':row['suite'].split('/')[-1],'input_sha256':row['input_sha256'],'local_correct':bool(score['correct']),'jev_correct':reference[row['native_task']['id']][0]=='c','local_valid':score['valid'],'local_predicted':score.get('predicted'),'expected':row['native_task']['expected']})
        for tier in ['easy','original','hard','all']:
            cases=[r for r in public if tier=='all' or r['suite'].endswith('/'+tier)]
            if not all(r['id'] in scored for r in cases):continue
            entry['public'][tier]=paired_summary(cases,[bool(scored[r['id']]['correct']) for r in cases],[reference[r['native_task']['id']][0]=='c' for r in cases])
            entry['public'][tier]['local_invalid']=sum(not scored[r['id']]['valid'] for r in cases)
        strict={r['id']:r for r in read_rows(OUT/f'jev_verified/{name}/predictions.jsonl')}
        entry['pilot_attempted']=len(strict)
        for dataset in ['agnews','emotiondair','banking77']:
            cases=[r for r in pilot if r['suite'].endswith('/'+dataset)]
            if not all(r['id'] in strict for r in cases):continue
            for r in cases:assert strict[r['id']]['input_sha256']==r['input_sha256']
            metrics=pilot_metrics(cases,[strict[r['id']] for r in cases]);metrics['published_jev_accuracy']=verification['pilot_published_accuracy'][dataset];metrics['difference_pp']=(metrics['accuracy']-metrics['published_jev_accuracy'])*100
            entry['pilot'][dataset]=metrics
        output['models'][name]=entry
    write_json(OUT/'jev_comparison.json',output)
    (OUT/'jev_paired_cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs),encoding='utf-8')
    render(output)
    return output

def render(report):
    esc=html.escape;sections=[]
    for tier,title in [('easy','Easy · 48 cases'),('original','Original · 72 cases'),('hard','Hard · 111 cases'),('all','All public cases · 231')]:
        rows=[]
        for m in report['models'].values():
            s=m['public'].get(tier)
            if s:
                low,high=s['paired_cluster_bootstrap95_pp']
                rows.append(f'<tr><th>{esc(m["label"])}</th><td>{s["local_correct"]}/{s["n"]} · {s["local_accuracy"]:.1%}</td><td>{s["jev_correct"]}/{s["n"]} · {s["jev_accuracy"]:.1%}</td><td>{s["difference_pp"]:+.1f} pp<small>95% interval {low:+.1f} to {high:+.1f}</small></td><td>{s["local_only_correct"]} / {s["jev_only_correct"]}</td><td>{s["local_invalid"]}</td></tr>')
        sections.append(f'<section><h2>{title}</h2><table><tr><th>Local model</th><th>Local accuracy</th><th>Published Jev</th><th>Local minus Jev</th><th>Local-only / Jev-only correct</th><th>Invalid local outputs</th></tr>{"".join(rows)}</table></section>')
    pilot_rows=[]
    for m in report['models'].values():
        for task,s in m['pilot'].items():pilot_rows.append(f'<tr><th>{esc(m["label"])}</th><td>{esc(task)}</td><td>{s["accuracy"]:.1%}</td><td>{s["published_jev_accuracy"]:.1%}</td><td>{s["difference_pp"]:+.1f} pp</td><td>{s["valid"]}/{s["n"]}</td></tr>')
    pending=sum(m['pilot_attempted']<300 for m in report['models'].values())
    v=report['verification']
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Verified Jev comparison</title><style>body{{font:15px system-ui,sans-serif;background:#f5f7fa;color:#172c43;margin:0}}main{{max-width:1250px;margin:auto;padding:40px 30px}}h1{{font-size:36px}}p{{line-height:1.65;max-width:1000px}}section{{background:white;padding:22px;margin:24px 0;border:1px solid #dce3e9}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:12px 8px;border-bottom:1px solid #e3e8ed;text-align:left}}tr:first-child{{font-size:12px;color:#516579}}small{{display:block;color:#607488}}.note{{padding:18px;background:#e5edf4}}a{{color:#176879}}</style><main><h1>Local models against verified Jev cases</h1><p class="note"><strong>INTERIM · Jev 1.13.0 is a historical published API reference.</strong><br>231 public items match the publisher's file hashes, questions, option order and per-case outcome IDs. Local partial tiers are excluded from the tables. This is not the official JevBench ranking, which also includes sealed tasks.</p><p>We verified source integrity and input equality. We have not independently adjudicated every answer key. Public benchmark exposure remains possible. Accuracy differences and their paired intervals are descriptive; many model comparisons increase the chance of a misleading apparent winner.</p>{''.join(sections)}<section><h2>Exact 300-case pilot</h2><p>100 news, 100 emotion and 100 banking-intent cases. Identical published manifest, original instructions, opaque label IDs, full option descriptions and order. The banking slice has 72 options. {pending} local model runs remain incomplete. Natural-label results from the main comparison are a separate prompt diagnostic.</p><table><tr><th>Local model</th><th>Task</th><th>Local accuracy</th><th>Published Jev</th><th>Difference</th><th>Valid requests</th></tr>{''.join(pilot_rows)}</table><p>Published Jev pilot reference: news 91%, emotion 48%, banking 87%. Its per-case predictions are not public, so this panel shows aggregate differences without a paired significance claim. The publisher's scoring code is reused; hosted and local timings are not ranked together.</p></section><section><h2>Evidence</h2><p><a href="{v['public_source']}">Pinned public Jev per-case outcomes</a> · <a href="{v['pilot_source']}">Pinned 300-case pilot report</a> · <a href="jev_verification.json">Local source verification</a> · <a href="jev_paired_cases.jsonl">Case-by-case comparison</a> · <a href="jev_comparison.json">Full metrics</a></p><p>Paired intervals resample the same local/Jev cases together, stratify by tier, and keep supplied paraphrase groups together. Missing or invalid local answers count as incorrect. Jev public reference: {v['public_jev_correct']}/231 correct. Public correctness outcomes do not provide the Jev probability vectors required for a paired calibration comparison.</p></section></main></html>'''
    (OUT/'jev_comparison.html').write_text(page,encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare']);args=parser.parse_args();prepare()
