"""Compare only compatible published claims; preserve unverifiable claims."""
import json
import html
from collections import Counter
from .common import ROOT,read_json,write_json,digest
from .alternatives_jev import read_rows

def kev_claim_score(name,suite):
    """Publisher's question denominator; duplicated source IDs removed only here."""
    source=ROOT/f'.cache/alternatives_research/code--jaredpalmer--kev/evals/{suite}/test.jsonl'
    native=read_rows(source);counts=Counter(r['_meta']['id'] for r in native)
    byid={r['id']:r for r in read_rows(ROOT/'data/prepared/alternatives_claims.jsonl')}
    pred={r['id']:r for r in read_rows(ROOT/f'results/alternatives/alternatives_claims/{name}/predictions.jsonl')}
    selected=[];excluded=[]
    for i,r in enumerate(native):
        key=f'kev_claim/{suite}/{i}'
        assert byid[key]['state']==r['state']
        if suite=='devtools-v1' and counts[r['_meta']['id']]>1:
            excluded.append({'row':i,'source_id':r['_meta']['id'],'heads':len(r['questions'])});continue
        selected.append(byid[key])
    heads=sum(len(r['gold']) for r in selected)
    complete=all(r['id'] in pred for r in selected)
    correct=None
    if complete:
        assert all(pred[r['id']]['input_sha256']==r['input_sha256'] for r in selected)
        correct=sum(not pred[r['id']].get('error') and not pred[r['id']].get('abstained',{}).get(k) and set(pred[r['id']]['pred'].get(k,[]))==set(g) for r in selected for k,g in r['gold'].items())
    return {'complete':complete,'heads':heads,'correct':correct,'accuracy':correct/heads if complete else None,
            'source_sha256':digest(source),'excluded_duplicate_source_ids':excluded}

def build(report):
    out=ROOT/'results/alternatives';audit=[]
    def card(name):
        meta=report['models'][name]['checkpoint']
        p=ROOT/'.cache/alternatives_research'/meta['repo'].replace('/','--')/'README.md'
        assert digest(p)==meta['model_card_sha256']
        return {'url':f'https://huggingface.co/{meta["repo"]}/blob/{meta["revision"]}/README.md','sha256':digest(p),'model_revision':meta['revision']}
    for name,targets in [('decider-4b',{'easy':1.,'original':.986,'hard':.649}),('decider-2b',{'easy':1.,'original':.889,'hard':.577})]:
        source=card(name)
        for tier,published in targets.items():
            actual=report['models'][name]['fixtures']['alternatives_claims']['native_jev'].get('jevbench_public/'+tier)
            expected={'easy':48,'original':72,'hard':111}[tier]
            complete=bool(actual and actual['n']==expected)
            gap=actual['accuracy']-published if complete else None
            verdict='pending' if not complete else 'matches_published_rounding' if abs(gap)<=.0005 else 'near_reproduction_within_2pp' if abs(gap)<=.02 else 'not_reproduced_in_this_runtime'
            audit.append({'model':name,'claim':'JevBench public '+tier,'published_accuracy':published,'expected_cases':expected,
                          'measured_accuracy':actual['accuracy'] if complete else None,'difference_pp':gap*100 if complete else None,
                          'verdict':verdict,'source':source,'scope':'Same verified public cases and native scorer; bf16 Windows/RTX runtime, graphs disabled. The 2pp band is an engineering check, not a significance test.'})
    claims=read_json(ROOT/'data/prepared/claims.json')
    for name,key in [('laya','english'),('laya-multilingual','multilingual')]:
        for task,suite in claims['suites'].items():
            if not suite.get('published'):continue
            actual=report['models'][name]['fixtures']['alternatives']['suites'].get('laya_claim/'+task)
            complete=bool(actual and actual['complete']);published=suite['published'][key]
            gap=actual['accuracy']-published if complete else None
            audit.append({'model':name,'claim':'Publisher '+task,'published_accuracy':published,
                          'measured_accuracy':actual['accuracy'] if complete else None,'difference_pp':gap*100 if complete else None,
                          'verdict':'pending' if not complete else 'near_reproduction_within_2pp' if abs(gap)<=.02 else 'outside_2pp_repeatability_band',
                          'source':claims['manifest']['publisher_files']['BENCHMARKS.md'],
                          'scope':'Reconstructed publisher questions and sampling. Publisher did not pin dataset revisions; our source revisions are fixed. SDK and device differ. Task training exposure is explicitly reported.'})
    for name in ['gliner-decide','gliner-decide-1b','gliner-decide-multi']:
        audit.append({'model':name,'claim':'Fast Decisions advertised 5100-example held-out average','verdict':'not_independently_verifiable_with_public_fixture',
                      'source':card(name),'scope':'The accessible fixture has 1700 development examples. Its results cannot verify or refute a claim on 5100 different held-out examples.'})
    for name,targets in [('kev-08b',{'documents-v1':.851,'hard-v1':.665,'devtools-v1':.637}),('kev-4b',{'hard-v1':.803,'devtools-v1':.756})]:
        for suite,published in targets.items():
            measured=kev_claim_score(name,suite);gap=measured['accuracy']-published if measured['complete'] else None
            audit.append({'model':name,'claim':suite+' locked test / question accuracy','published_accuracy':published,
                          'measured_accuracy':measured['accuracy'],'difference_pp':gap*100 if gap is not None else None,
                          'verdict':'pending' if gap is None else 'matches_published_rounding' if abs(gap)<=.0005 else 'near_reproduction_within_2pp' if abs(gap)<=.02 else 'not_reproduced_in_this_runtime',
                          'source':card(name),'details':measured,
                          'scope':'Same pinned public test and question metric; bf16 local serving instead of publisher fp32 claim evaluation. Devtools uses the publisher amendment: remove both rows sharing the duplicated CodeReviewer source ID (1071 questions). The common cross-model benchmark retains all 1073 questions. Hard-v1 holds out templates of trained skills; devtools labels are public proxies; documents use the same CFPB source and question templates as training.'})
    audit.append({'model':'kev-4b','claim':'Previous round-8 documents-v1 locked test 90.4%',
                  'verdict':'different_checkpoint_version','source':card('kev-4b'),
                  'scope':'The tested checkpoint is the later round-10 skills release. Its model card also documents older results. A round-10 score cannot verify or refute the older round-8 90.4% claim.'})
    audit.append({'model':'von','claim':'Jabr v2 macro accuracy 72.0% on 869 cases','verdict':'fixture_count_mismatch',
                  'source':card('von'),'scope':'Our pinned complete TOML contains 866 cases. Report it as a transfer result, not an exact replication. Von reports fitting its confidence map on the public JevBench cases and emotion in training.'})
    julia=report.get('native_claim_checks',{}).get('julia_typed')
    if julia:audit.append({'model':'julia','claim':'Native typed-decisions 2000 heads','verdict':'near_reproduction_within_2pp',**julia})
    audit.append({'model':'clm-int8','claim':'CLM DeepSWE best-of-N 31/38 (81.6%)','verdict':'exact_downstream_reproduction_from_published_embeddings',
                  'scope':'Earlier local run used the authors\' evaluator and public precomputed embeddings. This confirms the downstream evaluation, not full encoder/runtime equivalence. INT8 local classification is a separate test.',
                  'evidence':'../clm/FINDINGS_CLM.md'})
    payload={'updated':report['updated'],'claims':audit,'principle':'Only matching data, metric, model version and protocol support a reproduction verdict. A failed portability test or a different private fixture is not a refutation.'}
    write_json(out/'claim_audit.json',payload)
    render(payload)
    return payload

def render(payload):
    esc=lambda v:html.escape(str(v));rows=[]
    for c in payload['claims']:
        source=c.get('source',{});url=source.get('url') if isinstance(source,dict) else source if isinstance(source,str) else None
        measured=c.get('measured_accuracy',c.get('accuracy'));published=c.get('published_accuracy')
        values='—' if published is None else f'{published:.2%} published<br>'+('Pending' if measured is None else f'{measured:.2%} measured')
        link=f'<a href="{esc(url)}">Primary source</a>' if url else ''
        rows.append(f'<tr><th>{esc(c["model"])}<small>{esc(c["claim"])}</small></th><td>{values}</td><td>{esc(c["verdict"].replace("_"," "))}</td><td>{esc(c.get("scope",c.get("method","")))}<br>{link}</td></tr>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Decision model claim audit</title><style>body{font:16px/1.6 system-ui,sans-serif;background:#f5f7fa;color:#152b40;margin:0}main{max-width:1250px;margin:auto;padding:40px 28px}h1{font-size:36px}table{background:white;border-collapse:collapse;font-size:14px;width:100%}td,th{padding:16px;border-bottom:1px solid #dbe3ea;text-align:left;vertical-align:top}small{display:block;color:#607184}a{color:#166a79}.note{background:#e9eff4;padding:20px;border-left:4px solid #327f80}</style><main><h1>Which published claims hold up?</h1>'''
    page+=f'<p class="note">{esc(payload["principle"])} Updated {esc(payload["updated"])}. The ±2 percentage-point band is a practical replication tolerance, not a statistical equivalence test.</p>'
    page+='<p><a href="../jev_live/report.html">Live Jev comparison and business cases</a> · <a href="claim_audit.json">Full audit data and source hashes</a></p><table><thead><tr><th>Model and claim</th><th>Accuracy</th><th>Finding</th><th>Scope of the evidence</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    page+='<h2>What the Jev marketing statements establish</h2><p>The <a href="https://typesafe.ai/">TypeSafe website</a> advertises speed and cost multiples, calibrated confidence and zero hallucinations. This experiment measures accuracy, calibration and billable usage on fixed decision tasks. Its network timings and local GPU timings do not reproduce those speed or cost comparison protocols. Restricted output choices prevent invented labels; they do not prevent incorrect choices. “Zero hallucinations” must not be interpreted as zero decision errors.</p><p>Use the measured task-level errors and development-selected review thresholds in the <a href="../jev_live/report.html">live report</a>. Confidence is evaluated on its own merits; a syntactically valid probability is not proof of calibration.</p></main></html>'
    (ROOT/'results/alternatives/claim_audit.html').write_text(page,encoding='utf-8')

if __name__=='__main__':build(read_json(ROOT/'results/alternatives/comparison.json'))
