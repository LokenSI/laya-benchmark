"""Business-readable report from complete matched suites; no cherry-picked rows."""
import argparse
from collections import Counter,defaultdict
import html
import json
import math
import sys
from .common import ROOT,read_json,write_json,digest
from .alternatives_jev import read_rows,paired_summary
from .alternatives_report import NAMES
from .metrics import wilson
from .jev_api import OUT,now

JEV='jev-1.13.0'
NAMES={**NAMES,JEV:'Jev 1.13.0 · live API'}
PRIORITY=['laya','laya-multilingual','decider-08b','decider-2b','decider-4b','von','gliner-decide']
FOCUS=['fresh/news','fresh/emotion','fresh/spam','fresh/phishing',
       'fresh_business/routing/en','fresh_business/routing/nb','fresh_business/documents/en','fresh_business/documents/nb',
       'industry/routing/en','industry/routing/nb','industry/documents/en','industry/documents/nb',
       'laya_claim/news.replication','laya_claim/emotion.replication','laya_claim/spam.replication','laya_claim/phishing.replication',
       'jev_verified/agnews','jev_verified/emotiondair','jev_verified/banking77',
       'jevbench_public/easy','jevbench_public/original','jevbench_public/hard',
       'kev_claim/documents-v1','kev_claim/hard-v1','kev_claim/devtools-v1','massive_en','massive_nb','norec_no']
TITLES={'fresh/news':'News categories · fresh public test','fresh/emotion':'Emotion categories · fresh public test',
        'fresh/spam':'Spam screening · fresh public test','fresh/phishing':'Phishing / scam screening · fresh public test',
        'industry/routing/en':'Ticket routing · English','industry/routing/nb':'Ticket routing · Norwegian',
        'industry/documents/en':'Engineering documents · English','industry/documents/nb':'Engineering documents · Norwegian',
        'massive_en':'18 intent scenarios · English','massive_nb':'18 intent scenarios · Norwegian Bokmål',
        'norec_no':'Norwegian review sentiment','kev_claim/documents-v1':'Public financial-document decisions',
        'kev_claim/hard-v1':'Templated decision skills','kev_claim/devtools-v1':'Public developer-tooling decisions'}

def score(row,pred):
    assert pred['input_sha256']==row['input_sha256']
    if pred.get('error'):return {'correct':False,'valid':False,'heads':[], 'native_changed':False}
    heads=[];native_changed=False;abstained=False
    for key,q in row['questions'].items():
        # A native refusal must not become a forced answer when we standardize
        # tie handling. Keep it in the denominator, outside accepted coverage.
        if pred.get('abstained',{}).get(key):
            abstained=True
            heads.append({'correct':False,'gold':row['gold'][key],'guess':'__failed__','multi':q['type']=='multilabel','answered':False})
            continue
        gold=row['gold'][key];probs=pred.get('probabilities',{}).get(key,{})
        keys=list(map(str,range(len(q['criteria'])))) if q['type']=='score' else list(q.get('criteria') or ['false','true'])
        valid=set(probs)==set(keys) and all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in probs.values())
        valid=valid and (q['type']=='multilabel' or abs(sum(probs.values())-1)<=.020000001)
        if not valid:return {'correct':False,'valid':False,'heads':[], 'native_changed':False}
        if q['type']=='multilabel':
            guess=[k for k in keys if probs[k]>=.5]
            heads.append({'correct':set(guess)==set(gold),'gold':gold,'guess':guess,'multi':True})
        else:
            total=sum(probs.values());probs={k:probs[k]/total for k in keys}
            order=sorted(keys) if 'native_task' in row else keys
            guess=max(order,key=probs.__getitem__)
            # Noul false/true lexical order equals public no/yes order.
            native_changed|=pred.get('pred',{}).get(key)!=[guess]
            heads.append({'correct':guess==gold[0],'gold':gold[0],'guess':guess,'confidence':probs[guess],
                          'brier':sum((v-int(k==gold[0]))**2 for k,v in probs.items()),
                          'nll':-math.log(max(probs[gold[0]],1e-15)),'multi':False})
    return {'correct':all(h['correct'] for h in heads),'valid':not abstained,'heads':heads,'native_changed':native_changed,'abstained':abstained}

def aggregate(rows,predictions):
    scores=[score(r,p) for r,p in zip(rows,predictions)]
    n=len(rows);hits=sum(s['correct'] for s in scores);valid=sum(s['valid'] for s in scores)
    heads=[h for s in scores for h in s['heads']];cal=[h for h in heads if not h['multi'] and h.get('answered',True)]
    result={'n':n,'correct':hits,'accuracy':hits/n,'wilson95':wilson(hits,n),'valid':valid,'failures':n-valid,'coverage':valid/n,
            'heads':sum(len(r['questions']) for r in rows),'head_correct':sum(h['correct'] for h in heads),
            'native_tie_decisions_changed':sum(s['native_changed'] for s in scores),
            'abstained_records':sum(s.get('abstained',False) for s in scores),
            'truncated_records':sum(any(a.get('state_truncated') or a.get('instruction_truncated') or a.get('options_truncated') for a in p.get('truncation_audit',{}).values()) for p in predictions)}
    result['head_accuracy']=result['head_correct']/result['heads']
    # Review sentences and paraphrases from the same source are correlated.
    families=[r.get('native_task',{}).get('group') or r.get('family') or r.get('id',str(i)) for i,r in enumerate(rows)]
    if 1<len(set(families))<n:
        import numpy as np
        grouped=defaultdict(list)
        for family,s in zip(families,scores):grouped[family].append(s['correct'])
        sums=np.array([sum(v) for v in grouped.values()]);sizes=np.array([len(v) for v in grouped.values()])
        draws=np.random.default_rng(20261001).integers(0,len(sums),size=(2000,len(sums)))
        result['cluster_bootstrap95']=list(map(float,np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975])))
        result['independent_source_groups']=len(sums)
    if cal:
        ece=0
        for lo in range(10):
            block=[h for h in cal if lo/10<=h['confidence']<(lo+1)/10 or lo==9 and h['confidence']==1]
            if block:ece+=len(block)/len(cal)*abs(sum(h['confidence']-h['correct'] for h in block)/len(block))
        result.update(brier=sum(h['brier'] for h in cal)/len(cal),nll=sum(h['nll'] for h in cal)/len(cal),ece10=ece,
                      calibration_n=len(cal),confident_wrong=sum(h['confidence']>=.9 and not h['correct'] for h in cal))
    # F1 is useful only for a fixed single-label task; include failed predictions.
    signatures={json.dumps(list(r['questions'].values()),ensure_ascii=False) for r in rows}
    if len(signatures)==1 and all(len(r['questions'])==1 and next(iter(r['questions'].values()))['type']!='multilabel' for r in rows):
        q=next(iter(rows[0]['questions'].values()));keys=list(map(str,range(len(q['criteria'])))) if q['type']=='score' else list(q.get('criteria') or ['false','true'])
        pairs=[(next(iter(r['gold'].values()))[0],s['heads'][0]['guess'] if s['valid'] else '__failed__') for r,s in zip(rows,scores)]
        f1=[];cm={k:{j:0 for j in keys+['__failed__']} for k in keys}
        for a,b in pairs:cm[a][b]+=1
        for k in keys:
            tp=cm[k][k];fp=sum(cm[a][k] for a in keys if a!=k);fn=sum(v for b,v in cm[k].items() if b!=k)
            f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
        result.update(macro_f1=sum(f1)/len(f1),confusion_matrix=cm,labels=keys,
                      retrospective_majority_baseline=max(Counter(a for a,b in pairs).values())/n)
    return result,scores

def selective_policy(dev,devpred,test,testpred):
    ds=[score(r,p) for r,p in zip(dev,devpred)];ts=[score(r,p) for r,p in zip(test,testpred)]
    if not all(len(r['questions'])==1 for r in dev+test):return None
    def accepted(s,t):return s['valid'] and not s['heads'][0]['multi'] and s['heads'][0]['confidence']>=t
    chosen=None;devstats=None
    for threshold in [.5,.6,.7,.8,.9,.95,.98,.99,1.0]:
        subset=[s for s in ds if accepted(s,threshold)];n=len(subset);correct=sum(s['correct'] for s in subset)
        if n>=30 and wilson(correct,n)[0]>=.95:
            chosen=threshold;devstats={'n':n,'correct':correct,'wilson95':wilson(correct,n)};break
    if chosen is None:
        return {'threshold':None,'test_coverage':0.,'accepted_n':0,'accepted_accuracy':None,
                'reason':'No predeclared threshold passed a 95% development Wilson lower bound with at least 30 accepted examples. All cases remain for review.'}
    subset=[s for s in ts if accepted(s,chosen)];n=len(subset);correct=sum(s['correct'] for s in subset)
    return {'threshold':chosen,'development':devstats,'test_coverage':n/len(test),'accepted_n':n,
            'accepted_correct':correct,'accepted_accuracy':correct/n if n else None,'accepted_wilson95':wilson(correct,n),
            'errors_per_1000_all_test_cases':1000*(n-correct)/len(test),
            'note':'Selected on development only, applied unchanged to test. An empirical pilot estimate, not a production guarantee.'}

def build(charts=False):
    rows=read_rows(ROOT/'data/prepared/jev_live.jsonl');byid={r['id']:r for r in rows}
    groups=defaultdict(list)
    for r in rows:groups[r['suite']].append(r)
    predictions={JEV:{p['id']:p for p in read_rows(OUT/'predictions.jsonl')}}
    for path in (ROOT/'results/alternatives/models').glob('*.json'):
        name=path.stem;values=[]
        for fixture in ['jev_fresh','jev_verified','alternatives_claims','alternatives_typed','runs']:
            values.extend(read_rows(ROOT/f'results/alternatives/{fixture}/{name}/predictions.jsonl'))
        predictions[name]={p['id']:p for p in values}
    report={'updated':now(),'jev_version':JEV,'status':'in_progress','case_count':len(rows),
            'api_progress':read_json(OUT/'progress.json'),'spend':read_json(OUT/'spend.json'),
            'fixture_sha256':digest(ROOT/'data/prepared/jev_live.jsonl'),'models':{},'paired_comparisons':{}}
    from .industry import keyword_route
    report['keyword_baseline']={}
    for suite,items in groups.items():
        if not suite.startswith(('industry/','fresh_business/')) or suite.endswith('/reversed'):continue
        test=[r for r in items if r.get('split')!='development'];task=suite.split('/')[1]
        correct=sum(keyword_route(r['state'],task)==next(iter(r['gold'].values()))[0] for r in test)
        report['keyword_baseline'][suite]={'n':len(test),'correct':correct,'accuracy':correct/len(test),'wilson95':wilson(correct,len(test)),
                                         'note':'Existing fixed keyword rules, no fitting. Label-only baseline: no invented confidence or calibration.'}
    if (OUT/'diagnostics.json').exists():report['diagnostics']=read_json(OUT/'diagnostics.json')
    for name,preds in predictions.items():
        model={'label':NAMES.get(name,name),'suites':{},'completed_cases':sum(k in byid for k in preds)}
        for suite,items in groups.items():
            test=[r for r in items if r.get('split')!='development']
            available=[r for r in test if r['id'] in preds]
            entry={'expected':len(test),'attempted':len(available),'complete':len(available)==len(test)}
            if entry['complete']:
                entry.update(aggregate(test,[preds[r['id']] for r in test])[0])
                dev=[r for r in items if r.get('split')=='development']
                if dev and all(r['id'] in preds for r in dev):
                    entry['selective_policy']=selective_policy(dev,[preds[r['id']] for r in dev],test,[preds[r['id']] for r in test])
            model['suites'][suite]=entry
        report['models'][name]=model
    for name,preds in predictions.items():
        if name==JEV:continue
        comparisons={}
        for suite,items in groups.items():
            test=[r for r in items if r.get('split')!='development']
            if all(r['id'] in preds and r['id'] in predictions[JEV] for r in test):
                local=[score(r,preds[r['id']])['correct'] for r in test]
                jev=[score(r,predictions[JEV][r['id']])['correct'] for r in test]
                cases=[{**r,'native_task':{'group':r.get('native_task',{}).get('group') or r.get('family') or r['id']}} for r in test]
                comparisons[suite]=paired_summary(cases,local,jev)
                interval=comparisons[suite]['paired_cluster_bootstrap95_pp']
                comparisons[suite]['degenerate_bootstrap']=interval[0]==interval[1]
                if interval[0]==interval[1]:comparisons[suite]['inference_warning']='A degenerate empirical bootstrap is not proof of equivalence or zero uncertainty beyond these cases.'
        report['paired_comparisons'][name]=comparisons
    required=['fresh/news','fresh/emotion','fresh/spam','fresh/phishing']
    report['priority_comparison_complete']=all(all(report['models'][m]['suites'][s]['complete'] for s in required) for m in [JEV]+PRIORITY)
    if report['api_progress']['completed']==len(rows) and report['priority_comparison_complete']:report['status']='live_run_and_priority_comparison_complete'
    write_json(OUT/'comparison.json',report);render(report,rows,predictions)
    if charts:draw(report)
    print(report['status'],'API',report['api_progress']['completed'],'/',len(rows),'priority fresh complete:',report['priority_comparison_complete'],flush=True)
    return report

def render(report,rows,predictions):
    esc=lambda v:html.escape(str(v))
    pct=lambda v:'—' if v is None else f'{100*v:.1f}%'
    money=report['spend']['estimated_cost_usd'];progress=report['api_progress']
    sections=[]
    diagnostics=''
    if report.get('diagnostics',{}).get('completed')==report.get('diagnostics',{}).get('expected') and report.get('diagnostics'):
        d=report['diagnostics']['by_variant']
        diagnostics=f'<section><h2>Repeatability and option order</h2><p>Two exact repeats each preserved {d["repeat1"]["same_decision"]}/{d["repeat1"]["n"]} and {d["repeat2"]["same_decision"]}/{d["repeat2"]["n"]} sampled Jev decisions. Reversing the available choice order preserved {d["reversed_choices"]["same_decision"]}/{d["reversed_choices"]["n"]}. These probes use cases selected by a fixed ID hash, independent of their results; they are diagnostic repeats, not additional accuracy evidence.</p><p><a href="diagnostics.json">Changed case IDs and probability changes</a> · <a href="verification.json">Request, model and spend reconciliation</a></p></section>'
    for suite in FOCUS:
        items=[]
        for name,model in report['models'].items():
            s=model['suites'].get(suite,{})
            if not s.get('complete'):continue
            pair=report['paired_comparisons'].get(name,{}).get(suite)
            delta=f'{pair["difference_pp"]:+.1f} pp <small>95% paired interval {pair["paired_cluster_bootstrap95_pp"][0]:+.1f} to {pair["paired_cluster_bootstrap95_pp"][1]:+.1f}</small>' if pair else 'Reference' if name==JEV else 'Awaiting Jev'
            if pair and pair.get('degenerate_bootstrap'):delta=f'{pair["difference_pp"]:+.1f} pp <small>No variation in observed paired differences; bootstrap cannot establish equivalence.</small>'
            policy=s.get('selective_policy');accept='—' if not policy else pct(policy['test_coverage'])
            if policy and policy.get('accepted_n'):accept+=f'<small>{policy["accepted_correct"]}/{policy["accepted_n"]} accepted test cases correct</small>'
            trunc=f'<small>{s["truncated_records"]} inputs truncated by native defaults</small>' if s['truncated_records'] else ''
            interval=s.get('cluster_bootstrap95',s['wilson95']);method='source-cluster CI' if 'cluster_bootstrap95' in s else 'CI'
            abstention=f'<small>{s["abstained_records"]} native abstentions included</small>' if s.get('abstained_records') else ''
            head_note=f'<small>{s["head_correct"]}/{s["heads"]} individual questions correct ({pct(s["head_accuracy"])})</small>' if s['heads']!=s['n'] else ''
            items.append((name,f'<tr><th>{esc(model["label"])}{trunc}</th><td>{s["correct"]}/{s["n"]}<strong>{pct(s["accuracy"])}</strong><small>95% {method} {pct(interval[0])}–{pct(interval[1])}</small>{head_note}</td><td>{pct(s.get("macro_f1"))}</td><td>{s["failures"]}{abstention}</td><td>{delta}</td><td>{accept}</td></tr>'))
        if not items:continue
        body=''.join(t for _,t in sorted(items,key=lambda x:([JEV]+PRIORITY).index(x[0]) if x[0] in [JEV]+PRIORITY else 100+list(report['models']).index(x[0])))
        baseline=report['keyword_baseline'].get(suite)
        if baseline:body+=f'<tr><th>Fixed keyword baseline</th><td>{baseline["correct"]}/{baseline["n"]}<strong>{pct(baseline["accuracy"])}</strong></td><td colspan="4">Label-only rules; no confidence estimate.</td></tr>'
        content=f'<table><thead><tr><th>Completed task</th><th>Accuracy</th><th>Macro F1</th><th>Failed</th><th>Difference vs live Jev</th><th>Accepted without review</th></tr></thead><tbody>{body}</tbody></table>'
        title=esc(TITLES.get(suite,suite.replace('/',' · ')))
        if suite.startswith('kev_claim/'):
            content='<p>Accuracy above counts a request correct only when every question is correct; per-question accuracy is also shown. Kev was trained on these task distributions: CFPB complaint templates, generated skill families and public developer-data proxies. These tests do not establish independent industrial-domain performance. The common developer benchmark retains all 1073 questions; the separate claim audit follows the publisher amendment to 1071.</p>'+content
        if suite.startswith(('fresh_business/','laya_claim/','jev_verified/','jevbench_public/','kev_claim/')):content=f'<details><summary><strong>{title}</strong> · inspect complete results</summary>{content}</details>'
        else:content=f'<h2>{title}</h2>'+content
        sections.append(f'<section id="{esc(suite.replace("/","-"))}">{content}</section>')
    sample=[]
    for r in rows:
        if not r['suite'].startswith('fresh_business/') or r.get('split')!='test':continue
        sample.append({'text':r['state'],'language':r['language'],'task':r['suite'].split('/')[1],
                       'expected':next(iter(r['gold'].values()))[0], 'id':r['id'],
                       'models':{NAMES.get(m,m):{'prediction':p[r['id']].get('pred',{}),'confidence':p[r['id']].get('probabilities',{}),'error':p[r['id']].get('error'), 'abstained':any(p[r['id']].get('abstained',{}).values()),'correct':score(r,p[r['id']])['correct']} for m,p in predictions.items() if r['id'] in p}})
    sample_json=json.dumps(sample,ensure_ascii=False).replace('<','\\u003c')
    output=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Jev and local models: business evaluation</title>
<style>body{{margin:0;background:#f5f7fa;color:#152b40;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1260px;margin:auto;padding:42px 30px}}h1{{font-size:38px;line-height:1.2;letter-spacing:-1px}}h2{{font-size:24px;line-height:1.3}}.eyebrow{{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#546c82}}section{{background:#fff;border:1px solid #dce3e9;padding:24px 28px;margin:24px 0}}.note{{padding:18px 22px;background:#e9eff4;border-left:4px solid #327f80}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #e4eaf0}}th{{font-weight:550}}thead{{color:#587085;font-size:12px}}td strong,small{{display:block}}small{{font-size:11px;color:#5e7385}}a{{color:#166a79}}input,select{{font:inherit;padding:7px;border:1px solid #b9c8d3;border-radius:2px}}label{{display:inline-block;margin:8px 14px 8px 0}}input{{width:105px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;padding:16px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.cards div{{background:#eef3f6;padding:16px}}.cards strong{{font-size:26px;display:block}}@media(max-width:900px){{main{{padding:20px 10px}}section{{overflow-x:auto;padding:18px}}.cards{{grid-template-columns:1fr}}}}</style>
<main><div class="eyebrow">Decision models in software · 1 October 2026</div><h1>Where can a local model compete with Jev?</h1>
<p class="note"><strong>{'LIVE API RUN COMPLETE' if progress['completed']==progress['expected'] else 'RUN IN PROGRESS'}</strong> · {progress['completed']:,}/{progress['expected']:,} live requests completed. Local alternatives may still be running. Only complete task groups appear below.<br>Updated {esc(report['updated'])}. Fresh strong-task comparison: {'complete for the seven priority alternatives' if report['priority_comparison_complete'] else 'in progress'}.</p>
<div class="cards"><div>Live reference<strong>Jev 1.13.0</strong>Exact version, current API calls</div><div>New public test cases<strong>800</strong>Plus 400 development examples</div><div>Estimated API spend<strong>${money:.4f}</strong>Reported input tokens × dated public price</div></div>
<p><a href="figures/index.html">Open the shareable graphics: accuracy, response time, technology, confidence and business value</a></p>
<section><h2>The practical use cases</h2><table><thead><tr><th>Software workflow</th><th>Value to measure</th><th>Deployment decision</th></tr></thead><tbody>
<tr><th>Engineering document intake</th><td>Suggest drawing, inspection, purchase or maintenance-record folders in a document management system.</td><td>Start with a reviewer accepting or correcting suggestions. Measure seconds saved per document, corrections and search failures.</td></tr>
<tr><th>Manufacturing and oil-and-gas maintenance helpdesk</th><td>Suggest maintenance, procurement, document control or IT queues; refer mixed requests for review.</td><td>Evaluate on real, labelled tickets with technicians and document controllers. Track misrouting and resolution delay. This test does not cover equipment control or work-permit approval.</td></tr>
<tr><th>Helpdesk message screening</th><td>Prioritize suspected spam or scam messages for review and reduce irrelevant workload.</td><td>Measure false positives on legitimate messages. Public spam/phishing labels do not establish safe automatic deletion or security protection.</td></tr>
<tr><th>Content and knowledge-base organization</th><td>Attach broad topic labels to incoming articles; use confidence to offer a shortlist.</td><td>AG News is a transfer benchmark, not evidence that the same model understands a company's technical taxonomy.</td></tr>
</tbody></table></section>
<section><h2>A fair comparison</h2><p>Every model receives the same frozen input, question and complete option list. We chose whole task groups where local alternatives had shown promise, then drew new examples without consulting model outputs. These fresh results are shown separately from vendor replication, historical Jev results and the harder general suites. No individual cases were selected because a local model answered correctly.</p>
<p>Accuracy includes failed requests. Tables show raw counts, uncertainty and paired differences against the live API. Differences are descriptive and are not adjusted for multiple model comparisons. A narrow lead is not sufficient evidence of a general winner. Calibration uses the returned probability vectors; typed, valid output can still be wrong.</p>
<p>Confidence thresholds use development cases only. A threshold qualifies when its development accuracy has a Wilson 95% lower bound of at least 95%, with at least 30 accepted cases. The chosen threshold is then fixed for the test set. Zero accepted coverage means the experiment does not justify automatic acceptance. Tiny synthetic business sets are intended for reviewing behaviour, not certifying a threshold.</p>
<p>The exact banking pilot has all 72 long option descriptions. Laya's native question budget truncates these descriptions on every banking case, so its score measures an unsuitable default configuration for that request shape. It must not be read as a general banking-language score. Native truncation counts are shown alongside the relevant model.</p>
<p>The new business set contains 32 bilingual scenario families: 8 for development and 24 for testing. Labels and translations are assistant-authored and await domain-expert review. Public datasets may have been seen during training. <a href="https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md">Laya reports spam and phishing in its training mix</a>; perfect scores here do not establish general security performance. <a href="https://huggingface.co/wfzyx/von">Von reports emotion data in training and fitting its confidence map on public JevBench</a>. Other model cards disclose broader public multi-task training; Jev's row-level training exposure is unknown. This is a comparison of released models, not a guarantee of unseen training data. A planned full Intern reproduction overlaps the AG News corpus; overlapping results are not additional independent evidence.</p></section>
{''.join(sections)}{diagnostics}
<section><h2>Inspect a business example</h2><p>These are saved benchmark outputs. Selecting a case makes no API call. Review the expected label and model suggestions to understand failure modes.</p><select id="case" aria-label="Business example"></select><p id="caseText"></p><p id="expected"></p><table><thead><tr><th>Model</th><th>Suggested label</th><th>Confidence</th><th>Matches expected label</th></tr></thead><tbody id="answerRows"></tbody></table><details><summary>Inspect full saved probabilities</summary><pre id="answers"></pre></details></section>
<section><h2>Translate a pilot into business value</h2><p>These are editable assumptions, not measured savings. Use net time saved after review and corrections. Infrastructure, integration, support and API costs belong in annual operating cost.</p>
<label>Cases / year<br><input id="volume" type="number" value="100000" min="0"></label><label>Workflow adoption %<br><input id="adoption" type="number" value="60" min="0" max="100"></label><label>Net seconds saved<br><input id="seconds" type="number" value="20" min="0"></label><label>Loaded cost / hour<br><input id="rate" type="number" value="60" min="0"></label><label>Annual operating cost<br><input id="cost" type="number" value="5000" min="0"></label><p id="value"></p><small>Use one consistent currency for the hourly rate, operating cost and result. This model does not monetize avoided downtime or safety outcomes.</small></section>
<section><h2>Audit and limitations</h2><p><a href="comparison.json">Machine-readable results and calibration metrics</a> · <a href="protocol.json">Frozen protocol</a> · <a href="../alternatives/jev_fresh-protocol.json">Fresh-case provenance</a> · <a href="spend.json">Token and budget ledger summary</a> · <a href="../alternatives/report.html">Full local comparison</a> · <a href="../alternatives/claim_audit.html">Published claim audit</a> · <a href="../alternatives/jev_comparison.html">Historical Jev verification</a></p>
<p>Local models run on an RTX 5070 Ti 16 GB in Python virtual environments; precision and adapter details are retained per model. API timings include network transport and concurrent requests, and are not directly comparable with local GPU throughput or H100 marketing numbers. The current run cannot establish advertised speed or cost multiples against unrelated LLM workflows. Published input pricing was <a href="https://typesafe.ai/">$42 per billion tokens</a>; output tokens are free according to the <a href="https://api.typesafe.ai/openapi.json">API schema</a>. Estimated spend is not an account statement.</p></section></main>
<script>const examples={sample_json};const selector=document.getElementById('case');examples.forEach((e,i)=>{{let o=document.createElement('option');o.value=i;o.textContent=e.language.toUpperCase()+' / '+e.task+' / '+e.id.split('/').pop();selector.append(o)}});function show(){{const e=examples[Number(selector.value)];if(!e)return;document.getElementById('caseText').textContent=e.text;document.getElementById('expected').textContent='Expected label: '+e.expected;document.getElementById('answers').textContent=JSON.stringify(e.models,null,2);const body=document.getElementById('answerRows');body.replaceChildren();Object.entries(e.models).forEach(([name,a])=>{{const tr=document.createElement('tr');const key=Object.keys(a.prediction)[0];const label=a.error?'Request failed':a.abstained?'Sent for review':(a.prediction[key]||[]).join(', ');const confidence=a.confidence[key]&&a.prediction[key]&&a.prediction[key].length===1?a.confidence[key][a.prediction[key][0]]:null;[name,label,confidence==null?'—':(100*confidence).toFixed(1)+'%',a.correct?'Yes':'No'].forEach(text=>{{const cell=document.createElement('td');cell.textContent=text;tr.append(cell)}});body.append(tr)}})}}selector.addEventListener('change',show);show();function calc(){{const get=id=>Math.max(0,Number(document.getElementById(id).value)||0);let hours=get('volume')*Math.min(100,get('adoption'))/100*get('seconds')/3600;document.getElementById('value').textContent='Illustrative annual hours saved: '+hours.toLocaleString(undefined,{{maximumFractionDigits:0}})+'. Net annual value: '+(hours*get('rate')-get('cost')).toLocaleString(undefined,{{maximumFractionDigits:0}})+' in your chosen currency.'}}document.querySelectorAll('input').forEach(x=>x.addEventListener('input',calc));calc();</script></html>'''
    (OUT/'report.html').write_text(output,encoding='utf-8')

def draw(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    target=OUT/'figures';target.mkdir(exist_ok=True)
    panels=[('fresh/news','News categories'),('fresh/emotion','Emotion categories'),('fresh/spam','Spam screening'),('fresh/phishing','Phishing / scam screening')]
    if not all(report['models'][JEV]['suites'].get(s,{}).get('complete') for s,t in panels):return
    selected=[m for m in PRIORITY if all(report['models'][m]['suites'].get(s,{}).get('complete') for s,t in panels)]
    if not selected:return
    names=[JEV]+selected
    fig,axes=plt.subplots(2,2,figsize=(14,11),dpi=160)
    fig.patch.set_facecolor('white')
    for ax,(suite,title) in zip(axes.flat,panels):
        metrics=[report['models'][m]['suites'][suite] for m in names];y=np.arange(len(names))
        values=np.array([m['accuracy']*100 for m in metrics]);err=np.maximum(0,np.array([[v-m['wilson95'][0]*100,m['wilson95'][1]*100-v] for v,m in zip(values,metrics)]).T)
        ax.barh(y,values,color=['#153a55']+['#25817b']*len(selected),height=.6,xerr=err,error_kw={'elinewidth':.8,'capsize':2,'ecolor':'#78909f'})
        ax.set_yticks(y,[NAMES[m].replace(' · live API','') for m in names]);ax.invert_yaxis();ax.set_xlim(0,118)
        ax.set_xticks([0,25,50,75,100],['0','25','50','75','100%']);ax.set_title(title,loc='left',fontweight='bold',pad=14)
        for i,(v,m) in enumerate(zip(values,metrics)):ax.text(116,i,f'{v:.1f}',va='center',ha='right',fontsize=10)
        ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.suptitle('Local decision models against live Jev',x=.035,y=.975,ha='left',fontsize=23,fontweight='bold',color='#153a55')
    fig.text(.035,.925,'Fresh public test cases · 200 per task · identical inputs and labels · 95% accuracy intervals',fontsize=12,color='#506777')
    qualifier='Seven priority alternatives complete' if len(selected)==7 else f'INTERIM · {len(selected)} of seven priority alternatives complete'
    fig.text(.035,.034,qualifier+' · 1 October 2026\nTask training exposure: Laya reports spam/phishing; Von reports emotion. Exact test overlap unknown.\nNew cases from tasks selected for prior local strengths. Sampling intervals do not establish a general winner.',fontsize=10,color='#506777')
    fig.subplots_adjust(left=.2,right=.97,top=.86,bottom=.15,wspace=.72,hspace=.3)
    for ext in ['png','svg']:fig.savefig(target/f'fresh-local-vs-jev.{ext}',facecolor='white')
    plt.close(fig)
    # Separate, explicitly small synthetic industry pilot. Do not mix with public tests.
    panels=[('industry/routing/en','Ticket routing · English'),('industry/routing/nb','Ticket routing · Norwegian'),
            ('industry/documents/en','Engineering documents · English'),('industry/documents/nb','Engineering documents · Norwegian')]
    names=[m for m in [JEV]+PRIORITY if all(report['models'][m]['suites'].get(s,{}).get('complete') for s,t in panels)]
    if JEV not in names or len(names)<2:return
    fig,axes=plt.subplots(2,2,figsize=(14,11),dpi=160)
    fig.patch.set_facecolor('white')
    for ax,(suite,title) in zip(axes.flat,panels):
        metrics=[report['models'][m]['suites'][suite] for m in names];y=np.arange(len(names))
        values=np.array([m['accuracy']*100 for m in metrics]);err=np.maximum(0,np.array([[v-m['wilson95'][0]*100,m['wilson95'][1]*100-v] for v,m in zip(values,metrics)]).T)
        ax.barh(y,values,color=['#153a55' if m==JEV else '#25817b' for m in names],height=.6,xerr=err,error_kw={'elinewidth':.8,'capsize':2,'ecolor':'#78909f'})
        ax.set_yticks(y,[NAMES[m].replace(' · live API','') for m in names]);ax.invert_yaxis();ax.set_xlim(0,118)
        ax.set_xticks([0,25,50,75,100],['0','25','50','75','100%']);ax.set_title(title.replace('Engineering documents','Documents')+f'\n{metrics[0]["n"]} test cases',loc='left',fontweight='bold',pad=14,fontsize=12)
        for i,v in enumerate(values):ax.text(116,i,f'{v:.1f}',va='center',ha='right',fontsize=10)
        ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.suptitle('A business pilot: route requests and file documents',x=.035,y=.975,ha='left',fontsize=23,fontweight='bold',color='#153a55')
    fig.text(.035,.925,'English and Norwegian · same tasks, inputs and choices · accuracy with 95% intervals',fontsize=12,color='#506777')
    fig.text(.035,.034,'Seven priority local alternatives · synthetic pilot · 1 October 2026\n68 paired scenario families; assistant-authored labels. Domain-expert validation remains necessary.\nEvaluate as staff suggestions first. These results do not validate equipment control or safety decisions.',fontsize=10,color='#506777')
    fig.subplots_adjust(left=.2,right=.97,top=.86,bottom=.15,wspace=.72,hspace=.3)
    for ext in ['png','svg']:fig.savefig(target/f'business-local-vs-jev.{ext}',facecolor='white')
    plt.close(fig)
    # Broad bilingual transfer task; only models completing both languages.
    suites=['massive_en','massive_nb']
    names=[m for m in report['models'] if all(report['models'][m]['suites'].get(s,{}).get('complete') for s in suites)]
    if JEV not in names:return
    names=[JEV]+sorted((m for m in names if m!=JEV),key=lambda m:report['models'][m]['suites']['massive_nb']['accuracy'],reverse=True)
    fig,axes=plt.subplots(1,2,figsize=(14,max(7,3+.48*len(names))),dpi=160)
    fig.patch.set_facecolor('white')
    for ax,suite,title in zip(axes,suites,['English','Norwegian Bokmål']):
        metrics=[report['models'][m]['suites'][suite] for m in names];y=np.arange(len(names))
        values=np.array([s['accuracy']*100 for s in metrics])
        err=np.maximum(0,np.array([[v-s['wilson95'][0]*100,s['wilson95'][1]*100-v] for v,s in zip(values,metrics)]).T)
        ax.barh(y,values,color=['#153a55']+['#25817b']*(len(names)-1),height=.58,xerr=err,error_kw={'elinewidth':.8,'capsize':2,'ecolor':'#78909f'})
        ax.set_yticks(y,[NAMES[m].replace(' · live API','') for m in names]);ax.invert_yaxis();ax.set_xlim(0,115)
        ax.set_xticks([0,25,50,75,100],['0','25','50','75','100%']);ax.set_title(title,loc='left',fontweight='bold',pad=14)
        for i,v in enumerate(values):ax.text(113,i,f'{v:.1f}',va='center',ha='right',fontsize=10)
        ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.suptitle('Intent recognition across English and Norwegian',x=.035,y=.975,ha='left',fontsize=22,fontweight='bold',color='#153a55')
    fig.text(.035,.915,'MASSIVE · 18 intent scenarios · 2,948 test cases per language · 95% accuracy intervals',fontsize=12,color='#506777')
    fig.text(.035,.034,f'INTERIM model matrix · {len(names)-1} completed local alternatives and live Jev 1.13.0 · 1 October 2026\nIdentical frozen requests. Local precision varies; failed requests count as incorrect.\nPublic task training exposure is unknown. This is language transfer evidence, not industrial-domain validation.',fontsize=10,color='#506777')
    fig.subplots_adjust(left=.23,right=.97,top=.84,bottom=.18,wspace=.78)
    for ext in ['png','svg']:fig.savefig(target/f'language-local-vs-jev.{ext}',facecolor='white')
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--charts',action='store_true');a=p.parse_args();build(a.charts)
