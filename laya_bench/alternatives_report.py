"""Rebuild the comparison from auditable predictions; never rank partial runs."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import html
import json
import math
import sys
from .common import ROOT,read_json,write_json
from .metrics import wilson

OUT=ROOT/'results/alternatives'
NAMES={'gliner-decide':'GLiNER Decide 340M','gliner-decide-multi':'GLiNER Decide multilingual','gliner-decide-1b':'GLiNER Decide 1B','decider-08b':'Decider 0.8B','decider-2b':'Decider 2B','decider-4b':'Decider 4B','kev-08b':'Kev 0.8B','kev-4b':'Kev 4B','julia':'Julia 144M','jevk5':'JevK5 4B','jevk5-2b':'JevK5 2B','plumb-4b':'Plumb 4B','tev1':'Tev1 4B','von':'Von','nev-2b':'Nev 2B Q8','intern-decision-4b':'Intern-Decision 4B','wald-4b-v12':'Wald 4B v1.2','nimble-9b':'Nimble 9B NF4','imajev-2b':'Imajev 2B','imajev-4b':'Imajev 4B','laya':'Laya','laya-multilingual':'Laya multilingual','clm-int8':'CLM 8B INT8'}

NAMES.update({'winnow-12b-q8':'Winnow 12B Q8 (Windows CUDA)', 'decision-4b-v12':'Decision 4B v1.2', 'cygnet-12b-nf4':'Cygnet 12B NF4 (local variant)', 'jev-omni-12b-nf4':'Jev-Omni 12B NF4 (local variant)'})
NAMES.update({'decision2-kai-06b': 'Decision 2.0 Kai 0.6B', 'decision2-eos-08b': 'Decision 2.0 Eos 0.8B',
              'decision2-sol-2b': 'Decision 2.0 Sol 2B', 'decision2-nox-4b': 'Decision 2.0 Nox 4B'})

def lines(path):
    if not path.exists():return []
    # A running writer may have an incomplete trailing line; do not ingest it.
    content=path.read_text(encoding='utf-8');parts=content.splitlines()
    if content and not content.endswith('\n'):parts=parts[:-1]
    return [json.loads(s) for s in parts]

def evaluate(rows,predictions):
    assert len(predictions)==len({p['id'] for p in predictions})
    byid={p['id']:p for p in predictions};groups=defaultdict(list)
    for row in rows:groups[row['suite']].append(row)
    result={}
    for suite,items in groups.items():
        available=[(r,byid[r['id']]) for r in items if r['id'] in byid]
        if not available:continue
        correct=0;heads=0;head_correct=0;errors=Counter();scores=[];truth=[];pairs=[];truncations=0;abstentions=0
        for row,pred in available:
            assert pred['input_sha256']==row['input_sha256']
            correct+=bool(pred['correct'])
            if pred.get('error'):errors[pred['error'].split(':',1)[0]]+=1
            if any(pred.get('abstained',{}).values()) and not pred.get('error'):abstentions+=1
            if any(a.get('state_truncated') or a.get('options_truncated') or a.get('instruction_truncated') for a in pred.get('truncation_audit',{}).values()):truncations+=1
            for key,gold in row['gold'].items():
                heads+=1;guess=pred['pred'].get(key,[]);hit=set(guess)==set(gold);head_correct+=hit
                if row['questions'][key]['type']!='multilabel':
                    pairs.append((gold[0],guess[0] if guess else '__rejected__'))
                    p=pred.get('probabilities',{}).get(key)
                    if p and guess:
                        scores.append((max(p.values()),hit,sum((v-int(k==gold[0]))**2 for k,v in p.items())))
        n=len(available);successful=n-sum(errors.values())-abstentions
        # Head-wise calibration only over answered, single-label heads. Coverage
        # remains visible; no assertion that a confidence threshold is safe.
        ece=0
        for b in range(10):
            block=[s for s in scores if b/10<=s[0]<(b+1)/10 or b==9 and s[0]==1]
            if block:ece+=len(block)/max(1,len(scores))*abs(sum(s[0]-s[1] for s in block)/len(block))
        fixed_single_head=all(len(r['questions'])==1 and next(iter(r['questions'].values()))['type']!='multilabel' for r in items) and len({json.dumps([(q['type'],q.get('criteria')) for q in r['questions'].values()],sort_keys=True) for r in items})==1
        f1=[]
        for label in sorted({a for a,_ in pairs}) if fixed_single_head else []:
            tp=sum(a==b==label for a,b in pairs);fp=sum(a!=label and b==label for a,b in pairs);fn=sum(a==label and b!=label for a,b in pairs)
            f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
        result[suite]={'expected':len(items),'attempted':n,'complete':n==len(items),'correct':correct,'accuracy':correct/n,'wilson95':wilson(correct,n),'head_correct':head_correct,'heads':heads,'head_accuracy':head_correct/heads,'coverage':successful/n,'accuracy_on_successful':correct/successful if successful else None,'errors':dict(errors),'abstained_records':abstentions,'truncated_records':truncations,'macro_f1':sum(f1)/len(f1) if f1 else None,'answered_single_label_heads':len(scores),'ece10':ece if scores else None,'brier':sum(s[2] for s in scores)/len(scores) if scores else None}
    return result

def native_jev(rows,predictions):
    sys.path.insert(0,str(ROOT/'.cache/alternatives_research/code--fstandhartinger--jevbench'))
    from jevbench.scoring import score_task
    from jevbench.tasks import Task
    byid={p['id']:p for p in predictions};tiers=defaultdict(list)
    for r in rows:
        if not r['suite'].startswith('jevbench_public/') or r['id'] not in byid:continue
        pred=byid[r['id']]
        p={} if pred.get('abstained',{}).get('decision') else pred.get('probabilities',{}).get('decision',{})
        if r['questions']['decision']['type']=='noul':p={{'false':'no','true':'yes'}[k]:v for k,v in p.items()}
        tiers[r['suite']].append(score_task(p,Task.from_dict(r['native_task'])))
    return {k:{'n':len(v),'correct':sum(bool(x['correct']) for x in v),'accuracy':sum(bool(x['correct']) for x in v)/len(v),'strict_valid':sum(x['strict_valid'] for x in v),'valid':sum(x['valid'] for x in v),'renormalized':sum(x['renormalized'] for x in v)} for k,v in tiers.items()}

def build(charts=False):
    fixtures={f:lines(ROOT/f'data/prepared/{f}.jsonl') for f in ['alternatives','alternatives_claims','alternatives_typed','intern_claims','jev_verified','jev_fresh']}
    report={'updated':datetime.now(timezone.utc).isoformat(),'status':'in_progress','hardware':'RTX 5070 Ti 16 GB; Windows; Python venvs; 78% Torch allocation cap','models':{}}
    for meta_path in sorted((OUT/'models').glob('*.json')):
        name=meta_path.stem;model={'checkpoint':read_json(meta_path),'label':NAMES.get(name,name),'fixtures':{}}
        for fixture,rows in fixtures.items():
            root=OUT/('runs' if fixture=='alternatives' else fixture)/name
            predictions=lines(root/'predictions.jsonl')
            entry={'expected':len(rows),'attempted':len(predictions),'complete':len(predictions)==len(rows),'errors':sum(bool(p.get('error')) for p in predictions),'suites':evaluate(rows,predictions)}
            if (root/'metadata.json').exists():entry['runtime']=read_json(root/'metadata.json')
            if fixture=='alternatives_claims':entry['native_jev']=native_jev(rows,predictions)
            model['fixtures'][fixture]=entry
        report['models'][name]=model
    completed=[m for m in report['models'].values() if all(m['fixtures'][f]['complete'] for f in ['alternatives','alternatives_claims','alternatives_typed','jev_verified','jev_fresh'])]
    if (OUT/'jev_verification.json').exists():
        from .alternatives_jev import compare
        report['jev_comparison']=compare(report['models'])
    julia=OUT/'julia-native-typed/results.json'
    if julia.exists():
        native=read_json(julia);stats=native['original_criteria']['by_type']
        correct=sum(s['correct'] for s in stats.values());n=sum(s['count'] for s in stats.values())
        report['native_claim_checks']={'julia_typed':{'correct':correct,'n':n,'accuracy':correct/n,'published_accuracy':.7315,'difference_percentage_points':(correct/n-.7315)*100,'method':'Unmodified publisher reproduce_typed.py; verified model and Parquet hashes; CPU fp32 native Torch backend, batch one, 1024-token context','verdict':'Near reproduction within the predeclared 2 percentage point band; not exact numerical equality','source':'https://huggingface.co/SupersonicLabs/Julia-1/blob/a85b127321d580d65176c89ced8273f305745d85/metrics/accuracy-20260924.json','result':'julia-native-typed/results.json'}}
    report['complete_models']=len(completed);report['planned_models']=len(report['models'])
    from .alternatives_claim_audit import build as audit_claims
    report['claim_audit']=audit_claims(report)
    if len(completed)==len(report['models']) and len(completed)>=23:report['status']='shared_matrix_complete'
    write_json(OUT/'comparison.json',report)
    render(report)
    if charts:draw(report)
    if any(name.startswith('decision2-') for name in report['models']):
        from .decision2_report import build as decision2_results
        decision2_results()
    print(report['status'],f'{len(completed)}/{len(report["models"])} models finished all shared fixtures',flush=True)
    return report

def render(report):
    def esc(v):return html.escape(str(v))
    rows=[]
    for name,m in report['models'].items():
        cells=[]
        for f in ['alternatives','alternatives_claims','alternatives_typed','jev_verified','jev_fresh']:
            d=m['fixtures'][f];cells.append(f'<td>{d["attempted"]:,} / {d["expected"]:,}<small>{d["errors"]:,} failed requests</small></td>')
        rows.append('<tr><th>'+esc(m['label'])+'</th>'+''.join(cells)+'</tr>')
    metrics=[]
    claim_section=''
    for name,c in report.get('native_claim_checks',{}).items():
        claim_section+=f'<section><h2>Julia: first native claim check</h2><p>Our run: <strong>{c["accuracy"]:.2%}</strong> ({c["correct"]:,}/{c["n"]:,} decisions). Published: <strong>{c["published_accuracy"]:.2%}</strong>. Difference: {c["difference_percentage_points"]:+.2f} percentage points.</p><p>{esc(c["method"])}. {esc(c["verdict"])}. This verifies a public benchmark result, not deployment performance.</p><p><a href="{c["source"]}">Published metrics</a> · <a href="{c["result"]}">Local result</a></p></section>'
    for suite,title in [('massive_en','English intent scenarios'),('massive_nb','Norwegian intent scenarios'),('industry/routing/en','Industry routing · English'),('industry/routing/nb','Industry routing · Norwegian'),('industry/documents/en','Document classification · English'),('industry/documents/nb','Document classification · Norwegian')]:
        values=[]
        for m in report['models'].values():
            s=m['fixtures']['alternatives']['suites'].get(suite)
            if s and s['complete']:values.append((s['accuracy'],m['label'],s))
        body=''.join(f'<tr><th>{esc(label)}</th><td>{acc:.1%}</td><td>{s["wilson95"][0]:.1%}–{s["wilson95"][1]:.1%}</td><td>{s["coverage"]:.1%}</td></tr>' for acc,label,s in sorted(values,reverse=True))
        metrics.append(f'<section><h2>{title}</h2><table><thead><tr><th>Completed suite</th><th>Accuracy</th><th>95% interval</th><th>Request coverage</th></tr></thead><tbody>{body}</tbody></table></section>')
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Local decision model benchmark</title>
<style>body{{font:16px system-ui,sans-serif;color:#14273e;background:#f5f7fa;margin:0}}main{{max-width:1180px;margin:auto;padding:52px 36px}}h1{{font-size:38px;letter-spacing:-1px;margin:12px 0}}h2{{font-size:22px}}p{{max-width:900px;line-height:1.6}}.tag{{font-size:13px;text-transform:uppercase;letter-spacing:2px;color:#4b647f}}.note{{background:#e7edf3;padding:18px 22px;border-left:4px solid #59758e}}section{{background:white;padding:24px 28px;margin:24px 0;border:1px solid #d9e1e9}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:12px 10px;text-align:left;border-bottom:1px solid #e5eaf0}}thead{{color:#586b7e;font-size:13px}}small{{display:block;color:#6b7989;font-size:12px;margin-top:4px}}a{{color:#155972}}</style>
<main><div class="tag">Unofficial local software evaluation · {esc(report['updated'][:10])}</div><h1>Decision models under test</h1>
<p class="note"><strong>{report['complete_models']} of {report['planned_models']} models have recorded results for all shared fixtures.</strong><br>Updated {esc(report['updated'])}. Partial groups are excluded from accuracy tables. Native rejections and runtime failures remain in the denominator. Use as is, without warranty or vendor endorsement.</p>
<p>The question is where a small local model makes software more useful: classifying documents, suggesting ticket queues, or screening engineering records. Public benchmark replication and independent task transfer answer different questions. There is no pooled winner across unrelated tasks.</p>
<section><h2>Live and verified Jev comparisons</h2><p><a href="../jev_live/report.html">Current live Jev comparison and business findings</a> covers the shared fixtures and new disjoint test cases. <a href="jev_comparison.html">Historical Jev comparison on matching cases</a> uses 231 verified public per-case outcomes and a separate exact 300-case pilot. <a href="claim_audit.html">Claim-by-claim audit</a> separates reproduced, near-reproduced and unverifiable statements. Historical scores, current API calls and sealed leaderboard claims remain separate.</p></section>
<section><h2>Execution status</h2><p>{esc(report['hardware'])}. Failed requests stay in accuracy denominators. An unfinished or failed adapter is not a measured zero.</p><table><thead><tr><th>Model</th><th>Practical cases</th><th>Public claim cases</th><th>Typed cases · 5 decisions each</th><th>Exact Jev pilot</th><th>Fresh cases · dev + test</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
{claim_section}{''.join(metrics)}
<section><h2>How to interpret this</h2><p>MASSIVE measures 18 intent scenarios in English and Norwegian Bokmål, using 2,948 deduplicated test examples per language. The industry set contains assistant-authored scenarios: 40 routing and 28 document cases per language. These small synthetic samples need domain-expert review before a production decision. Reversing option order is a separate diagnostic, not additional independent evidence.</p>
<p>Accuracy includes rejected requests. Coverage reports the proportion receiving valid outputs. Confidence intervals are Wilson intervals for individual cases; they do not establish performance on new organizations or languages. Public publisher fixtures can have training or calibration exposure. Typed-decision reference labels use teacher agreement. Claim scoring uses the publisher's rules where available, including JevBench's distribution checks and tie-breaking.</p>
<p>Latency and peak memory need a separate controlled timing run. Current batch timings include different batching implementations; CLM can reuse an embedding cache, and Nev uses an external Vulkan process. These figures must not be compared with advertised H100 server latency.</p></section>
<section><h2>Business value to validate</h2><p><strong>Helpdesk:</strong> suggest a queue and retain the evidence for an agent to accept or correct. <strong>Oil and gas / engineering:</strong> suggest a document type or responsible discipline for incoming records. <strong>Manufacturing:</strong> organize maintenance descriptions and route exception reports. The benchmark does not validate safety decisions or equipment control.</p>
<p>Measure review time saved, correction rate, routing delays, and local deployment cost in a pilot. Net annual value = cases × adoption rate × net seconds saved ÷ 3,600 × loaded hourly cost − annual operating cost. Seconds saved must include correction and review time; benchmark accuracy alone cannot supply this number.</p></section></main></html>'''
    (OUT/'report.html').write_text(document,encoding='utf-8')

def draw(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    selected=[]
    for name,m in report['models'].items():
        suites=m['fixtures']['alternatives']['suites']
        if all(k in suites and suites[k]['complete'] for k in ['massive_en','massive_nb']):selected.append((m['label'],suites['massive_en'],suites['massive_nb']))
    if not selected:return
    selected.sort(key=lambda v:v[2]['accuracy'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    fig,ax=plt.subplots(figsize=(12,max(6,2.8+.55*len(selected))),dpi=160)
    fig.patch.set_facecolor('white');y=np.arange(len(selected));h=.32
    for offset,index,color,label in [(-h/2,1,'#647c93','English'),(h/2,2,'#147d78','Norwegian Bokmål')]:
        values=[r[index]['accuracy']*100 for r in selected]
        ax.barh(y+offset,values,height=h,color=color,label=label)
        for row,value in zip(y+offset,values):ax.text(value+1,row,f'{value:.1f}',va='center',fontsize=10,color='#14273e')
    ax.set_yticks(y,[r[0] for r in selected]);ax.set_xlim(0,105);ax.set_xticks([0,25,50,75,100],[f'{x}%' for x in [0,25,50,75,100]]);ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
    ax.set_xlabel('Accuracy · failed requests count as incorrect');ax.legend(frameon=False,loc='lower right')
    fig.suptitle('Does English performance transfer to Norwegian?',x=.035,ha='left',fontsize=20,fontweight='bold',color='#14273e',y=.97)
    fig.text(.035,.905,'MASSIVE · 18 intent scenarios · 2,948 test examples per language',fontsize=12,color='#516579')
    fig.text(.035,.025,f'Unofficial testing · {len(selected)} completed language groups · Windows / RTX 5070 Ti 16 GB · {report["updated"][:10]}\nPinned inputs and checkpoints · full methodology: results/alternatives/report.html',fontsize=9,color='#516579')
    fig.subplots_adjust(left=.28,right=.97,bottom=.16,top=.82)
    path=OUT/'figures';path.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(path/f'language-transfer-interim.{ext}',facecolor='white')
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--charts',action='store_true');args=parser.parse_args();build(args.charts)
    if (ROOT/'results/alternatives/leader-license-audit.json').exists():
        from .leaders_report import build as leader_report
        leader_report()
