"""Publication graphics drawn from measurements and explicit business assumptions."""
import argparse
import html
import json
from .common import ROOT,read_json,write_json,digest
from .alternatives_jev import read_rows
from .alternatives_report import NAMES

OUT=ROOT/'results/jev_live/figures'
NAVY='#153a55';TEAL='#25817b';MUTED='#506777';PALE='#edf3f6';RULE='#dce5eb'

def canvas(title,subtitle):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(14,11),dpi=160,facecolor='white')
    fig.text(.045,.95,title,fontsize=25,fontweight='bold',color=NAVY,va='top')
    fig.text(.045,.887,subtitle,fontsize=13,color=MUTED,va='top')
    return fig

def save(fig,name):
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',facecolor='white')
    plt.close(fig)

def technology():
    from matplotlib.patches import Rectangle
    fig=canvas('How decision models turn text into a choice',
               'Three local model designs and one managed service · architecture and deployment trade-offs')
    fig.text(.045,.817,'APPROACH / EXAMPLES',fontsize=10,color=MUTED,weight='bold')
    fig.text(.385,.817,'HOW IT WORKS',fontsize=10,color=MUTED,weight='bold')
    fig.text(.70,.817,'WHAT IT MEANS FOR SOFTWARE',fontsize=10,color=MUTED,weight='bold')
    entries=[
      ('Encoder + decision head','Laya · Laya multilingual\nGLiNER Decide · Julia · Von',
       'Read the text and question.\nScore the supplied choices\nwith a trained decision head.',
       'Compact local models.\nCheck context and option budgets;\nsmall size does not ensure accuracy.'),
      ('Language model + readout','Decider · Kev · JevK5',
       'Use a causal language-model\nbackbone and a trained readout\nto score the available choices.',
       'More GPU memory to manage.\nCompare each task and language;\nmodel size is not a quality guarantee.'),
      ('Contrastive embeddings','CLM v0.1 · Qwen3-8B backbone',
       'Encode state and actions separately.\nApply trained projection heads.\nRank by compatibility.',
       'Reusable action embeddings can\nhelp repeated candidate sets.\nMeasure cached and uncached paths.'),
      ('Managed decision API','TypeSafe Jev 1.13.0',
       'Send state and typed questions\nover HTTPS. Receive choices\nand probability distributions.',
       'No local GPU to operate.\nNetwork and service affect latency.\nInternal architecture is not verified here.')]
    for i,(name,examples,mechanism,meaning) in enumerate(entries):
        y=.775-i*.163
        fig.patches.append(Rectangle((.04,y-.128),.92,.15,transform=fig.transFigure,facecolor=PALE if i%2==0 else 'white',edgecolor=RULE,linewidth=.7))
        fig.patches.append(Rectangle((.04,y-.128),.004,.15,transform=fig.transFigure,facecolor=TEAL if i<3 else NAVY,edgecolor='none'))
        fig.text(.06,y,name,fontsize=16,color=NAVY,weight='bold',va='top')
        fig.text(.06,y-.04,examples,fontsize=12,color=MUTED,va='top',linespacing=1.6)
        fig.text(.385,y,mechanism,fontsize=13,color=NAVY,va='top',linespacing=1.7)
        fig.text(.70,y,meaning,fontsize=12.3,color=NAVY,va='top',linespacing=1.7)
    fig.text(.045,.082,'The output contract is similar. Accuracy, confidence and runtime still need separate tests.',fontsize=14,color=NAVY,weight='bold')
    fig.text(.045,.04,'Based on pinned publisher model cards and native inference code · 1 October 2026\nArchitecture summary, not a speed ranking. Sources and benchmark protocol accompany these graphics.',fontsize=10,color=MUTED)
    save(fig,'technology-approaches')

def business_value():
    from matplotlib.patches import Rectangle
    fig=canvas('Where a decision model can create business value',
               'Illustrative operating scenario · these savings have not been measured in a customer deployment')
    fig.texts[0].set_fontsize(23)
    usecases=[('HELPDESK','Suggest the right queue','Less manual sorting; fewer transfers'),
              ('ENGINEERING / OIL AND GAS','Suggest a document category','Faster intake and retrieval'),
              ('MANUFACTURING','Organize maintenance requests','Less administration for technicians')]
    for i,(domain,action,value) in enumerate(usecases):
        x=.045+i*.312
        fig.patches.append(Rectangle((x,.68),.294,.145,transform=fig.transFigure,facecolor=PALE,edgecolor=RULE,linewidth=.7))
        fig.text(x+.018,.8,domain,fontsize=10,color=TEAL,weight='bold')
        fig.text(x+.018,.757,action,fontsize=13,color=NAVY,weight='bold')
        fig.text(x+.018,.709,value,fontsize=11.3,color=MUTED)
    fig.text(.045,.622,'Assume 100,000 cases / year × 60% workflow adoption',fontsize=18,color=NAVY,weight='bold')
    fig.text(.045,.582,'Net time saved includes human review, corrections and rework.',fontsize=13,color=MUTED)
    ax=fig.add_axes([.20,.23,.365,.285])
    values=[50000,150000,250000];labels=['10 seconds saved','20 seconds saved','30 seconds saved']
    ax.barh(range(3),values,color=['#9db4c2',TEAL,'#9db4c2'],height=.48)
    ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.set_xlim(0,300000)
    ax.set_xticks([0,100000,200000,300000],['0','100k','200k','300k']);ax.set_xlabel('Illustrative net annual value · NOK',color=MUTED,labelpad=12)
    ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    for i,v in enumerate(values):ax.text(v+5000,i,f'{v:,.0f}',va='center',fontsize=12,color=NAVY)
    fig.text(.66,.493,'20 SECONDS PER ADOPTED CASE',fontsize=10,color=MUTED,weight='bold')
    fig.text(.66,.425,'333 hours',fontsize=31,color=NAVY,weight='bold')
    fig.text(.66,.384,'annual staff time recovered',fontsize=13,color=MUTED)
    fig.text(.66,.304,'NOK 150,000',fontsize=29,color=TEAL,weight='bold')
    fig.text(.66,.264,'illustrative net annual value',fontsize=13,color=MUTED)
    fig.text(.045,.135,'Pilot first: log accepted suggestions, corrections, task time and downstream errors.',fontsize=14,color=NAVY,weight='bold')
    fig.text(.045,.071,'Assumptions: NOK 600 loaded hourly cost; NOK 50,000 annual operating cost including integration and support.\nFormula: cases × adoption × net seconds saved ÷ 3,600 × hourly cost − operating cost.\nNo monetized safety or downtime benefit. Public benchmark accuracy cannot supply the time-saving assumption.',fontsize=10.5,color=MUTED,linespacing=1.5)
    save(fig,'business-value-scenario')

def latency():
    import numpy as np
    from .latency_benchmark import MODELS
    source=ROOT/'data/prepared/latency.jsonl';protocol=read_json(ROOT/'results/latency/protocol.json')
    assert digest(source)==protocol['fixture_sha256']
    ids={r['id']:r['input_sha256'] for r in read_rows(source)};stats={};expected=len(ids)*protocol['passes']
    names=MODELS+['jev-1.13.0']
    for name in names:
        rows=read_rows(ROOT/f'results/latency/{name}.jsonl')
        if len(rows)!=expected:continue
        assert len({(r['id'],r['pass']) for r in rows})==expected
        assert all(r['input_sha256']==ids[r['id']] for r in rows)
        valid=[r for r in rows if not r['error']];ms=[r['seconds']*1000 for r in valid]
        stats[name]={'calls':len(rows),'successful':len(valid),'errors':len(rows)-len(valid),
                     'median_ms':float(np.median(ms)) if ms else None,'p95_ms':float(np.quantile(ms,.95)) if ms else None,
                     'truncated_calls':sum(r['truncated'] for r in rows),
                     'by_pass':{str(i):{'median_ms':float(np.median([r['seconds']*1000 for r in valid if r['pass']==i]))} for i in range(1,4)}}
    write_json(ROOT/'results/latency/summary.json',{'complete':len(stats)==len(names),'protocol':protocol,'models':stats})
    if len(stats)!=len(names):print('Response-time graphic pending:',len(stats),'of',len(names),'models');return False
    fig=canvas('Response time on the same short-message workload',
               '80 fixed messages × 3 passes per model · one request at a time · model loading and warm-up excluded')
    order=sorted(stats,key=lambda m:stats[m]['median_ms'] if stats[m]['median_ms'] is not None else float('inf'))
    ax=fig.add_axes([.22,.27,.71,.51]);y=np.arange(len(order));h=.27
    med=[stats[m]['median_ms'] for m in order];p95=[stats[m]['p95_ms'] for m in order]
    ax.barh(y-h/2,med,height=h,color=TEAL,label='Median · typical request')
    ax.barh(y+h/2,p95,height=h,color=NAVY,label='95th percentile · slower requests')
    ax.set_yticks(y,[NAMES.get(m,'Jev 1.13.0 · API') for m in order]);ax.invert_yaxis();ax.tick_params(axis='y',length=0)
    end=max(p95);ax.set_xlim(0,end*1.15)
    for ys,values in [(y-h/2,med),(y+h/2,p95)]:
        for row,v in zip(ys,values):ax.text(v+end*.012,row,f'{v:.1f}',va='center',fontsize=10,color=NAVY)
    ax.set_xlabel('Milliseconds per request · lower is faster',labelpad=14,color=MUTED)
    ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);ax.spines['left'].set_visible(False)
    ax.legend(loc='upper left',bbox_to_anchor=(-.23,-.16),ncol=2,frameon=False,fontsize=12)
    errors=sum(v['errors'] for v in stats.values());truncated=sum(v['truncated_calls'] for v in stats.values())
    fig.text(.045,.116,f'{expected:,} timed calls per model · {errors} failed calls · {truncated} local calls with native truncation',fontsize=12,color=NAVY,weight='bold')
    fig.text(.045,.043,'Local: RTX 5070 Ti 16 GB, Windows; Decider uses eager execution and reference convolution kernels.\nAPI: persistent HTTPS connection; network and service time included; hosted hardware unknown.\nShort messages only (state ≤512 characters). Describes this deployment; does not reproduce vendor speed multiples. 1 October 2026.',fontsize=10.5,color=MUTED,linespacing=1.5)
    save(fig,'response-time')
    return True

def confidence():
    import numpy as np
    from matplotlib.patches import Patch
    from .jev_live_report import PRIORITY,JEV
    report=read_json(ROOT/'results/jev_live/comparison.json');names=[JEV]+PRIORITY
    fig=canvas('Confidence gates: coverage and mistakes',
               'Thresholds chosen on development data · applied unchanged to 200 test cases per task')
    for pos,suite,title in [([.21,.27,.31,.49],'fresh/spam','Spam screening'),([.67,.27,.29,.49],'fresh/phishing','Phishing / scam screening')]:
        ax=fig.add_axes(pos);y=np.arange(len(names));correct=[];wrong=[];review=[];policies=[]
        for name in names:
            s=report['models'][name]['suites'][suite];p=s['selective_policy'];policies.append(p)
            hit=p.get('accepted_correct',0);miss=p['accepted_n']-hit
            correct.append(100*hit/s['n']);wrong.append(100*miss/s['n']);review.append(100*(s['n']-p['accepted_n'])/s['n'])
        ax.barh(y,correct,color=TEAL,height=.53)
        ax.barh(y,wrong,left=correct,color='#b96748',height=.53)
        ax.barh(y,review,left=np.array(correct)+wrong,color='#e5ebf0',height=.53)
        ax.set_yticks(y,[NAMES.get(m,'Jev 1.13.0') for m in names]);ax.invert_yaxis();ax.tick_params(axis='y',length=0)
        ax.set_xlim(0,128);ax.set_xticks([0,50,100],['0','50','100%']);ax.set_title(title,loc='left',weight='bold',pad=19)
        ax.spines['left'].set_visible(False);ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
        for i,p in enumerate(policies):
            label=f'{p["accepted_n"]}/200' if p['accepted_n'] else 'Review all'
            ax.text(126,i,label,ha='right',va='center',fontsize=10,color=NAVY)
    fig.legend(handles=[Patch(facecolor=TEAL,label='Accepted and correct'),Patch(facecolor='#b96748',label='Accepted but wrong'),Patch(facecolor='#e5ebf0',label='Sent for review')],loc='lower left',bbox_to_anchor=(.045,.155),ncol=3,frameon=False,fontsize=12)
    fig.text(.045,.117,'Labels show accepted cases. Grey bars mean no qualifying threshold, or confidence below it.',fontsize=12,color=NAVY)
    fig.text(.045,.043,'Gate: development Wilson 95% lower accuracy bound ≥95%, at least 30 accepted cases; fixed threshold grid.\nSeven priority local alternatives and live Jev. Laya reports spam/phishing in training; exact test overlap unknown.\nA public-data pilot, not permission to delete messages or a security guarantee. 1 October 2026.',fontsize=10.5,color=MUTED,linespacing=1.5)
    save(fig,'confidence-gates')

def gallery():
    import zipfile
    files=[('fresh-local-vs-jev','Accuracy on fresh public test cases'),('business-local-vs-jev','Synthetic business pilot'),
           ('language-local-vs-jev','English and Norwegian intent recognition'),('response-time','Measured response time'),
           ('technology-approaches','Technology approaches'),('confidence-gates','Confidence, review coverage and mistakes'),('business-value-scenario','Illustrative business value')]
    cards=[]
    for name,title in files:
        if not (OUT/f'{name}.png').exists():continue
        cards.append(f'<section><h2>{html.escape(title)}</h2><a href="{name}.png"><img src="{name}.png" alt="{html.escape(title)}"></a><p><a href="{name}.png">PNG</a> · <a href="{name}.svg">Editable vector SVG</a></p></section>')
    sources=[]
    for name in ['laya','laya-multilingual','gliner-decide','julia','von','decider-4b','kev-4b','jevk5','clm-int8']:
        m=read_json(ROOT/f'results/alternatives/models/{name}.json')
        cached=ROOT/'.cache/alternatives_research'/m['repo'].replace('/','--')
        revision=read_json(cached/'api.json')['sha'] if (cached/'api.json').exists() else m['revision']
        sources.append({'model':name,'url':f'https://huggingface.co/{m["repo"]}/blob/{revision}/README.md','sha256':digest(cached/'README.md')})
    write_json(OUT/'technology-sources.json',{'sources':sources,'jev_contract':'https://api.typesafe.ai/openapi.json','note':'Architecture descriptions come from pinned model cards and native code. Business implications are engineering interpretations, not measured performance guarantees.'})
    bundle=['response-time','technology-approaches','confidence-gates','business-value-scenario']
    if all((OUT/f'{name}.png').exists() for name in bundle):
        with zipfile.ZipFile(OUT/'linkedin-companion-graphics.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            for name in bundle:
                for ext in ['png','svg']:archive.write(OUT/f'{name}.{ext}',f'{name}.{ext}')
            archive.write(ROOT/'results/latency/protocol.json','response-time-protocol.json')
            archive.write(ROOT/'results/latency/summary.json','response-time-measurements.json')
            archive.write(OUT/'technology-sources.json','technology-sources.json')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Decision model benchmark graphics</title><style>body{font:16px/1.6 system-ui,sans-serif;background:#f5f7fa;color:#153a55;margin:0}main{max-width:1250px;margin:auto;padding:40px 24px}h1{font-size:36px}section{background:white;padding:22px;margin:24px 0;border:1px solid #dce5eb}img{display:block;width:100%;height:auto}a{color:#166a79}</style><main><h1>Decision model benchmark graphics</h1><p>PNG images for sharing and SVG files for editing. Measurements, technology explanations and illustrative financial assumptions are labelled separately.</p><p><a href="../report.html">Full business report</a> · <a href="../../latency/protocol.json">Response-time protocol</a> · <a href="../../latency/summary.json">Response-time measurements</a> · <a href="technology-sources.json">Technology sources</a></p>'''+''.join(cards)+'</main></html>'
    page=page.replace('<h1>Decision model benchmark graphics</h1>','<h1>Decision model benchmark graphics</h1><p><a href="linkedin-companion-graphics.zip">Download the four companion graphics and supporting data as ZIP</a></p>')
    (OUT/'index.html').write_text(page,encoding='utf-8')

def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    technology();business_value();confidence()
    if (ROOT/'results/latency/protocol.json').exists():latency()
    gallery()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int);args=p.parse_args()
    if args.after_pid:
        import psutil
        try:
            proc=psutil.Process(args.after_pid);assert 'laya_bench.latency_benchmark' in proc.cmdline();proc.wait()
        except psutil.NoSuchProcess:pass
    build()
