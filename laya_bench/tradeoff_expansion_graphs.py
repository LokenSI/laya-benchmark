"""Expanded, task-matched bubble plots; complete timing blocks only."""
import html
import json
import math
import zipfile
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image, ImageOps

from .common import ROOT, read_json, write_json, digest
from .metrics import wilson
from .tradeoff_expansion import OUT as DATA, TIERS, JEV

OUT = ROOT / 'results/linkedin/expanded-tradeoffs'
INK, MUTED, GRID = '#142D3C', '#50616C', '#DCE3E6'
# Stable identifiers avoid unreadable long names at crowded data coordinates.
DETAILS = {
 'laya':('L','Laya'), 'laya-multilingual':('LM','Laya multilingual'),
 'decider-08b':('D.8','Decider 0.8B'), 'decider-2b':('D2','Decider 2B'), 'decider-4b':('D4','Decider 4B v2.1'),
 'von':('V','Von'), 'gliner-decide':('G','GLiNER Decide'), JEV:('J','Jev 1.13.0 API'),
 'julia':('JU','Julia'), 'gliner-decide-multi':('GM','GLiNER multilingual'),
 'gliner-decide-1b':('G1','GLiNER Decide 1B'), 'kev-08b':('K.8','Kev 0.8B'),
 'jevk5-2b':('J52','JevK5 2B'), 'nev-2b':('NE','Nev 2B Q8'), 'imajev-2b':('I2','Imajev 2B'),
 'decision-4b-v12':('DN','Decision 4B v1.2'), 'winnow-12b-q8':('W','Winnow 12B Q8'),
 'cygnet-12b-nf4':('CY','Cygnet 12B NF4'), 'jev-omni-12b-nf4':('OM','Jev-Omni 12B NF4'),
 'clm-int8':('CL','CLM 8B INT8'), 'nimble-9b':('NI','Nimble 9B NF4'),
 'kev-4b':('K4','Kev 4B'), 'intern-decision-4b':('IN','Intern-Decision 4B'),
 'wald-4b-v12':('WA','Wald 4B v1.2'), 'tev1':('TE','Tev1 4B'),
 'plumb-4b':('PL','Plumb 4B'), 'jevk5':('J54','JevK5 4B'), 'imajev-4b':('I4','Imajev 4B'),
}
COMPACT = ['laya','laya-multilingual','julia','von','gliner-decide','gliner-decide-multi','gliner-decide-1b',
           'decider-08b','kev-08b','decider-2b','jevk5-2b','nev-2b','imajev-2b',JEV]
LARGER = [m for m in DETAILS if m not in COMPACT]+[JEV]
PALETTE = ['#057B91','#4C9D97','#9872A7','#7541A3','#364CA2','#B98526','#C26842','#26333D',
           '#548931','#AA5B78','#447BC0','#937243','#49966A','#8B4C43','#6D8690',
           '#386C75','#8761AF','#A66A35','#B34862','#477490','#506D3C','#6F759D',
           '#A76F83','#737534','#406D64','#89506C','#7B6571','#7582AF']
COLORS = dict(zip(DETAILS, PALETTE))
CARDS = []


def evidence():
    report = read_json(DATA/'accuracy-snapshot.json')
    protocol = read_json(DATA/'protocol.json')
    fixture = [json.loads(s) for s in (DATA/'fixture.jsonl').read_text(encoding='utf-8').splitlines()]
    by_id = {r['id']:r for r in fixture}
    assert digest(DATA/'fixture.jsonl') == protocol['fixture_sha256']
    rows=[]; pending=[]
    for model,groups in protocol['models'].items():
        path=DATA/model/'summary.json'
        timing=read_json(path) if path.exists() else {'groups':{}}
        for suite in groups:
            b=timing['groups'].get(suite,{})
            if not b.get('complete'):
                pending.append({'model':model,'suite':suite,'reason':'Timing incomplete'});continue
            raw_path=DATA/model/(suite.replace('/','--')+'.jsonl')
            assert digest(raw_path)==b['predictions_sha256']
            raw=[json.loads(s) for s in raw_path.read_text(encoding='utf-8').splitlines()]
            expected=protocol['groups'][suite]['timing_cases']*3
            assert len(raw)==b['calls']==expected
            assert len({(r['id'],r['pass']) for r in raw})==expected
            assert all(r['input_sha256']==by_id[r['id']]['input_sha256'] for r in raw)
            times=[r['seconds']*1000 for r in raw if not r['error'] and not r['abstained']]
            assert len(times)==b['answered']
            if times:assert abs(float(np.median(times))-b['median_ms'])<1e-7
            src=TIERS if suite=='public_jevbench' else [suite]
            ms=[report['models'][model]['suites'][s] for s in src]
            assert all(v['complete'] for v in ms)
            n=sum(v['n'] for v in ms);hits=sum(v['correct'] for v in ms)
            ci=ms[0].get('cluster_bootstrap95',ms[0]['wilson95']) if len(ms)==1 else wilson(hits,n)
            row={'model':model,'name':DETAILS[model][1],'code':DETAILS[model][0],'suite':suite,
                 'n':n,'correct':hits,'accuracy':hits/n,'interval':ci,
                 'accuracy_failures':sum(v['failures'] for v in ms),
                 **b,'timing_coverage':len(times)/len(raw),
                 'plottable':bool(times) and len(times)/len(raw)>=.9,
                 'config':timing.get('adapter',{}).get('precision','API' if model==JEV else 'native')}
            rows.append(row)
    result={'created':datetime.now(timezone.utc).isoformat(),'protocol':protocol,
            'code_sha256':{name:digest(ROOT/'laya_bench'/name) for name in ['tradeoff_expansion.py','tradeoff_expansion_graphs.py']},
            'accuracy_snapshot_utc':report['updated'],'rows':rows,'pending':pending,
            'plot_rule':'Only complete timing blocks with >=90% answered calls are plotted. Median covers answered calls. Asterisk flags timing refusals/errors/abstentions or native truncation. Excluded blocks and full counts remain in this file.',
            'limits':'Accuracy and timing use different sample sizes. Industry cases are authored and not expert-validated. Public JevBench Wilson intervals are descriptive; related cases and unknown training exposure limit inference. Device-level VRAM deltas include external processes and desktop noise, so do not directly compare bubble areas to the earlier PyTorch-reserved-memory chart.'}
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'measurements.json',result)
    return result


def place_labels(fig, ax, points):
    """Place short labels using display geometry; never shift measured points."""
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    placed=[]
    coords=[ax.transData.transform((p['median_ms'],100*p['accuracy'])) for p in points]
    radii=[math.sqrt((100 if p['model']==JEV else max(8,p['added_device_peak_gib']*66))/math.pi)*fig.dpi/72+4 for p in points]
    bubbles=[Bbox.from_bounds(c[0]-r,c[1]-r,2*r,2*r) for c,r in zip(coords,radii)]
    bounds=ax.get_window_extent().padded(-7)
    for p,origin in sorted(zip(points,coords),key=lambda x:x[0]['median_ms']):
        code=DETAILS[p['model']][0]+('*' if p['timing_coverage']<1 or p['truncated_calls'] else '')
        candidate=ax.text(0,0,code,fontsize=10.5,weight='semibold')
        fig.canvas.draw();bb=candidate.get_window_extent(renderer);candidate.remove()
        w,h=bb.width+8,bb.height+5
        choices=[]
        for radius in [19,29,41,55,71,89]:
            for angle in [45,135,315,225,0,180,90,270,30,150,210,330]:
                theta=math.radians(angle);cx=origin[0]+radius*math.cos(theta);cy=origin[1]+radius*math.sin(theta)
                b=Bbox.from_bounds(cx-w/2,cy-h/2,w,h)
                outside=max(0,bounds.x0-b.x0)+max(0,b.x1-bounds.x1)+max(0,bounds.y0-b.y0)+max(0,b.y1-bounds.y1)
                overlaps=sum(b.overlaps(x) for x in placed)
                near=sum(b.overlaps(circle) for circle in bubbles)
                cost=outside*1000+overlaps*10000+near*800+radius
                choices.append((cost,cx,cy,b))
        _,cx,cy,b=min(choices,key=lambda x:x[0]);placed.append(b)
        offset=((cx-origin[0])*72/fig.dpi,(cy-origin[1])*72/fig.dpi)
        ax.annotate(code,(p['median_ms'],100*p['accuracy']),xytext=offset,textcoords='offset points',
                    ha='center',va='center',fontsize=10.5,weight='semibold',color=INK,
                    bbox={'boxstyle':'square,pad=.12','fc':'white','ec':'none','alpha':.93},
                    arrowprops={'arrowstyle':'-','color':COLORS[p['model']],'lw':.65},zorder=8)


def chart(data, stem, title, panels, models, footnote):
    eligible=[r for r in data['rows'] if r['model'] in models and r['suite'] in [p[0] for p in panels]]
    points=[r for r in eligible if r['plottable']]
    if not all(any(r['suite']==s for r in points) for s,_ in panels):return
    plt.rcParams.update({'font.family':'Segoe UI','text.color':INK,'axes.labelcolor':MUTED,
                         'xtick.color':MUTED,'ytick.color':MUTED,'svg.fonttype':'none'})
    two=len(panels)==2
    fig=plt.figure(figsize=(18,11.5 if two else 15),dpi=160,facecolor='white')
    fig.text(.055,.97,title,fontsize=27,weight='semibold',va='top')
    fig.text(.055,.928 if two else .936,'Accuracy vs median latency  |  Bubble area = added device VRAM during inference  |  Higher and further left is better',fontsize=13,color=MUTED,va='top')
    if two:positions=[[.06,.345,.41,.48],[.56,.345,.41,.48]]
    else:positions=[[.06,.606,.41,.268],[.56,.606,.41,.268],[.06,.282,.41,.268],[.56,.282,.41,.268]]
    allx=[r['median_ms'] for r in points]
    xmin=max(1,min(allx)/1.65);xmax=max(allx)*1.9
    ticks=[v for e in range(-1,6) for v in [10**e,2*10**e,5*10**e] if xmin<=v<=xmax]
    # Use the same axes within a figure; fixed 0–100 accuracy scale across all figures.
    for i,((suite,label),pos) in enumerate(zip(panels,positions)):
        ax=fig.add_axes(pos);ax.set_xscale('log');ax.set_xlim(xmin,xmax);ax.set_ylim(0,110)
        ax.set_xticks(ticks,[str(int(v)) if v>=1 else str(v) for v in ticks]);ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_yticks([0,20,40,60,80,100]);ax.tick_params(length=0,pad=7,labelsize=11)
        ax.set_ylabel('Accuracy (%)',fontsize=12)
        if two or i>=2:ax.set_xlabel('Median response (ms, log scale)',fontsize=12,labelpad=10)
        ax.set_title(label,loc='left',fontsize=18,weight='semibold',pad=13)
        ax.grid(color=GRID,linewidth=.7);ax.set_axisbelow(True)
        for side in ['top','right']:ax.spines[side].set_visible(False)
        for side in ['bottom','left']:ax.spines[side].set_color(GRID)
        subset=[r for r in points if r['suite']==suite]
        for r in sorted(subset,key=lambda x:x['added_device_peak_gib'] or 0,reverse=True):
            model=r['model'];x=r['median_ms'];y=100*r['accuracy'];lo,hi=[100*v for v in r['interval']]
            ax.errorbar(x,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='none',ecolor=COLORS[model],elinewidth=.8,alpha=.32,capsize=2,zorder=2)
            ax.scatter(x,y,s=100 if model==JEV else max(8,r['added_device_peak_gib']*66),
                       marker='D' if model==JEV else 'o',facecolor='white' if model==JEV else COLORS[model],
                       edgecolor=COLORS[model],alpha=1 if model==JEV else .37,linewidth=1.1,zorder=4)
        n=next((r['n'] for r in subset),None)
        ax.text(.99,.026,f'n={n:,} accuracy cases · {len(subset)} models',transform=ax.transAxes,ha='right',fontsize=10,color=MUTED)
        place_labels(fig,ax,subset)
    # Every model with data is listed, including unplotted unsupported blocks.
    visible=[m for m in models if any(r['model']==m for r in eligible)]
    columns=4;step=.026 if not two else .032;start=.217 if not two else .272
    for i,model in enumerate(visible):
        code,name=DETAILS[model];x=.06+(i%columns)*.23;y=start-(i//columns)*step
        ram=[r['added_device_peak_gib'] for r in points if r['model']==model and r['added_device_peak_gib'] is not None]
        usage='hosted: unknown' if model==JEV else ((f'{min(ram):.1f} GiB' if round(min(ram),1)==round(max(ram),1) else f'{min(ram):.1f}–{max(ram):.1f} GiB') if ram else 'timing coverage <90%')
        fig.text(x,y,f'{code}  {name}',fontsize=10.5,color=COLORS[model],weight='semibold',va='top')
        fig.text(x,y-step*.46,usage,fontsize=9,color=MUTED,va='top')
    missing=sum(not r['plottable'] for r in eligible)
    fig.text(.06,.059,'Accuracy: full completed test sets; bars: descriptive 95% intervals. Latency: 3 serial passes on fixed samples; model load and warm-up excluded.',fontsize=9.2,color=MUTED)
    fig.text(.06,.042,'VRAM: NVML device peak minus idle baseline; includes runtime/cache and external servers. Desktop noise is possible. RTX 5070 Ti 16 GB; Windows; batch 1.',fontsize=9.2,color=MUTED)
    fig.text(.06,.025,footnote,fontsize=9.2,color=MUTED)
    fig.text(.06,.008,f'* Timed refusal/error/abstention or native truncation. {missing} completed blocks excluded for <90% answered timing calls. Jev includes network time; hosted VRAM unknown. 1 Oct 2026.',fontsize=9.2,color=MUTED)
    fig.canvas.draw()
    for ext in ['png','svg']:fig.savefig(OUT/f'{stem}.{ext}',facecolor='white')
    plt.close(fig)
    CARDS.append({'stem':stem,'title':title,'panels':[s for s,_ in panels],'models':visible,'plotted_points':len(points),'excluded_blocks':missing})


def package(data):
    cards=''.join(f'<section><h2>{html.escape(c["title"])}</h2><a href="{c["stem"]}.png"><img src="{c["stem"]}.png" alt="{html.escape(c["title"])}: task accuracy versus latency, with bubble area representing incremental device VRAM."></a><p><a href="{c["stem"]}.png">PNG</a> · <a href="{c["stem"]}.svg">Editable SVG</a></p></section>' for c in CARDS)
    values=''
    for r in data['rows']:
        ram='Unknown' if r['added_device_peak_gib'] is None else f'{r["added_device_peak_gib"]:.2f}'
        med='No answered calls' if r['median_ms'] is None else f'{r["median_ms"]:.1f}'
        p95='—' if r['p95_ms'] is None else f'{r["p95_ms"]:.1f}'
        values+=f'<tr><td>{html.escape(r["name"])}</td><td>{html.escape(r["suite"])}</td><td>{r["correct"]}/{r["n"]} ({100*r["accuracy"]:.1f}%)</td><td>{med}</td><td>{p95}</td><td>{ram}</td><td>{r["answered"]}/{r["calls"]}</td><td>{r["truncated_calls"]}</td></tr>'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Expanded model trade-off plots</title><style>body{font:16px/1.6 Segoe UI,Arial,sans-serif;color:#142d3c;margin:30px auto;padding:0 24px;max-width:1500px}img{width:100%;height:auto}a{color:#057b91}table{border-collapse:collapse;font-size:12px}td,th{padding:9px;text-align:left;border-bottom:1px solid #dce3e6}section{padding:20px 0}aside{background:#f0f4f5;padding:20px}</style><h1>Accuracy, latency and VRAM across more use cases</h1><p><a href="expanded-tradeoff-graphs.zip">Download graphs, SVGs and measurements</a> · <a href="measurements.json">Exact measurements</a></p><aside><p>Bubble area now uses added <b>whole-device VRAM measured by NVML</b>, including external model servers. The idle baseline is subtracted separately for each model. This differs from the earlier PyTorch-reserved-memory chart. Desktop memory changes remain a source of noise; values are not a minimum GPU specification.</p><p>Only complete timing blocks with at least 90% answered calls are plotted. Latency describes answered calls; raw failures, abstentions and native truncation remain in the table. Missing full accuracy suites are not estimated. Industrial cases are small authored diagnostics. Model training overlap with public data is unknown.</p></aside>'''+cards+'<h2>Exact values and coverage</h2><table><tr><th>Model</th><th>Task</th><th>Accuracy</th><th>Median ms</th><th>Added VRAM GiB</th><th>Timed answered</th><th>Truncated calls</th></tr>'+values+'</table></html>'
    page=page.replace('<th>Median ms</th>','<th>Median ms</th><th>P95 ms</th>')
    (OUT/'index.html').write_text(page,encoding='utf-8')
    write_json(OUT/'figures.json',CARDS)
    if CARDS:
        thumbnails=[]
        for c in CARDS:
            im=Image.open(OUT/f'{c["stem"]}.png').convert('RGB');im.thumbnail((640,550),Image.Resampling.LANCZOS)
            tile=Image.new('RGB',(660,570),'#edf2f3');tile.paste(im,((660-im.width)//2,(570-im.height)//2));thumbnails.append(tile)
        contact=Image.new('RGB',(1320,570*math.ceil(len(thumbnails)/2)),'#edf2f3')
        for i,im in enumerate(thumbnails):
            x=330 if len(thumbnails)%2 and i==len(thumbnails)-1 else (i%2)*660
            contact.paste(im,(x,(i//2)*570))
        contact.save(OUT/'contact-sheet.jpg',quality=94)
    with zipfile.ZipFile(OUT/'expanded-tradeoff-graphs.zip','w',zipfile.ZIP_DEFLATED) as z:
        files=[OUT/f'{c["stem"]}.{ext}' for c in CARDS for ext in ['png','svg']]
        files += [OUT/name for name in ['index.html','measurements.json','figures.json','contact-sheet.jpg']]
        for p in files:
            if p.is_file():z.write(p,p.name)


def build():
    CARDS.clear()
    data=evidence()
    fresh=[('fresh/news','News classification'),('fresh/emotion','Emotion tagging'),('fresh/spam','Spam screening'),('fresh/phishing','Phishing screening')]
    chart(data,'01-compact-models','Compact models: accuracy, latency and VRAM',fresh,COMPACT,
          'Timing: same 20 short messages/task × 3 passes. Public training overlap unknown; Laya lists news/spam/phishing as training tasks.')
    chart(data,'02-larger-models','Larger models: accuracy, latency and VRAM',fresh,LARGER,
          'Timing: same 20 short messages/task × 3 passes. NF4 / Q8 / INT8 variants labelled. CLM uses warm candidate embeddings and recomputes each state.')
    paired=[m for m in DETAILS if all(s in data['protocol']['models'].get(m,[]) for s in ['massive_en','massive_nb'])]
    chart(data,'03-english-norwegian','English and Norwegian: accuracy, latency and VRAM',
          [('massive_en','English · 18 intent scenarios'),('massive_nb','Norwegian Bokmål · same 18 scenarios')],paired,
          'Timing: 20 matched source IDs/language × 3 passes. Full 2,948-example accuracy per language. Public training overlap unknown.')
    chart(data,'04-public-decisions','Public decision tests: accuracy, latency and VRAM',
          [('public_jevbench','Public JevBench · 231 cases'),('kev_claim/devtools-v1','Developer-tool decisions · 900 cases')],list(DETAILS),
          'Timing: 20 stratified JevBench cases / 12 developer-tool cases × 3 passes. Public JevBench is not the sealed leaderboard or its composite score.')
    chart(data,'05-industry-workflows','Industry workflows: accuracy, latency and VRAM',
          [('industry/routing/en','Ticket routing · English'),('industry/routing/nb','Ticket routing · Bokmål'),
           ('industry/documents/en','Document categories · English'),('industry/documents/nb','Document categories · Bokmål')],list(DETAILS),
          'Small authored diagnostics: 40 routing / 28 document cases per language; not domain-expert validated. Timing: 12 cases/task × 3 passes.')
    package(data)
    print(len(CARDS),'figures;',len(data['rows']),'complete timing blocks;',len(data['pending']),'pending',flush=True)


if __name__=='__main__':build()
