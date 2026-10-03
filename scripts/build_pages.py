"""Build and validate the public static site using only the standard library."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SHARE = 'results/share/decision2-2026-10-03'
LINK = re.compile(r'''\b(?:href|src)=["']([^"']+)["']''')
ANCHOR = re.compile(r'''<a\b([^>]*?)href=(["'])(.*?)\2([^>]*)>(.*?)</a>''', re.S | re.I)

def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))

def local_target(path, url):
    value = urlsplit(html.unescape(url))
    if value.scheme or value.netloc or not value.path or '${' in url:
        return None
    return Path(os.path.normpath(path.parent / unquote(value.path)))

def build(output):
    output = Path(output).resolve()
    assert output != ROOT and output not in ROOT.parents, 'Choose a separate build directory'
    manifest = read('docs/public-files.json')
    files = manifest['files']
    assert len(files) == len(set(files))
    for source in files:
        path = ROOT/source
        assert ROOT in path.resolve().parents, f'Public path escapes workspace: {source}'
        assert path.is_file(), f'Missing public source: {source}'
        assert source.startswith(('results/', 'docs/')) or source == 'sources.json'
        assert path.suffix in {'.html','.json','.md','.txt','.log','.png','.svg'}
        assert not re.search(r'(predictions|paired_cases|errors\.csv|\.cache|secrets)', source)
        target = output/source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (output/'docs/public-files.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    evidence = read(SHARE+'/evidence.json')
    for source in evidence['sources'].values():
        assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest() == source['sha256'], \
            f"Stale share-card evidence: {source['path']}; regenerate the business summary"
    rows=''.join(f'<tr><th scope="row">{html.escape(m["model"])}</th><td>{m["deployment"]}</td><td><strong>{m["accuracy"]:.1%}</strong></td></tr>' for m in evidence['metrics'])
    document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local decision AI — benchmark and business value</title><meta name="description" content="English and Norwegian decision-model benchmarks, Decision 2.0 accuracy and load tests, and practical business implications.">
<meta property="og:title" content="Less sorting. More time for the work."><meta property="og:description" content="Local decision-model benchmarks: Norwegian accuracy, practical workflow value, and serving limits.">
<meta property="og:image" content="https://lokensi.github.io/laya-benchmark/SHARE/business-summary.png"><meta property="og:url" content="https://lokensi.github.io/laya-benchmark/">
<style>:root{--ink:#152d3b;--teal:#187c76;--muted:#526671;--pale:#edf3f3}*{box-sizing:border-box}body{margin:0;background:#fcfdfc;color:var(--ink);font:17px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:50px 28px}h1{font-size:clamp(36px,6vw,64px);line-height:1.06;letter-spacing:-2px;max-width:850px;margin:20px 0 26px}h2{font-size:27px;line-height:1.2}h3{font-size:20px;line-height:1.3}p{max-width:850px}a{color:var(--teal);text-underline-offset:4px}a:focus-visible{outline:3px solid #4a789a;outline-offset:5px}.eyebrow{color:var(--teal);font-size:12px;font-weight:650;letter-spacing:1.5px;text-transform:uppercase}.lead{font-size:22px;color:var(--muted);max-width:810px}.actions{display:flex;flex-wrap:wrap;gap:12px;margin:25px 0 40px}.button{display:inline-block;text-decoration:none;border:1px solid var(--teal);padding:10px 17px;border-radius:6px}.primary{background:var(--teal);color:white}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:30px 0}.card{padding:24px;background:var(--pale);border-radius:10px}.card p{font-size:16px}.twocol{display:grid;grid-template-columns:1.1fr 1fr;gap:40px;margin:50px 0;align-items:start}img{display:block;width:100%;height:auto;border:1px solid #d7e0e3}table{width:100%;border-collapse:collapse;font-size:15px}td,th{text-align:left;padding:14px 8px;border-bottom:1px solid #d7e0e3}small,.muted{color:var(--muted)}section{margin:55px 0}footer{padding-top:26px;border-top:1px solid #d7e0e3;font-size:14px}.reports{display:grid;grid-template-columns:1fr 1fr;gap:18px}.reports a{padding:20px;background:var(--pale);text-decoration:none}.reports strong{display:block;font-size:19px}.reports span{display:block;font-size:14px;color:var(--muted);margin-top:7px}@media(max-width:740px){main{padding:30px 20px}.grid,.twocol,.reports{grid-template-columns:1fr}.twocol{gap:24px}table{font-size:13px}td,th{padding:10px 5px}h1{letter-spacing:-1px}}</style></head><body><main>
<div class="eyebrow">Measured locally · Updated 3 October 2026</div><h1>Less sorting.<br>More time for the work.</h1>
<p class="lead">Decision models can suggest categories, route requests and tag documents. We tested their accuracy in English and Norwegian, then checked what happens under load.</p>
<div class="actions"><a class="button primary" href="results/alternatives/decision2.html">New: Decision 2.0 results</a><a class="button" href="results/jev_live/report.html">Full business comparison</a><a class="button" href="https://github.com/LokenSI/laya-benchmark">Code &amp; methodology on GitHub</a></div>
<div class="grid"><article class="card"><h3>Keep processing local</h3><p>After downloading the models, classification can run in your environment. That can be useful when messages should stay inside it.</p></article><article class="card"><h3>Help staff handle the queue</h3><p>Start with category suggestions that people review. Measure handling time, corrections and the consequences of mistakes.</p></article><article class="card"><h3>Test the simpler option</h3><p>A trained classifier still led on Norwegian routing when labelled training data was available. Choose for the workflow, then measure the value.</p></article></div>
<section class="twocol"><div><h2>How well did Norwegian routing work?</h2><p>Same 2,948 public Bokmål test inputs across 18 MASSIVE assistant-service scenarios. These are a proxy for routing, not a test of a company's support inbox.</p><table><thead><tr><th>Model</th><th>Deployment</th><th>Accuracy</th></tr></thead><tbody>ROWS</tbody></table><p class="muted">The supervised TF-IDF + linear SVM baseline reached SUPERVISED. It used official labelled training data; the models in this table were evaluated without task-specific fitting.</p><p>Nox's score is close to the tested hosted API's score on these cases. This is not proof of equivalent performance. Roughly one in five Nox suggestions was wrong, so human review remains part of the proposed pilot.</p></div><div><a href="SHARE/business-summary.png"><img src="SHARE/business-summary.png" width="1080" height="1350" alt="Business summary comparing Norwegian routing accuracy and proposing a reviewed local AI pilot."></a><p><a href="SHARE/business-summary.png" download>Download the share PNG</a> · <a href="SHARE/alt-text.txt">Image alt text</a></p></div></section>
<section><h2>The new Decision 2.0 run</h2><p>Kai 0.6B, Eos 0.8B, Sol 2B and Nox 4B each completed 17,576 standard fixture records: <strong>70,304 saved predictions</strong> in total. Independent verification found zero final saved prediction errors. Four interrupted process attempts required resumable retries; their history is preserved. Incorrect model answers are separate from execution errors.</p><p>Lux 9B and Vega 27B were deferred for the local 16 GB GPU budget. Public training-data overlap is unknown. Authored business examples are diagnostics and are not the headline accuracy sample.</p><p><a href="results/alternatives/decision2-verification.json">Result verification and interruption history</a> · <a href="results/alternatives/decision2-summary.json">Complete Decision 2.0 aggregate metrics</a></p></section>
<section><h2>Concurrency adds a serving requirement</h2><p>Eos and Sol completed <strong>9,600 timed localhost HTTP requests</strong>, with no request errors or changes in predicted labels relative to the serial reference. At 512 clients, p95 response time reached roughly 34–39 seconds. More submitted work did not mean proportional throughput in this one-executor FIFO setup.</p><p>Both models handled 128 questions in one request; 512 questions exhausted the configured 78% VRAM allocation budget. That test is different from 512 concurrent clients. The complete published Sol package could not load in stock <strong>vLLM 0.30.0</strong> under either tested implementation because its forward interface did not meet the required contract.</p><p>Business implication: plan response-time limits and serving integration before promising interactive scale. These results describe the tested native deployment; they do not reproduce a vendor production serving stack.</p><p><a href="results/decision2-stress/report.html">Read the stress protocol, measurements and vLLM failures</a></p></section>
<section><h2>Explore the evidence</h2><div class="reports"><a href="results/alternatives/decision2.html"><strong>Decision 2.0 accuracy</strong><span>Four native models across the standard frozen fixtures.</span></a><a href="results/decision2-stress/report.html"><strong>Concurrency &amp; compatibility</strong><span>Native HTTP queue measurements, question capacity and direct vLLM probes.</span></a><a href="results/jev_live/report.html"><strong>Local models vs hosted API</strong><span>Matched tasks, confidence gates and business workflow examples. Updated with Decision 2.0.</span></a><a href="results/alternatives/report.html"><strong>Full local comparison</strong><span>English, Norwegian, public task groups and typed decisions.</span></a><a href="results/full/report.html"><strong>Original Laya business report</strong><span>Baseline quality, calibration and an editable, illustrative NOK cost model.</span></a><a href="results/alternatives/claim_audit.html"><strong>Published claim audit</strong><span>What was reproduced, what differed and what could not be tested.</span></a></div></section>
<footer>Measured on an RTX 5070 Ti 16 GB. Public benchmark scores do not establish production accuracy or financial savings. Raw datasets, model weights, credentials and per-case payloads are excluded from this public site. <a href="docs/METHODOLOGY.md">Methods</a> · <a href="docs/PUBLICATION.md">Publication scope</a> · <a href="SHARE/evidence.json">Share-card evidence hashes</a> · <a href="asset-manifest.json">Published asset hashes</a></footer>
</main></body></html>'''
    document = document.replace('ROWS', rows).replace('SUPERVISED', f"{evidence['supervised_baseline_accuracy']:.1%}").replace('SHARE', SHARE)
    (output/'index.html').write_text(document, encoding='utf-8')
    (output/'.nojekyll').write_text('', encoding='utf-8')
    # This generated link must exist while the HTML link checker runs.
    (output/'asset-manifest.json').write_text('{}\n', encoding='utf-8')
    local_only = []
    for path in output.rglob('*.html'):
        content=path.read_text(encoding='utf-8')
        def fix(match):
            url=match.group(3)
            target=local_target(path, url)
            if target is not None and not target.exists():
                local_only.append({'report':path.relative_to(output).as_posix(),'local_evidence':url})
                return '<span title="Raw evidence remains local; see publication scope">'+match.group(5)+' (local-only evidence)</span>'
            return match.group(0)
        content=ANCHOR.sub(fix,content)
        if path.name!='index.html' or path.parent!=output:
            home=os.path.relpath(output/'index.html',path.parent).replace('\\','/')
            nav=f'<nav aria-label="Benchmark navigation" style="padding:12px 24px;background:#edf3f3;font:14px system-ui"><a href="{home}">← Benchmark home &amp; latest Decision 2.0 results</a></nav>'
            content=re.sub(r'(<body[^>]*>)',r'\1'+nav,content,count=1,flags=re.I) if '<body' in content else content.replace('<main>',nav+'<main>',1)
        path.write_text(content,encoding='utf-8')
    broken=[]
    for path in output.rglob('*.html'):
        for url in LINK.findall(path.read_text(encoding='utf-8')):
            target=local_target(path,url)
            if target is not None and not target.exists(): broken.append((str(path),url))
    assert not broken, f'Broken public links: {broken}'
    assets={p.relative_to(output).as_posix(): {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in output.rglob('*') if p.is_file() and p.name!='asset-manifest.json'}
    (output/'asset-manifest.json').write_text(json.dumps({'assets':assets,'local_only_evidence_links':local_only},indent=2)+'\n',encoding='utf-8')
    print(f'Built {len(assets)} public assets; no broken local links; {len(local_only)} raw-evidence links labelled local-only.')
    return output

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT/'.cache/pages-site'))
    build(parser.parse_args().output)
