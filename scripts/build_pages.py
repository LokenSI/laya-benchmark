"""Build and validate the public static site using only the standard library."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import zipfile
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
        assert path.suffix in {'.html','.json','.md','.txt','.log','.png','.svg','.jpg','.zip'}
        assert not re.search(r'(predictions|paired_cases|errors\.csv|\.cache|secrets)', source)
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                for entry in archive.infolist():
                    assert Path(entry.filename).name == entry.filename, 'Only flat public graphics bundles are permitted'
                    assert Path(entry.filename).suffix in {'.png','.svg','.jpg','.html','.json'}
                    assert not re.search(r'(predictions|payload|secret|credential)', entry.filename)
        target = output/source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (output/'docs/public-files.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    evidence = read(SHARE+'/evidence.json')
    completion = read('results/completion/summary.json')
    assert completion['sources']['comparison_sha256'] == hashlib.sha256((ROOT/'results/jev_live/comparison.json').read_bytes()).hexdigest(), 'Stale completion report'
    verification = read('results/completion/final-verification.json')
    assert completion['sources']['final_verification_sha256'] == hashlib.sha256((ROOT/'results/completion/final-verification.json').read_bytes()).hexdigest(), 'Stale final verification'
    assert verification['missing_cases'] == completion['remaining_missing'] == 0
    assert verification['remaining_runtime_failures_with_at_least_three_attempts'] == completion['remaining_runtime_failures']
    assert verification['protected_predictions_unchanged'] and verification['temporary_restored_weights_removed']
    assert verification['local_models'] == completion['local_models']
    for name, model in completion['models'].items():
        for fixture, block in model['fixtures'].items():
            assert verification['prediction_sha256'][name][fixture] == block['predictions_sha256'], 'Stale prediction verification'
    for source in evidence['sources'].values():
        assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest() == source['sha256'], \
            f"Stale share-card evidence: {source['path']}; regenerate the business summary"
    from public_overview import render
    document = render(
        read('results/jev_live/comparison.json'),
        read('results/linkedin/completion-tradeoffs/measurements.json'),
        completion, evidence['supervised_baseline_accuracy'])
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
            nav=f'<nav aria-label="Benchmark navigation" style="padding:12px 24px;background:#edf3f3;font:14px system-ui"><a href="{home}">Compare all models by task</a><br>Unofficial, independent testing. Use as is, without warranty or vendor endorsement. Presentation revision: overview-2026-10-05. Evidence revision: completion-2026-10-04. Historical report dates and scope remain as stated.</nav>'
            content=re.sub(r'(<body[^>]*>)',r'\1'+nav,content,count=1,flags=re.I) if '<body' in content else content.replace('<main>',nav+'<main>',1)
        path.write_text(content,encoding='utf-8')
    broken=[]
    for path in output.rglob('*.html'):
        for url in LINK.findall(path.read_text(encoding='utf-8')):
            target=local_target(path,url)
            if target is not None and not target.exists(): broken.append((str(path),url))
    assert not broken, f'Broken public links: {broken}'
    assets={p.relative_to(output).as_posix(): {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in output.rglob('*') if p.is_file() and p.name!='asset-manifest.json'}
    expected_assets=set(files)|{'docs/public-files.json','index.html','.nojekyll'}
    assert set(assets)==expected_assets, f'Unexpected or missing public artifacts: {set(assets)^expected_assets}'
    (output/'asset-manifest.json').write_text(json.dumps({'assets':assets,'local_only_evidence_links':local_only},indent=2)+'\n',encoding='utf-8')
    print(f'Built {len(assets)} public assets; no broken local links; {len(local_only)} raw-evidence links labelled local-only.')
    return output

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT/'.cache/pages-site'))
    build(parser.parse_args().output)
