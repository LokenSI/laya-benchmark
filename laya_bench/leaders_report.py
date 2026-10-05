"""Business-facing eligibility, execution and public-claim audit for added leaders."""
import html
from datetime import datetime, timezone
from .common import ROOT, read_json, write_json, digest
from .completion import FIXTURES

OUT = ROOT / 'results/alternatives'
ADDED = [
    {'id': 'decision-4b-v12', 'name': 'Decision 4B v1.2', 'license': 'Apache 2.0 code, adapter and base',
     'source': 'https://huggingface.co/flymy-ai/decision-4b-v1.2', 'fit': '4B BF16; native verified package, eager GPU',
     'claim_correct': [203], 'claim_n': 231, 'claim_scope': 'Publisher merged-model public JevBench result. CUDA graphs disabled locally.',
     'business': 'Ticket and document routing; compare accuracy and calibrated review thresholds.'},
    {'id': 'winnow-12b-q8', 'name': 'Winnow-12B Q8', 'license': 'Apache 2.0 weights; MIT inference engine',
     'source': 'https://huggingface.co/EldanRing/Winnow-12B', 'fit': 'Publisher Q8 GGUF; native CUDA engine ported to Windows',
     'code': 'https://github.com/EldanRing/winnow-inference', 'claim_correct': [198], 'claim_n': 231,
     'claim_scope': 'Published Q8 public-set accuracy; local Windows/context settings differ. Vision and 64K capacity are outside this text evaluation.',
     'business': 'Potentially stronger local decisions where data residency and offline operation matter.'},
    {'id': 'cygnet-12b-nf4', 'name': 'Cygnet 12B NF4 — local variant', 'license': 'MIT recipe; Apache 2.0 Gemma weights',
     'source': 'https://github.com/blockbrain-ai/cygnet-recipe', 'fit': '4-bit NF4 / Transformers adaptation for 16 GB',
     'claim_correct': [203, 204], 'claim_n': 231,
     'claim_scope': 'Publisher BF16/vLLM claim is a reference only. Quantization and the local letter-readout backend differ; this run cannot verify or refute the exact published configuration.',
     'business': 'Tests whether an untuned general model with constrained decisions is competitive locally.'},
    {'id': 'jev-omni-12b-nf4', 'name': 'Jev-Omni 12B NF4 — local variant', 'license': 'Apache 2.0 code and weights',
     'source': 'https://huggingface.co/akhilaaa3/Jev-Omni', 'fit': '4-bit backbone; original FP32 classifier head',
     'claim_correct': [202], 'claim_n': 231,
     'claim_scope': 'Publisher micro accuracy is a BF16 reference; local NF4 text-only results cannot verify or refute that configuration or multimodal claims.',
     'business': 'Tests the value of a learned classifier head on a larger local backbone.'},
]


def shared_coverage(comparison, model):
    available = comparison['models'].get(model, {}).get('fixtures', {})
    groups = [available[name] for name in FIXTURES if name in available]
    complete = len(groups) == len(FIXTURES) and all(g['complete'] for g in groups)
    return {'complete': complete, 'recorded': sum(g['attempted'] for g in groups),
            'saved_failures': sum(g['errors'] for g in groups)}


def build():
    from .alternatives_report import lines, native_jev
    comparison = read_json(OUT/'comparison.json')
    rows = lines(ROOT / 'data/prepared/alternatives_claims.jsonl')
    assert sum(r['suite'].startswith('jevbench_public/') for r in rows)==231, 'Wrong public JevBench fixture'
    source = ROOT / '.cache/leader_research/jevbench-v1.5.4.json'
    result = {'updated': datetime.now(timezone.utc).isoformat(), 'leaderboard_release': 'v1.5.4',
              'source_url': 'https://benchmarkheaven.com/jev-models',
              'source_sha256': digest(source) if source.exists() else None,
              'eligibility': 'Publicly licensed inference code and model weights/adapters; this does not assert that all training data is released.',
              'selection': 'Union of top ten overall and top ten Jev-class capability entries, with duplicate versions identified.',
              'added': [],
              'existing': [
                  {'id': 'jevk5', 'name': 'JevK5 v0.3', 'status': 'Pinned checkpoint verified as v0.3; scoped native recovery disclosed in the completion report.'},
                  {'id': 'plumb-4b', 'name': 'Plumb-4B', 'status': 'Native readout/calibration retained; scoped native recovery disclosed in the completion report.'},
                  {'id': 'imajev-4b', 'name': 'Imajev-4B', 'status': 'Publisher abstentions remain unanswered.'},
                  {'id': 'decider-4b', 'name': 'Decider 4B', 'status': 'Local v2.1 differs from the leaderboard v2 checkpoint.'}],
              'deferred': [
                  {'name': 'Surogate Rune 26B-A4B v3', 'license': 'Apache 2.0', 'source': 'https://huggingface.co/surogate/rune-26b-a4b-GGUF',
                   'reason': 'Official v3 weights are gated and require contact-information sharing; BF16 weights alone are about 51.6 GB. No gate accepted and no external compute provisioned.'},
                  {'name': 'djev / DiffusionGemma', 'license': 'Apache 2.0 code and weights', 'source': 'https://github.com/Davipar/djev-dev',
                   'reason': 'Reference is a 26B BF16 model with a patched Linux vLLM runtime; weights exceed the 16 GB GPU. No faithful supported 16 GB path established.'},
                  {'name': 'Decision 4B v1.1', 'license': 'Apache 2.0', 'source': 'https://huggingface.co/flymy-ai/decision-4b-v1.1',
                   'reason': 'Older version of the same family; v1.2 selected to avoid duplicating the first expansion.'}],
              'closed': 'No new closed-source systems added. Existing paid Jev results remain the comparison reference.'}
    for item in result['existing']:
        item['coverage'] = shared_coverage(comparison, item['id'])
        if item['coverage']['complete']:
            item['status'] = f"All {item['coverage']['recorded']:,} shared cases recorded. " + item['status']
    jobs = []
    for state_name in ['queue-leaders-priority', 'queue-leaders-full']:
        path = OUT / f'{state_name}.json'
        if path.exists():
            state = read_json(path)
            jobs.extend(state.get('jobs', []))
            result[state_name] = state
    result['queue_history_note'] = 'Earlier queue states are archival only; current coverage comes from saved fixture results.'
    for spec in ADDED:
        item = dict(spec)
        meta = OUT / f'models/{spec["id"]}.json'
        smoke = OUT / f'leader-smoke/{spec["id"]}.json'
        item['checkpoint_ready'] = meta.exists()
        item['integration_passed'] = smoke.exists() and read_json(smoke).get('status') == 'interface_passed'
        item['status'] = 'Integration passed; queued/running evaluation' if item['integration_passed'] else 'Queued; GPU integration pending' if meta.exists() else 'Downloading pinned weights'
        mine = [job for job in jobs if job['model'] == spec['id']]
        if mine and mine[-1]['status'] == 'needs_repair':
            item['status'] = 'Integration or execution needs repair; no score inferred'
        pred = lines(OUT / f'alternatives_claims/{spec["id"]}/predictions.jsonl')
        tiers = native_jev(rows, pred)
        n = sum(t['n'] for t in tiers.values())
        correct = sum(t['correct'] for t in tiers.values())
        item['public_jevbench'] = {'complete': n == 231, 'attempted': n, 'correct': correct if n == 231 else None, 'tiers': tiers}
        if n == 231:
            item['public_jevbench']['accuracy'] = correct / n
            item['public_jevbench']['matches_published_count'] = correct in spec['claim_correct']
            item['status'] = 'Public JevBench complete; see other suite progress'
        item['shared_coverage'] = shared_coverage(comparison, spec['id'])
        if item['shared_coverage']['complete']:
            item['status'] = (f"All {item['shared_coverage']['recorded']:,} shared cases recorded; "
                              f"{item['shared_coverage']['saved_failures']:,} saved failures or native rejections. "
                              'Answered coverage is reported separately.')
        result['added'].append(item)
    write_json(OUT / 'leader-expansion.json', result)
    esc = html.escape
    table = ''.join(f'<tr><td><a href="{esc(x["source"])}">{esc(x["name"])}</a><small>{esc(x["license"])}</small></td><td>{esc(x["fit"])}</td><td>{esc(x["status"])}</td></tr>' for x in result['added'])
    existing = ''.join(f'<li><strong>{esc(x["name"])}</strong>: {esc(x["status"])}</li>' for x in result['existing'])
    deferred = ''.join(f'<li><a href="{esc(x["source"])}">{esc(x["name"])}</a> ({esc(x["license"])}): {esc(x["reason"])}</li>' for x in result['deferred'])
    claims = ''.join(f'<tr><td>{esc(x["name"])}</td><td>{" or ".join(map(str,x["claim_correct"]))}/231</td><td>{str(x["public_jevbench"]["correct"])+"/231" if x["public_jevbench"]["complete"] else "Pending — no partial score"}</td><td>{esc(x["claim_scope"])}</td></tr>' for x in result['added'])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Open-source leaderboard expansion</title>
<style>body{{font:16px/1.55 Segoe UI,Arial,sans-serif;background:#f3f6f8;color:#162b3b;margin:0}}main{{max-width:1180px;margin:auto;padding:52px 32px}}h1{{font-size:38px;letter-spacing:-1px}}h2{{font-size:22px}}section{{background:white;padding:24px;margin:24px 0;border:1px solid #dce4ea}}table{{border-collapse:collapse;width:100%}}td,th{{padding:13px 10px;border-bottom:1px solid #e1e8ed;text-align:left;vertical-align:top}}th,small{{font-size:13px;color:#546979}}small{{display:block}}a{{color:#17657d}}li{{margin:12px 0}}.tag{{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#546979}}.note{{border-left:4px solid #3b7a8c;padding-left:18px}}</style>
<main><div class="tag">Local software evaluation · {result["updated"][:10]} · Unofficial testing</div><h1>Testing the open-source leaders</h1><p class="note">Four additions to the shared comparison. Eligibility is verified separately from execution. An available checkpoint, successful build, or passing interface probe is not an accuracy result.</p>
<p>Selection uses <a href="https://benchmarkheaven.com/jev-models">JevBench v1.5.4</a>, saved with SHA256 <code>{result['source_sha256']}</code>. Both overall and capability rankings were checked.</p>
<section><h2>Models and verified coverage</h2><table><tr><th>Model and license</th><th>Local configuration</th><th>Execution status</th></tr>{table}</table><p>Updated {esc(result['updated'])}. The runtime uses Python venvs and one GPU job at a time. Winnow uses a compiled native CUDA server controlled by the Python runner.</p></section>
<section><h2>Fair comparison protocol</h2><p>The first checks cover 231 public JevBench cases plus the existing 300-case exact pilot, followed by the same frozen fresh English and Norwegian business/task fixtures used with live Jev. The shared classification, industry and typed-decision suites use the same five frozen fixtures: 17,576 records per model. Recorded failures and native rejections remain visible. Inputs and source hashes stay fixed; unsupported requests and model errors remain in denominators.</p><p>NF4 variants are labelled separately. They test practicality on this GPU and cannot establish that the published BF16 configurations reproduce or fail their claims. Confidence thresholds must come from development data. Batch run times are not the separate controlled response-time experiment.</p><p><a href="../jev_live/report.html">Live Jev comparison</a> · <a href="report.html">Full benchmark progress</a> · <a href="leader-expansion.json">Structured evidence and status</a> · <a href="leader-license-audit.json">Detailed publisher license audit</a></p></section>
<section><h2>Public accuracy claims</h2><table><tr><th>Model</th><th>Published correct</th><th>Local correct</th><th>Scope</th></tr>{claims}</table><p>The sealed leaderboard is not reproduced here. Public answer keys are source-verified; they have not been independently relabelled by domain experts.</p></section>
<section><h2>Already covered</h2><ul>{existing}</ul></section><section><h2>Deferred</h2><ul>{deferred}</ul><p>{esc(result['closed'])}</p></section>
<section><h2>What this can establish for a business</h2><p>We can compare routing and document decisions, Norwegian reliability, review coverage and error types under the same inputs. We cannot infer production safety or return on investment from a leaderboard position. Industrial examples remain small authored diagnostics until a domain expert reviews them.</p></section></main></html>'''
    (OUT / 'leader-expansion.html').write_text(page, encoding='utf-8')
    return result


if __name__ == '__main__':
    build()
