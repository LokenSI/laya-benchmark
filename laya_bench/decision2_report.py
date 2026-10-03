"""Scoped Decision 2.0 results from complete frozen suites and saved predictions."""
import html
from .common import ROOT, read_json, write_json, digest
from .alternatives_report import lines, native_jev, evaluate, NAMES
from .decision2_prepare import MODELS
from .jev_api import now

OUT = ROOT / 'results/alternatives'

def build():
    fixture = ROOT / 'data/prepared/alternatives_claims.jsonl'
    assert digest(fixture) == read_json(OUT / 'alternatives_claims-protocol.json')['fixture_sha256']
    public = [row for row in lines(fixture) if row['suite'].startswith('jevbench_public/')]
    assert len(public) == 231
    result = {'updated': now(), 'scope': '231 public JevBench cases; original scoring. Separate from the sealed leaderboard and from full English/Norwegian evaluation.',
              'fixture_sha256': digest(fixture), 'models': {},
              'deferred': ['Decision 2.0 Lux 9B and Vega 27B: native precision exceeds local GPU budget'],
              'limitations': ['Windows NVIDIA CUDA/Transformers/FLA environment differs from publisher ROCm latency environment.',
                             'Public benchmark training contamination and model-selection exposure are not independently established.',
                             'No model weights, prompts or test labels were tuned during this run.']}
    verification = OUT / 'decision2-verification.json'
    if verification.exists():
        verified = read_json(verification)
        current = True
        for name, fixtures in verified['models'].items():
            for kind, saved in fixtures.items():
                folder = 'runs' if kind == 'alternatives' else kind
                predictions = OUT / folder / name / 'predictions.jsonl'
                if not predictions.exists() or digest(predictions) != saved['predictions_sha256']:
                    current = False
        if current:
            result['verification'] = verified
    for name in [*MODELS, 'decision-4b-v12']:
        metadata = OUT / f'models/{name}.json'
        if not metadata.exists():
            continue
        all_predictions = lines(OUT / f'alternatives_claims/{name}/predictions.jsonl')
        wanted = {row['id']: row for row in public}
        predictions = [p for p in all_predictions if p['id'] in wanted]
        assert len(predictions) == len({p['id'] for p in predictions})
        for prediction in predictions:
            assert prediction['input_sha256'] == wanted[prediction['id']]['input_sha256']
        tiers = native_jev(public, predictions)
        entry = {'label': NAMES[name], 'source': 'https://huggingface.co/' + read_json(metadata)['repo'],
                 'revision': read_json(metadata)['revision'], 'public231': {
                     'expected': 231, 'attempted': len(predictions), 'complete': len(predictions) == 231,
                     'correct': sum(tier['correct'] for tier in tiers.values()),
                     'errors': sum(bool(p.get('error')) for p in predictions), 'tiers': tiers}, 'shared_suites': {}}
        entry['public231']['accuracy'] = entry['public231']['correct'] / 231 if len(predictions) == 231 else None
        for kind in ['jev_verified', 'jev_fresh', 'alternatives', 'alternatives_typed']:
            path = ROOT / f'data/prepared/{kind}.jsonl'
            protocol = OUT / ('protocol.json' if kind == 'alternatives' else f'{kind}-protocol.json')
            assert digest(path) == read_json(protocol)['fixture_sha256']
            rows = lines(path)
            runroot = 'runs' if kind == 'alternatives' else kind
            saved = lines(OUT / f'{runroot}/{name}/predictions.jsonl')
            scored_rows = [row for row in rows if row.get('split') != 'development'] if kind == 'jev_fresh' else rows
            entry['shared_suites'][kind] = {'expected': len(rows), 'attempted': len(saved),
                                           'evaluation_expected': len(scored_rows),
                                           'suites': evaluate(scored_rows, saved)}
        result['models'][name] = entry
    write_json(OUT / 'decision2-summary.json', result)
    esc = html.escape
    rows = []
    for name, entry in result['models'].items():
        public_score = entry['public231']
        accuracy = f"{public_score['accuracy']:.1%}" if public_score['complete'] else 'Pending'
        rows.append(f"<tr><td><a href='{esc(entry['source'])}'>{esc(entry['label'])}</a></td>"
                    f"<td>{public_score['attempted']}/231</td><td>{accuracy}</td><td>{public_score['errors']}</td></tr>")
    shared = []
    execution = ''
    if 'verification' in result:
        verified = result['verification']
        execution = f"<p>All four native models completed all five frozen fixtures: {verified['saved_case_predictions']:,} saved case predictions, {verified['saved_prediction_errors']} final saved prediction errors. Windows allocation/kernel failures interrupted {len(verified['runtime_events'])} process attempts; exact-runtime retries resumed saved IDs. <a href='decision2-verification.json'>Independent result verification and crash history</a>.</p>"
    for entry in result['models'].values():
        if not entry['label'].startswith('Decision 2.0'):
            continue
        for fixture_name, fixture_result in entry['shared_suites'].items():
            for suite, score in fixture_result['suites'].items():
                if score['complete']:
                    shared.append(f"<tr><td>{esc(entry['label'])}</td><td>{esc(suite)}</td><td>{score['attempted']}</td><td>{score['accuracy']:.1%}</td><td>{score['coverage']:.1%}</td></tr>")
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Decision 2.0 benchmark</title><style>body{{font:16px system-ui;background:#f4f6f9;color:#16283a;margin:0}}main{{max-width:1120px;margin:40px auto;padding:28px;background:white;border-radius:14px}}table{{border-collapse:collapse;width:100%;margin:24px 0}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #d9e1ea}}a{{color:#17699a}}p{{line-height:1.6}}small{{color:#596c7d}}</style>
<main><h1>Decision 2.0: local benchmark</h1><small>Updated {esc(result['updated'])}</small>
<p>{esc(result['scope'])} Results use the same pinned inputs and retain native rejections as failures. Scores appear only when the complete suite has been attempted.</p>
{execution}
<h2>Public JevBench comparison</h2><table><tr><th>Model</th><th>Cases attempted</th><th>Original-score accuracy</th><th>Inference errors</th></tr>{''.join(rows)}</table>
<p>Decision 4B v1.2 is the separate FlyMy model previously benchmarked here.</p>
<h2>Completed shared suites</h2><table><tr><th>Model</th><th>Suite</th><th>Cases</th><th>Exact-match accuracy</th><th>Coverage</th></tr>{''.join(shared) or '<tr><td colspan="5">Pending</td></tr>'}</table>
<p>Fresh suite accuracies exclude development records. Score classes use the modal level; native expected scores are retained in raw results. Multilabel tasks use separate native Yes/No questions. Authored Norwegian/business suites are diagnostics.</p>
<p>{esc(' '.join(result['limitations']))} {esc(' '.join(result['deferred']))}</p>
<p><a href="queue-decision2.json">Initial execution</a> · <a href="queue-decision2-retry.json">Resumable retries</a> · <a href="decision2-summary.json">Result summary</a> · <a href="../decision2-stress/report.html">Concurrency and vLLM compatibility</a> · <a href="report.html">Full comparison</a></p></main></html>'''
    (OUT / 'decision2.html').write_text(document, encoding='utf-8')
    print({name: entry['public231'] for name, entry in result['models'].items()}, flush=True)
    return result

if __name__ == '__main__':
    build()
