"""Publishable aggregate completion evidence; no case payloads or credentials."""
import html
import json
from .common import ROOT, read_json, digest
from .completion import audit, OUT, now, prediction_path
from .completion_state import atomic_json, load_records
from .alternatives_report import NAMES

NOTICE = 'Unofficial, independent testing. Results and code are provided as is, without warranty. No vendor endorsement. Validate suitability on your own data before production use.'
FOCUS = ['wald-4b-v12', 'jev-1.13.0', 'decider-4b', 'decision2-nox-4b']
TASKS = [('massive_en', 'English request routing'), ('massive_nb', 'Norwegian Bokmål request routing'),
         ('norec_no', 'Norwegian review sentiment'), ('fresh/news', 'News classification'),
         ('fresh/emotion', 'Emotion classification'), ('fresh/spam', 'Spam screening'),
         ('fresh/phishing', 'Phishing screening'), ('public_jevbench', 'Public JevBench'),
         ('kev_claim/devtools-v1', 'Developer-tool decisions'),
         ('industry/routing/en', 'Industrial ticket routing — English'),
         ('industry/routing/nb', 'Industrial ticket routing — Bokmål'),
         ('industry/documents/en', 'Engineering document categories — English'),
         ('industry/documents/nb', 'Engineering document categories — Bokmål')]


def timing_coverage():
    protocol = read_json(OUT/'timing/protocol.json')
    complete, missing = 0, []
    for name, groups in protocol['models'].items():
        path = OUT/'timing'/name/'summary.json'
        state = read_json(path) if path.exists() else {'groups': {}}
        for suite in groups:
            if state['groups'].get(suite, {}).get('complete'):
                complete += 1
            else:
                reason = state.get('unavailable_groups', {}).get(suite)
                if reason is None and name == 'winnow-12b-q8':
                    reason = 'Historical timing gap; Winnow is outside the approved model restoration and memory-retry queue.'
                missing.append({'model': name, 'suite': suite, 'reason': reason or 'Timing not yet complete'})
    return {'models': len(protocol['models']), 'complete_groups': complete,
            'expected_groups': complete + len(missing), 'missing_groups': missing,
            'protocol_sha256': digest(OUT/'timing/protocol.json')}


def build():
    check = audit()
    verified = read_json(OUT/'final-verification.json')
    assert verified['remaining_runtime_failures_with_at_least_three_attempts'] == check['runtime_failures']
    comparison = read_json(ROOT/'results/jev_live/comparison.json')
    table = []
    for suite, label in TASKS:
        values = {}
        for name in FOCUS:
            suites = ['jevbench_public/easy','jevbench_public/original','jevbench_public/hard'] if suite == 'public_jevbench' else [suite]
            metrics = [comparison['models'][name]['suites'][s] for s in suites]
            if not all(v['complete'] for v in metrics):
                values[name] = {'complete': False}
                continue
            total = sum(v['n'] for v in metrics)
            values[name] = {'complete': True, 'n': total, 'correct': sum(v['correct'] for v in metrics),
                            'accuracy': sum(v['correct'] for v in metrics)/total,
                            'answered': sum(v['valid'] for v in metrics)}
            timing_path = OUT/'timing'/name/'summary.json'
            if timing_path.exists():
                block = read_json(timing_path)['groups'].get(suite, {})
                if block.get('complete'):
                    values[name]['timing'] = {k:block[k] for k in ['calls','answered','median_ms','p95_ms','added_device_peak_gib']}
        table.append({'suite': suite, 'label': label, 'models': values})
    initial = read_json(OUT/'initial-audit.json')
    jev = read_json(OUT/'jev-verification.json')
    recoveries = []
    for manifest in sorted((OUT/'integration-recoveries').glob('*/*.json')):
        recovery = read_json(manifest)
        saved = load_records(prediction_path(recovery['model'], recovery['fixture']))
        applied = sum(saved.get(key, {}).get('inference_path') == recovery['method']
                      for key in recovery['cases'])
        recoveries.append({k: recovery[k] for k in ['model', 'fixture', 'method', 'reason']})
        recoveries[-1].update(requested_cases=len(recovery['cases']), recorded_via_native=applied,
                              manifest_sha256=digest(manifest))
    result = {'revision': 'completion-2026-10-04', 'updated': now(), 'notice': NOTICE,
              'status': 'all_cases_accounted' if check['missing'] == 0 else 'completion_in_progress',
              'cases_per_model': 17576, 'local_models': len(check['models']),
              'initial_missing': initial['missing'], 'remaining_missing': check['missing'],
              'newly_recorded': initial['missing']-check['missing'],
              'initial_runtime_failures': initial['runtime_failures'], 'remaining_runtime_failures': check['runtime_failures'],
              'runtime_failures_recovered': verified['runtime_failures_recovered'],
              'runtime_retry_archives_verified': verified['original_retry_archives_verified'],
              'models': check['models'], 'wald_comparison': table, 'jev_retry': jev,
              'integration_recoveries': recoveries, 'timing_coverage': timing_coverage(),
              'sources': {'comparison_sha256': digest(ROOT/'results/jev_live/comparison.json'),
                          'audit_sha256': digest(OUT/'audit.json'),
                          'final_verification_sha256': digest(OUT/'final-verification.json')},
              'limitations': ['All 17,576 fixture records include development examples; task accuracy excludes development using the original split rules.',
                              'Recorded coverage, answered coverage, native rejection and infrastructure failure are separate.',
                              'Native Windows adapters and pinned precision are retained; no Linux/vLLM or publisher hardware parity claim.',
                              'Public training overlap is unknown. Industrial cases are small authored diagnostics, not domain-expert validation.',
                              'Recovery-worker time is excluded from controlled latency plots. API latency includes network transport.']}
    for recovery in recoveries:
        result['limitations'].append(
            f'{NAMES.get(recovery["model"], recovery["model"])}: '
            f'{recovery["recorded_via_native"]} of {recovery["requested_cases"]} explicitly selected '
            'integration-recovery cases recorded through the publisher native serial question API. '
            'Existing successful predictions and the original parity tolerance are unchanged. '
            'The exact input hashes, original crashes and recovery configuration are retained locally.')
    result['unavailable_timing_groups'] = []
    for path in sorted((OUT/'verified').glob('*-timing.json')):
        timing = read_json(path)
        for suite, reason in timing.get('unavailable_groups', {}).items():
            result['unavailable_timing_groups'].append({'model': timing['model'], 'suite': suite, 'reason': reason})
            result['limitations'].append(f'{NAMES.get(timing["model"], timing["model"])} / {suite}: {reason}')
    historical_gaps = [v for v in result['timing_coverage']['missing_groups'] if v['model'] == 'winnow-12b-q8']
    if historical_gaps:
        result['limitations'].append(
            f'Winnow 12B Q8 retains {len(historical_gaps)} historical timing gaps outside the approved restoration queue. '
            'Its full accuracy results are available; no latency or VRAM values are estimated for the missing groups.')
    atomic_json(OUT/'summary.json', result)
    render(result)
    print(result['status'], result['newly_recorded'], 'new records;', result['remaining_missing'], 'missing', flush=True)
    return result


def render(result):
    """Refresh presentation from the existing verified aggregate without rerunning audits."""
    coverage=''
    for name, model in result['models'].items():
        coverage+='<tr><th>'+html.escape(NAMES.get(name,name))+'</th>'+''.join(f'<td>{model[k]:,}</td>' for k in ['recorded','missing','answered','abstentions','native_rejections','runtime_failures','other_failures'])+'</tr>'
    limits=''.join('<li>'+html.escape(v)+'</li>' for v in result['limitations'])
    document=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shared benchmark completion — revision 2026-10-04</title><style>body{{font:16px/1.6 system-ui,sans-serif;color:#152d3b;max-width:1450px;margin:40px auto;padding:0 24px}}h1{{font-size:38px;line-height:1.15}}h2{{margin-top:40px}}aside{{padding:20px;background:#edf3f3}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:12px 10px;border-bottom:1px solid #d7e0e3;text-align:left;vertical-align:top}}small{{color:#526671}}.scroll{{overflow-x:auto}}a{{color:#187c76}}</style></head><body><p>Evidence revision completion-2026-10-04 · Updated {result['updated'][:10]}<br>Presentation revision overview-2026-10-05 · 5 October 2026</p><h1>Completing the same benchmark for every model</h1><aside>{html.escape(NOTICE)}</aside><p>Each local model is evaluated against the same five frozen fixtures: <strong>17,576 records</strong>. This pass has added <strong>{result['newly_recorded']:,}</strong> previously missing records; <strong>{result['remaining_missing']:,}</strong> remain. Completion retries recovered <strong>{result['runtime_failures_recovered']:,}</strong> recorded memory failures. <strong>{result['remaining_runtime_failures']:,}</strong> remain after at least three retained retry attempts each. Incorrect answers and native rejections are retained.</p><p>Jev 1.13.0 now has 17,576 records with zero API failures after two pinned-version retries. Original attempts are preserved. Existing successful local predictions and fixture hashes are verified against the pre-run baseline.</p><p><a href="../../index.html#leaders">Task leaders across all local models</a> · <a href="../../index.html#compare">Compare every model by accuracy, latency and VRAM</a></p><h2>Controlled timing coverage</h2><p>{result["timing_coverage"]["complete_groups"]} of {result["timing_coverage"]["expected_groups"]} timing groups are complete across {result["timing_coverage"]["models"]} models. Missing groups are listed in the limitations and aggregate evidence. The graphs also withhold complete timing groups with fewer than 90% answered calls.</p><h2>Coverage and failure accounting</h2><p>Recorded means that a case has a validated saved result; it does not mean that the model answered correctly or answered at all. Native rejections include option limits and native context/schema restrictions.</p><div class="scroll"><table><thead><tr><th>Model</th><th>Recorded</th><th>Missing</th><th>Answered</th><th>Abstained</th><th>Native rejection</th><th>Runtime failure</th><th>Other failure</th></tr></thead><tbody>{coverage}</tbody></table></div><h2>Business interpretation</h2><p>Compare models by the actual workflow: suggested ticket categories, engineering document labels, screening or developer-tool selection. These tests support choosing candidates for a reviewed pilot. They do not measure staff-time savings, production safety or financial return.</p><ul>{limits}</ul><p><a href="summary.json">Aggregate completion evidence</a> · <a href="final-verification.json">Final verification</a> · <a href="../linkedin/completion-tradeoffs/index.html">Accuracy, latency and VRAM graphs</a> · <a href="../jev_live/report.html">Full matched API comparison</a></p></body></html>'''
    (OUT/'report.html').write_text(document, encoding='utf-8')


if __name__ == '__main__':
    build()
