"""Render the all-model landing page from committed aggregate evidence only."""
import html
import json
from pathlib import Path

JEV = 'jev-1.13.0'
REVISION = 'overview-2026-10-05'
FEATURED = {
    'massive_en': 'English request routing',
    'massive_nb': 'Norwegian request routing',
    'norec_no': 'Norwegian review sentiment',
    'fresh/news': 'News classification',
    'fresh/emotion': 'Emotion classification',
    'fresh/spam': 'Spam screening',
    'fresh/phishing': 'Phishing screening',
    'public_jevbench': 'Public JevBench',
    'kev_claim/devtools-v1': 'Developer-tool selection',
}
INDUSTRY = {
    'industry/routing/en': 'Industrial ticket routing — English',
    'industry/routing/nb': 'Industrial ticket routing — Norwegian',
    'industry/documents/en': 'Engineering document categories — English',
    'industry/documents/nb': 'Engineering document categories — Norwegian',
}
TIERS = ['jevbench_public/easy', 'jevbench_public/original', 'jevbench_public/hard']


def metrics(model, suite):
    """Keep failures in the denominator; combine only the three JevBench tiers."""
    blocks = [model['suites'][key] for key in (TIERS if suite == 'public_jevbench' else [suite])]
    if not all(b['complete'] and b['n'] == b['expected'] == b['attempted'] for b in blocks):
        return None
    n = sum(b['n'] for b in blocks)
    correct = sum(b['correct'] for b in blocks)
    answered = sum(b['valid'] for b in blocks)
    assert 0 <= correct <= answered <= n and n > 0
    return {'n': n, 'correct': correct, 'answered': answered, 'accuracy': correct / n}


def leaders(rows, field, highest=False):
    eligible = [r for r in rows if r['id'] != JEV and r.get(field) is not None]
    if not eligible:
        return []
    best = (max if highest else min)(r[field] for r in eligible)
    return [r for r in eligible if r[field] == best]


def assemble(comparison, measurements, completion):
    models = comparison['models']
    assert set(models) == set(completion['models']) | {JEV}
    assert measurements['accuracy_snapshot_utc'] == comparison['updated'], 'Stale timing accuracy snapshot'
    suites = set(models[JEV]['suites'])
    assert all(set(m['suites']) == suites for m in models.values()), 'Different model task sets'
    names = {key: model['label'] for key, model in models.items()}
    names.update({JEV: 'Jev 1.13.0 API', 'decider-4b': 'Decider 4B v2.1'})
    timings = {(r['model'], r['suite']): r for r in measurements['rows']}
    assert len(timings) == len(measurements['rows']), 'Duplicate timing group'
    tasks = []
    order = [*FEATURED, *INDUSTRY, *sorted(suites - set(FEATURED) - set(INDUSTRY))]
    for suite in order:
        rows = []
        for key, model in models.items():
            metric = metrics(model, suite)
            row = {'id': key, 'accuracy': None, 'median_ms': None, 'vram_gib': None,
                   'timing_note': 'No controlled measurement for this task.'}
            if metric:
                row.update(metric)
            timing = timings.get((key, suite))
            if timing:
                assert metric and timing['n'] == metric['n'] and timing['correct'] == metric['correct'], 'Timing/accuracy mismatch'
                assert timing['complete']
                coverage = timing['answered'] / timing['calls']
                row['timing_note'] = (f"{timing['answered']}/{timing['calls']} timing calls answered; "
                                      f"{timing.get('truncated_calls', 0)} calls with native truncation.")
                if coverage >= .9 and timing['median_ms'] is not None:
                    row['median_ms'] = timing['median_ms']
                    row['vram_gib'] = timing['added_device_peak_gib'] if key != JEV else None
                else:
                    row['timing_note'] += ' Withheld: fewer than 90% answered timing calls.'
            rows.append(row)
        assert len({r['n'] for r in rows if 'n' in r}) == 1, 'Different accuracy denominators'
        rows.sort(key=lambda r: (-(r['accuracy'] if r['accuracy'] is not None else -1), names[r['id']]))
        tasks.append({'id': suite, 'label': FEATURED.get(suite, INDUSTRY.get(suite, suite.replace('/', ' / ').replace('_', ' '))),
                      'group': 'Main public comparisons' if suite in FEATURED else
                               'Authored industry diagnostics' if suite in INDUSTRY else 'Other published tasks and diagnostics',
                      'rows': rows})
    return {'revision': REVISION, 'names': names, 'tasks': tasks, 'api': JEV}


def leaderboard(data):
    rows = []
    for task in data['tasks'][:len(FEATURED)]:
        winners = leaders(task['rows'], 'accuracy', highest=True)
        reference = next(r for r in task['rows'] if r['id'] == JEV)
        name = ' / '.join(data['names'][r['id']] for r in winners)
        score = f"{winners[0]['accuracy']:.1%}" if winners else 'Unavailable'
        api_score = f"{reference['accuracy']:.1%}" if reference['accuracy'] is not None else 'Unavailable'
        rows.append(f'<tr><th scope="row">{html.escape(task["label"])}</th><td>{html.escape(name)}</td>'
                    f'<td class="number">{score}</td><td class="number">{api_score}</td>'
                    f'<td class="number">{reference["n"]:,}</td></tr>')
    return ''.join(rows)


def initial_rows(data):
    task = next(t for t in data['tasks'] if t['id'] == 'massive_nb')
    rows = []
    for r in task['rows']:
        accuracy = f"{r['accuracy']:.1%}" if r['accuracy'] is not None else 'Withheld'
        median = f"{r['median_ms']:.1f}" if r['median_ms'] is not None else 'Unavailable'
        memory = f"{r['vram_gib']:.2f}" if r['vram_gib'] is not None else ('Unknown' if r['id'] == JEV else 'Unavailable')
        count = f"{r['correct']:,}/{r['n']:,}" if 'n' in r else 'Incomplete'
        answered = f"{r['answered']:,}/{r['n']:,}" if 'n' in r else 'Incomplete'
        rows.append(f'<tr><th scope="row">{html.escape(data["names"][r["id"]])}</th>'
                    f'<td>{"Hosted API" if r["id"] == JEV else "Local"}</td><td class="number"><strong>{accuracy}</strong>'
                    f'<small>{count} correct</small></td><td class="number">{answered}</td>'
                    f'<td class="number">{median}</td><td class="number">{memory}</td>'
                    f'<td class="timing-note">{html.escape(r["timing_note"])}</td></tr>')
    return ''.join(rows)


def render(comparison, measurements, completion, baseline):
    data = assemble(comparison, measurements, completion)
    template = Path(__file__).with_name('templates').joinpath('overview.html').read_text(encoding='utf-8')
    replacements = {
        '@@REVISION@@': REVISION,
        '@@LEADERS@@': leaderboard(data),
        '@@ROWS@@': initial_rows(data),
        '@@DATA@@': json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c'),
        '@@BASELINE@@': f'{baseline:.1%}',
        '@@LOCAL_MODELS@@': str(completion['local_models']),
        '@@FAILURES@@': str(completion['remaining_runtime_failures']),
        '@@TIMED@@': str(completion['timing_coverage']['complete_groups']),
        '@@TIMING_TOTAL@@': str(completion['timing_coverage']['expected_groups']),
    }
    for token, value in replacements.items():
        assert token in template
        template = template.replace(token, value)
    assert '@@' not in template
    return template
