"""Reproducible LinkedIn figures from completed benchmark evidence; no inference."""
from datetime import datetime, timezone
import hashlib
import html
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageOps

from .common import ROOT, read_json, write_json, digest

OUT = ROOT / 'results/linkedin/2026-10-01-summary'
INK = '#152D3B'
TEAL = '#187C76'
BLUE = '#4A789A'
MUTED = '#526671'
RULE = '#D7E0E3'
PALE = '#EDF3F3'
BG = '#FCFDFC'
WARM = '#9D593A'
TOTAL = 12
JEV = 'jev-1.13.0'
FRESH = ['fresh/news', 'fresh/emotion', 'fresh/spam', 'fresh/phishing']
TIERS = ['jevbench_public/easy', 'jevbench_public/original', 'jevbench_public/hard']
NAMES = {
    JEV: 'Jev 1.13.0 · API', 'laya': 'Laya', 'laya-multilingual': 'Laya multilingual',
    'julia': 'Julia', 'von': 'Von', 'gliner-decide': 'GLiNER Decide',
    'gliner-decide-multi': 'GLiNER multilingual', 'decider-08b': 'Decider 0.8B',
    'decider-2b': 'Decider 2B', 'nev-2b': 'Nev 2B Q8', 'imajev-2b': 'Imajev 2B',
    'decider-4b': 'Decider 4B v2.1', 'decision-4b-v12': 'Decision 4B v1.2',
    'kev-4b': 'Kev 4B', 'intern-decision-4b': 'Intern-Decision 4B',
    'tev1': 'Tev1 4B', 'wald-4b-v12': 'Wald 4B v1.2', 'imajev-4b': 'Imajev 4B',
    'nimble-9b': 'Nimble 9B NF4', 'winnow-12b-q8': 'Winnow 12B Q8',
    'cygnet-12b-nf4': 'Cygnet 12B NF4', 'jev-omni-12b-nf4': 'Jev-Omni 12B NF4',
    'clm-int8': 'CLM 8B INT8',
}
CARDS = []
QA = []
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 15,
                     'text.color': INK, 'axes.labelcolor': MUTED,
                     'svg.fonttype': 'none', 'axes.unicode_minus': False})


def text(fig, x, y, value, size=16, color=INK, weight='normal', **kwargs):
    return fig.text(x, y, value, fontsize=size, color=color, weight=weight,
                    va='top', linespacing=1.35, **kwargs)


def rule(fig, y, x=.07, right=.93):
    fig.add_artist(plt.Line2D([x, right], [y, y], transform=fig.transFigure,
                            color=RULE, lw=1))


def box(fig, x, y, w, h, color=PALE):
    fig.patches.append(Rectangle((x, y), w, h, transform=fig.transFigure,
                                 facecolor=color, edgecolor='none', zorder=-1))


def page(number, title, subtitle, section):
    fig = plt.figure(figsize=(10.8, 13.5), dpi=200, facecolor=BG)
    text(fig, .07, .957, 'DECISION MODELS / FIELD NOTES', 11, TEAL, 'semibold')
    text(fig, .93, .957, section.upper(), 11, MUTED, ha='right')
    text(fig, .07, .911, title, 32, INK, 'semibold')
    text(fig, .07, .810, subtitle, 15, MUTED)
    rule(fig, .064)
    text(fig, .07, .046, 'LOCAL BENCHMARK  /  01 OCT 2026', 10.5, MUTED)
    text(fig, .93, .046, f'{number:02d} / {TOTAL:02d}', 10.5, MUTED, ha='right')
    return fig


def note(fig, message, y=.132):
    text(fig, .07, y, message, 11.5, MUTED)


def save(fig, stem, alt):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    overflow = []
    for item in fig.texts:
        b = item.get_window_extent(renderer)
        if b.x0 < 0 or b.y0 < 0 or b.x1 > width or b.y1 > height:
            overflow.append(item.get_text())
    assert not overflow, f'Text outside canvas in {stem}: {overflow}'
    for extension in ['png', 'svg']:
        fig.savefig(OUT / f'{stem}.{extension}', facecolor=BG)
    image = Image.open(OUT / f'{stem}.png')
    assert image.size == (2160, 2700)
    image.convert('RGB').resize((1080, 1350), Image.Resampling.LANCZOS).save(
        OUT / 'upload' / f'{stem}.png')
    QA.append({'file': f'{stem}.png', 'pixels': list(image.size), 'text_bounds': 'passed'})
    CARDS.append({'stem': stem, 'alt': alt, 'title': fig.texts[2].get_text().replace('\n', ' ')})
    plt.close(fig)


def load_evidence():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'upload').mkdir(exist_ok=True)
    sources = {}
    docs = {}
    paths = {
        'comparison': 'results/jev_live/comparison.json',
        'latency': 'results/latency/summary.json',
        'leaders': 'results/alternatives/leader-expansion.json',
        'claims': 'results/claims/summary.json',
        'full': 'results/full/summary.json',
        'clm': 'results/clm_int8/summary.json',
        'fresh_protocol': 'results/alternatives/jev_fresh-protocol.json',
    }
    for key, path in paths.items():
        raw = (ROOT / path).read_bytes()
        docs[key] = json.loads(raw)
        sources[key] = {'path': path, 'sha256': hashlib.sha256(raw).hexdigest()}
    report = docs['comparison']
    metrics = {}
    for model, result in report['models'].items():
        metrics[model] = {}
        for suite, v in result['suites'].items():
            if v.get('complete'):
                assert v['n'] == v['expected'] == v['attempted']
                assert abs(v['accuracy'] - v['correct'] / v['n']) < 1e-12
                metrics[model][suite] = {k: v[k] for k in [
                    'n', 'correct', 'accuracy', 'failures', 'coverage', 'wilson95',
                    'abstained_records', 'truncated_records'] if k in v}
    assert docs['latency']['complete']
    for model, v in docs['latency']['models'].items():
        assert v['calls'] == v['successful'] == 240 and v['errors'] == 0
    for v in docs['leaders']['added']:
        p = v['public_jevbench']
        assert p['complete'] and p['attempted'] == 231
        assert sum(metrics[v['id']][s]['correct'] for s in TIERS) == p['correct']
    # Recompute all plotted four-task/public accuracies from original predictions.
    # score() also checks every input hash against the frozen fixture.
    from .alternatives_jev import read_rows
    from .jev_live_report import score
    source_rows = read_rows(ROOT / 'data/prepared/jev_live.jsonl')
    groups = {s: [r for r in source_rows if r['suite'] == s and r.get('split') != 'development']
              for s in FRESH + TIERS}
    verified_outcomes = 0
    for model, suites in metrics.items():
        if not all(s in suites for s in FRESH):
            continue
        files = [ROOT / 'results/jev_live/predictions.jsonl'] if model == JEV else [
            ROOT / f'results/alternatives/{fixture}/{model}/predictions.jsonl'
            for fixture in ['jev_fresh', 'alternatives_claims']]
        predictions = {r['id']: r for p in files for r in read_rows(p)}
        for suite in FRESH + TIERS:
            if suite not in suites:
                continue
            rows = groups[suite]
            scores = [score(r, predictions[r['id']]) for r in rows]
            assert sum(s['correct'] for s in scores) == suites[suite]['correct']
            assert len(rows) == suites[suite]['n']
            verified_outcomes += len(rows)
    # The original Laya and CLM reports use the same MASSIVE test rows.
    assert digest(ROOT / 'data/prepared/full.json') == docs['full']['data_sha256']
    from .questions import question
    full = read_json(ROOT / 'data/prepared/full.json')
    fixtures = {r['id']: r for r in read_rows(ROOT / 'data/prepared/alternatives.jsonl')}
    for s in ['massive_en', 'massive_nb']:
        for row in full['suites'][s]['splits']['test']:
            other = fixtures[f'{s}/{row["id"]}']
            assert other['state'] == row['text']
            assert other['questions'] == question(full['suites'][s])
            assert other['gold'] == {'decision': [row['label']]}
    # Freeze provenance without private API ledgers or credentials.
    checkpoints = {}
    for p in sorted((ROOT / 'results/alternatives/models').glob('*.json')):
        v = read_json(p)
        checkpoints[p.stem] = {k: v[k] for k in ['repo', 'revision', 'license', 'model_card_sha256'] if k in v}
        if 'repo' in v and 'revision' in v:
            checkpoints[p.stem]['card_url'] = f'https://huggingface.co/{v["repo"]}/blob/{v["revision"]}/README.md'
    evidence = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'comparison_snapshot_utc': report['updated'], 'sources': sources,
        'completed_suite_metrics': metrics, 'latency': docs['latency'],
        'publisher_claims': docs['leaders']['added'],
        'laya_replication': {k: {a: b for a, b in v.items() if a in ['n', 'correct', 'accuracy', 'published_accuracy']}
                             for k, v in docs['claims']['results'].items() if 'extension' not in k},
        'original_laya_language': {m: {s: {k: v[k] for k in ['n', 'correct', 'accuracy']}
                                             for s, v in x['suites'].items()}
                                  for m, x in docs['full']['models'].items()},
        'supervised_baselines': docs['full']['baselines'],
        'clm_language': {s: {k: v[k] for k in ['n', 'correct', 'accuracy']}
                        for s, v in docs['clm']['results'].items() if s.startswith('massive/')},
        'paired_decider_jev_norwegian': report['paired_comparisons']['decider-4b']['massive_nb'],
        'fresh_protocol': docs['fresh_protocol'], 'checkpoints': checkpoints,
        'validation': {'recomputed_case_model_outcomes': verified_outcomes,
                       'input_hashes': 'checked against frozen fixtures',
                       'original_and_current_massive_test_texts_questions_labels': 'identical'},
        'size_basis': 'Rounded publisher nominal parameter counts. Specific encoder totals come from pinned model cards; B-scale family names describe nominal backbone sizes. Counts do not measure allocated memory.',
        'limits': ['Only completed suites receive accuracy scores.',
                   'Public training overlap is unknown; Laya lists news, spam and phishing among training tasks.',
                   'Model sizes are nominal publisher counts, not measured VRAM.',
                   'NF4 variants differ from publisher BF16 configurations.',
                   'Local and API latency describe different deployment paths.',
                   'Industrial examples are authored diagnostics, without independent domain-expert validation.'],
    }
    write_json(OUT / 'evidence.json', evidence)
    return docs, metrics


def overview(d, m):
    fig = page(1, 'Can local models make\nuseful business decisions?',
               'Accuracy, response time and deployment trade-offs\nfrom Laya, open alternatives and a live Jev API baseline.', 'The findings')
    entries = [
        ('87.9%', 'Decision 4B v1.2', '203 of 231 public JevBench cases correct.\nJev: 199. Four cases do not establish an overall winner.'),
        ('25.9 ms', 'Laya median response', 'Short messages on a local RTX 5070 Ti.\nJev API median: 261.2 ms including network time.'),
        ('82.0%', 'Decider 4B in Norwegian', '2,418 of 2,948 intent scenarios correct.\nJev: 81.8%. No clear difference in this paired test.'),
    ]
    for i, (value, title, body) in enumerate(entries):
        y = .716 - i * .161
        text(fig, .07, y, value, 37, TEAL, 'semibold')
        text(fig, .365, y - .006, title, 18, INK, 'semibold')
        text(fig, .365, y - .047, body, 14, MUTED)
        rule(fig, y - .133)
    text(fig, .07, .205, 'The useful choice depends on the task.', 23, INK, 'semibold')
    note(fig, 'Completed test suites only. The wider benchmark matrix is still running.\nPublic benchmarks indicate where to pilot; they do not establish production reliability.')
    save(fig, '01-the-findings', 'Three measured findings: Decision 4B scored 203/231 on public JevBench, Laya median response was 25.9 ms, and Decider 4B scored 82.0% on Norwegian scenarios. Scope differs between the three tests.')


def sizes(d, m):
    fig = page(2, 'How much model\ndo you need to run?',
               'Publisher parameter counts. M = million; B = billion.\nSmaller models can be useful, but size alone does not predict accuracy.', 'Model size')
    rows = [('Julia', '144M'), ('GLiNER multilingual', '287M'), ('Laya multilingual', '322M'),
            ('GLiNER Decide', '340M'), ('Von', '395M'), ('Laya', '421M'),
            ('Decider 0.8B', '0.8B'), ('Decider / Nev / Imajev 2B', '2B'),
            ('Decider / Decision / Kev / other 4B models', '4B'),
            ('CLM · Qwen3 backbone', '8B'), ('Nimble', '9B'),
            ('Winnow / Cygnet / Jev-Omni', '12B'), ('Jev · managed API', 'Undisclosed')]
    for i, (label, size) in enumerate(rows):
        y = .733 - i * .0335
        if i % 2 == 0: box(fig, .06, y - .030, .88, .034)
        text(fig, .077, y - .004, label, 15)
        text(fig, .919, y - .004, size, 16, TEAL if i < 12 else MUTED, 'semibold', ha='right')
    box(fig, .07, .178, .86, .087)
    text(fig, .09, .247, 'Parameters are not a VRAM requirement.', 18, INK, 'semibold')
    text(fig, .09, .214, 'Precision, context length and runtime all affect memory use.', 14, MUTED)
    note(fig, 'Local variants: Winnow Q8; Cygnet, Jev-Omni and Nimble NF4; CLM INT8.\nQuantization reduces weight storage, not parameter count. Counts are rounded.\nPinned publisher cards and checkpoint references are included in the evidence file.', .141)
    save(fig, '02-model-size', 'Nominal sizes range from Julia 144 million parameters to 12 billion parameter models. Jev parameter count is undisclosed. Precision and context also affect deployment memory.')


def accuracy_page(number, ids, title, caption, stem, m):
    fig = page(number, title, 'Four English classification tasks. Same frozen test cases for every model.\nAccuracy (%) · 200 test examples per task · higher is better.', 'Task accuracy')
    cols = [.465, .597, .729, .861]
    text(fig, .07, .728, 'MODEL', 11, MUTED, 'semibold')
    for x, label in zip(cols, ['NEWS', 'EMOTION', 'SPAM', 'PHISHING']):
        text(fig, x, .728, label, 10.5, MUTED, 'semibold', ha='center')
    cmap = LinearSegmentedColormap.from_list('score', ['#F0E1D6', '#F4F5EF', '#BFDCD5'])
    for i, model in enumerate(ids):
        y = .689 - i * .0355
        text(fig, .07, y - .002, NAMES[model], 14.5, INK, 'semibold' if model == JEV else 'normal')
        for x, suite in zip(cols, FRESH):
            v = m[model][suite]
            assert v['n'] == 200
            box(fig, x - .061, y - .029, .122, .034, cmap(v['accuracy']))
            suffix = '*' if v['failures'] else ''
            text(fig, x, y - .002, f'{100*v["accuracy"]:.1f}{suffix}', 16, INK, 'semibold', ha='center')
    text(fig, .07, .235, caption, 18, INK, 'semibold')
    missing = [(NAMES[x], sum(m[x][s]['failures'] for s in FRESH)) for x in ids]
    missing = [(x, n) for x, n in missing if n]
    foot = '* Unanswered/invalid calls stay in the denominator.\n' if missing else ''
    if missing:
        foot += '; '.join(f'{x}: {n}/800' for x, n in missing) + '.\n'
    foot += 'Task selection used earlier local strengths; test rows were frozen before calls.\nPublic training overlap is unknown. No single score is averaged across these tasks.'
    if number == 3:
        foot += '\nLaya lists news, spam and phishing as training tasks; 100% here is not a guarantee.'
    else:
        foot += '\nNF4 rows are local quantized variants. Accuracy is specific to these deployments.'
    note(fig, foot, .180)
    save(fig, stem, caption.replace('\n', ' ') + ' All percentages use 200 test cases per task; invalid or unanswered cases count against accuracy. Exact counts and intervals accompany the images.')


def public(d, m):
    ids = ['decision-4b-v12', 'winnow-12b-q8', JEV, 'jev-omni-12b-nf4', 'cygnet-12b-nf4', 'decider-4b']
    fig = page(5, 'The public JevBench\ncomparison is close.',
               '231 source-verified cases: 48 easy + 72 original + 111 hard.\nNew additions, live Jev and our Decider 4B reference.', 'Same public cases')
    text(fig, .07, .724, 'MODEL / LOCAL CONFIGURATION', 11, MUTED, 'semibold')
    text(fig, .93, .724, 'CORRECT / 231', 11, MUTED, 'semibold', ha='right')
    for i, model in enumerate(ids):
        hits = sum(m[model][s]['correct'] for s in TIERS)
        y = .679 - i * .072
        text(fig, .07, y, NAMES[model], 17, INK, 'semibold')
        text(fig, .93, y, f'{hits}  /  {100*hits/231:.1f}%', 19, TEAL, 'semibold', ha='right')
        box(fig, .07, y - .042, .86, .008, PALE)
        box(fig, .07, y - .042, .86 * hits / 231, .008, TEAL if model != JEV else BLUE)
    text(fig, .07, .206, 'Decision matches the published 203/231 count.', 20, INK, 'semibold')
    note(fig, 'A matching total does not prove identical decisions or probabilities.\nThese public cases do not reproduce the sealed leaderboard or its composite score.\nSmall gaps do not establish a general winner; training exposure is unknown.\nCygnet and Jev-Omni use local NF4 variants; Winnow uses publisher Q8 on Windows.', .151)
    save(fig, '05-public-jevbench', 'Public JevBench: Decision 203/231, Winnow and Jev 199/231, Jev-Omni 198/231, Cygnet 195/231, Decider 191/231. These results do not reproduce the sealed leaderboard.')


def language(d, m):
    fig = page(6, 'Norwegian needs\nits own evaluation.',
               'MASSIVE: 18 intent scenarios, with 2,948 matched test cases\nin each language. Norwegian means Bokmål in this test.', 'Language transfer')
    rows = []
    for model in ['decider-4b', JEV, 'kev-4b', 'decider-2b', 'gliner-decide']:
        rows.append((NAMES[model], *[100*m[model][s]['accuracy'] for s in ['massive_en', 'massive_nb']]))
    for model in ['laya-multilingual', 'laya']:
        rows.append((NAMES[model], *[100*d['full']['models'][model]['suites'][s]['accuracy'] for s in ['massive_en', 'massive_nb']]))
    rows.append(('CLM 8B INT8', *[100*d['clm']['results'][f'massive/{s}']['accuracy'] for s in ['massive_en', 'massive_nb']]))
    text(fig, .07, .725, 'SELECTED MODELS', 11, MUTED, 'semibold')
    for x, label in [(.675, 'ENGLISH'), (.89, 'BOKMÅL')]: text(fig, x, .725, label, 11, MUTED, 'semibold', ha='right')
    for i, (label, en, nb) in enumerate(rows):
        y = .682 - i * .040
        if i % 2 == 0: box(fig, .06, y - .034, .88, .040)
        text(fig, .07, y, label, 17)
        text(fig, .675, y, f'{en:.1f}%', 18, MUTED, ha='right')
        text(fig, .89, y, f'{nb:.1f}%', 18, TEAL, 'semibold', ha='right')
    box(fig, .07, .247, .86, .079)
    baseline = d['full']['baselines']['massive_nb']['tfidf_linear_svm']['accuracy']
    text(fig, .09, .311, f'A supervised text baseline reached {100*baseline:.1f}% in Bokmål.', 17, INK, 'semibold')
    text(fig, .09, .279, 'TF-IDF + linear SVM, trained on labelled training data.', 14, MUTED)
    text(fig, .07, .211, 'Decider vs Jev: +0.27 percentage points.', 19, INK, 'semibold')
    note(fig, 'Paired 95% bootstrap interval: −0.85 to +1.36 points; no clear difference.\nTwo Jev API failures count as incorrect. All rows use the same test texts and labels.\nLaya / CLM are earlier runs; the fitted baseline has task-specific training.\nNew 12B models do not yet have complete matched language results.', .158)
    save(fig, '06-norwegian', 'Bokmål accuracy: Decider 4B 82.0%, Jev 81.8%, Laya multilingual 51.3%, English Laya 21.1%, CLM INT8 30.3%. A separately trained text baseline reached 89.1%. Different training regimes are labelled.')


def latency(d, m):
    fig = page(7, 'Fast enough to sit\ninside a workflow.',
               'Short-message response time. One request at a time.\n80 fixed messages × 3 passes = 240 timed calls per model.', 'Measured response')
    rows = sorted(d['latency']['models'].items(), key=lambda x: x[1]['median_ms'])
    text(fig, .07, .726, 'MODEL', 11, MUTED, 'semibold')
    text(fig, .745, .726, 'MEDIAN', 11, TEAL, 'semibold', ha='right')
    text(fig, .93, .726, '95TH %ILE', 11, MUTED, 'semibold', ha='right')
    for i, (model, v) in enumerate(rows):
        y = .681 - i * .047
        text(fig, .07, y, NAMES[model], 16)
        box(fig, .411, y - .020, .18, .009, PALE)
        box(fig, .411, y - .020, .18 * v['median_ms'] / 270, .009, TEAL)
        text(fig, .745, y, f'{v["median_ms"]:.1f} ms', 18, TEAL, 'semibold', ha='right')
        text(fig, .93, y, f'{v["p95_ms"]:.1f} ms', 17, MUTED, ha='right')
    box(fig, .07, .205, .86, .080)
    text(fig, .09, .270, 'Local GPU time and API time answer different questions.', 16, INK, 'semibold')
    text(fig, .09, .236, 'The API measurement includes the network and hosted service.', 14, MUTED)
    note(fig, 'Local: RTX 5070 Ti 16 GB, Windows, batch 1; loaded models, warm-up excluded.\nIncludes tokenization and output handling. 0 failed calls and 0 native truncations.\nState ≤512 characters. API uses persistent HTTPS; hosted hardware is unknown.\nNew leaders have no matched latency result yet. No throughput claim is made.', .157)
    save(fig, '07-response-time', 'Median and 95th-percentile response times for eight completed controlled runs. Local medians range from 19.1 to 64.6 ms. Live Jev median is 261.2 ms including the network and hosted service.')


def claims(d, m):
    values = [v for k, v in d['claims']['results'].items() if 'extension' not in k]
    within = sum(abs(v['accuracy'] - v['published_accuracy']) <= .02000001 for v in values)
    assert len(values) == 16 and within == 15
    fig = page(8, 'What happened when\nwe checked the claims?',
               'Replication asks whether a stated setup reproduces.\nTransfer asks whether the result holds on another task.', 'Claims audit')
    rows = [
        ('LAYA / BOTH CHECKPOINTS', '15 of 16 figures within 2 percentage points',
         'Publisher tasks and sampling rebuilt, 300–400 cases per task.\nSDK and hardware differ; upstream dataset revisions were unpinned.'),
        ('DECISION 4B v1.2', '203/231 published. 203/231 measured.',
         'The public correct count matches using the native BF16 model.\nCUDA graphs disabled. This is not a sealed-board replication.'),
        ('WINNOW 12B Q8', '198/231 published. 199/231 measured.',
         'One additional correct case with the publisher Q8 weights.\nWindows runtime and local context settings differ.'),
        ('CYGNET / JEV-OMNI', 'Exact published configuration: untested here.',
         'Our NF4 results are 195/231 and 198/231 respectively.\nThey cannot verify or refute the publishers’ BF16 claims.'),
    ]
    for i, (label, headline, body) in enumerate(rows):
        y = .733 - i * .133
        text(fig, .07, y, label, 11, TEAL, 'semibold')
        text(fig, .07, y - .026, headline, 19, INK, 'semibold')
        text(fig, .07, y - .063, body, 14, MUTED)
        rule(fig, y - .116)
    note(fig, 'Laya’s 2-point band is an engineering repeatability check, not a significance test.\nThe remaining result was 5 points higher. News/email training tasks are disclosed.\nFixed answer labels prevent invented labels; they do not prevent wrong decisions.', .144)
    save(fig, '08-claims-audit', 'Laya reproduced 15 of 16 figures within a 2-point engineering band; Decision matched 203/231; Winnow scored one case above its 198/231 Q8 claim. Local NF4 tests cannot verify publisher BF16 claims.')


def technology(d, m):
    fig = page(9, 'Similar outputs.\nDifferent technology.',
               'Each approach returns a choice or score that software can use.\nThe operating trade-offs come from the model and its runtime.', 'How they work')
    rows = [
        ('01', 'Encoder + decision head', 'Laya · GLiNER Decide · Julia · Von',
         'Read the input, then score the supplied choices.',
         'Compact deployment; validate label and context budgets.'),
        ('02', 'Language model + decision readout', 'Decider · Decision · Kev · Winnow · Jev-Omni',
         'Use a language-model backbone to score allowed answers.',
         'Larger memory footprint; quantization changes the deployment.'),
        ('03', 'Contrastive embeddings', 'CLM v0.1 · Qwen3-8B backbone',
         'Embed the state and candidate actions, then compare them.',
         'Reusable action embeddings may help fixed candidate sets.'),
        ('04', 'Managed decision API', 'TypeSafe Jev 1.13.0',
         'Send typed questions over HTTPS and receive decisions.',
         'No local model hosting; service and network affect response time.'),
    ]
    for i, (n, title, models, mechanism, implication) in enumerate(rows):
        y = .735 - i * .137
        text(fig, .07, y, n, 13, TEAL, 'semibold')
        text(fig, .125, y + .002, title, 21, INK, 'semibold')
        text(fig, .125, y - .034, models, 13, TEAL)
        text(fig, .125, y - .063, mechanism, 14.5)
        text(fig, .125, y - .091, implication, 13, MUTED)
        rule(fig, y - .118)
    note(fig, 'Architecture summaries use pinned publisher model cards and inference code.\nOperational implications are engineering interpretations, not measured guarantees.\nJev’s internal architecture and parameter count were not independently verified.', .143)
    save(fig, '09-technology', 'Four approaches: encoder classifiers, language models with decision readouts, contrastive embeddings, and the hosted Jev API. Each exposes structured decisions with different hosting trade-offs.')


def limits(d, m):
    fig = page(10, 'Can it handle your\nactual list of choices?',
               '100 verified banking requests with 72 candidate intents each.\nThis pilot tests both decision quality and interface limits.', 'Workflow fit')
    ids = [JEV, 'jev-omni-12b-nf4', 'decision-4b-v12', 'winnow-12b-q8', 'cygnet-12b-nf4']
    text(fig, .07, .728, 'MODEL', 11, MUTED, 'semibold')
    text(fig, .93, .728, 'RESULT ON THE 72-CHOICE TASK', 11, MUTED, 'semibold', ha='right')
    caps = {'decision-4b-v12': 26, 'winnow-12b-q8': 64, 'cygnet-12b-nf4': 26}
    for i, model in enumerate(ids):
        v = m[model]['jev_verified/banking77']
        assert v['n'] == 100
        y = .674 - i * .068
        text(fig, .07, y, NAMES[model], 17, INK, 'semibold')
        if model in caps:
            assert v['failures'] == 100
            result = f'Rejected · {caps[model]}-choice cap'
            color = WARM
        else:
            assert v['failures'] == 0
            result = f'{v["correct"]}/100 correct'
            color = TEAL
        text(fig, .93, y, result, 17, color, ha='right')
        rule(fig, y - .040)
    box(fig, .07, .233, .86, .098)
    text(fig, .09, .313, 'An unsupported request is a deployment constraint.', 19, INK, 'semibold')
    text(fig, .09, .273, 'It is recorded as unanswered, never silently converted\ninto a successful prediction.', 14.5, MUTED)
    note(fig, 'These are native interface caps in the configurations tested.\nA shortlist or two-stage router changes the task and needs a new end-to-end test.\nPublic JevBench accuracy alone does not establish workflow compatibility.\nThis is the exact 100-case banking pilot, separate from the 231-case public suite.', .160)
    save(fig, '10-workflow-fit', 'On the 100-case banking pilot with 72 choices, Jev got 89 correct and Jev-Omni NF4 52. Decision and Cygnet rejected the 72-choice interface because of 26-choice caps; Winnow had a 64-choice cap.')


def business(d, m):
    fig = page(11, 'Where this could\ncreate business value.',
               'Start with frequent decisions that people can inspect and correct.\nThe financial example below is an assumption, not a measured return.', 'Software use cases')
    rows = [('HELPDESK', 'Suggest the right support queue', 'Measure accepted suggestions, reassignments and handling time.'),
            ('OIL & GAS / ENGINEERING', 'Classify incoming technical documents', 'Measure review time and retrieval success with domain experts.'),
            ('MANUFACTURING', 'Organize maintenance requests', 'Measure technician admin time and corrected categories.')]
    for i, (sector, action, measure) in enumerate(rows):
        y = .732 - i * .103
        text(fig, .07, y, sector, 11, TEAL, 'semibold')
        text(fig, .07, y - .026, action, 21, INK, 'semibold')
        text(fig, .07, y - .063, measure, 13.5, MUTED)
    box(fig, .07, .221, .86, .172)
    text(fig, .09, .374, 'ILLUSTRATIVE ANNUAL SCENARIO', 11, MUTED, 'semibold')
    text(fig, .09, .342, '333 hours', 34, TEAL, 'semibold')
    text(fig, .51, .342, 'NOK 150,000', 30, TEAL, 'semibold')
    text(fig, .09, .286, 'staff time recovered', 14, MUTED)
    text(fig, .51, .286, 'net annual value', 14, MUTED)
    text(fig, .09, .251, '100,000 cases × 60% adoption × 20 net seconds saved', 15, INK, 'semibold')
    note(fig, 'Assumptions: NOK 600/hour; NOK 50,000/year integration and operating cost.\nNet seconds include review, corrections and rework. Savings are not measured.\nOur industry examples are small authored diagnostics, not expert-validated trials.\nUse recommendations for a pilot; safety-critical operational decisions remain unvalidated.', .168)
    save(fig, '11-business-value', 'Potential pilots: helpdesk routing, engineering document intake and manufacturing maintenance triage. An illustrative scenario of 100,000 cases, 60% adoption and 20 seconds net saved yields 333 hours and NOK 150,000 after assumed costs.')


def close(d, m):
    fig = page(12, 'Choose the workflow.\nThen choose the model.',
               'The strongest business case combines useful accuracy,\nacceptable response time and a measurable reduction in work.', 'From test to pilot')
    rows = [
        ('Freeze the task', 'Use real labels, representative documents and both languages.\nKeep development cases separate from the final test.'),
        ('Count every outcome', 'Report wrong answers, refusals, invalid outputs and truncation.\nMeasure the time users wait in the intended deployment.'),
        ('Keep a simple comparator', 'With labelled data, test a trained classifier too.\nOur Norwegian TF-IDF baseline outscored the tested decision models.'),
        ('Measure value in a pilot', 'Track accepted suggestions, corrections and net time saved.\nSelect review thresholds on development data, then freeze them.'),
    ]
    for i, (title, body) in enumerate(rows):
        y = .726 - i * .116
        text(fig, .07, y, title, 23, INK, 'semibold')
        text(fig, .07, y - .041, body, 15, MUTED)
        rule(fig, y - .098)
    text(fig, .07, .203, 'Local deployment is viable for selected tasks.', 22, TEAL, 'semibold')
    note(fig, 'This carousel uses completed suites from a continuing benchmark.\nIt includes all models with complete four-task results at the saved snapshot.\nSource hashes, model revisions, exact counts and uncertainty intervals accompany it.\nThe full GPU matrix and matched response times for new leaders remain unfinished.', .151)
    save(fig, '12-from-benchmark-to-pilot', 'Select a workflow first, freeze representative test cases, include failures, compare a simple trained baseline, and measure net value in a pilot. Broader testing remains in progress.')


def package(d, m):
    thumbs = []
    for card in CARDS:
        im = Image.open(OUT / f'{card["stem"]}.png').convert('RGB')
        im.thumbnail((360, 450), Image.Resampling.LANCZOS)
        thumbs.append(ImageOps.expand(im, border=8, fill='#D7E0E3'))
    contact = Image.new('RGB', (4*376, 3*466), '#E8EEEE')
    for i, im in enumerate(thumbs): contact.paste(im, ((i % 4)*376, (i // 4)*466))
    contact.save(OUT / 'contact-sheet.jpg', quality=94)
    post = '''I tested local decision models against the live Jev API, including Laya, Decider, Decision, Winnow and other open alternatives.

The useful finding is how much the answer changes with the task.

Decision 4B matched its published 203/231 correct count on public JevBench. Live Jev scored 199/231. That four-case gap does not establish a general winner.

On 2,948 Norwegian Bokmål intent scenarios, Decider 4B reached 82.0% and Jev 81.8%, with no clear difference in the paired test. A supervised TF-IDF classifier trained on task-specific labels reached 89.1%.

Local response times were attractive on the short-message workload: medians of 19–65 ms among the seven models we timed. Jev's 261 ms median includes the network and hosted service. The new leaders still need the same controlled timing test.

Most Laya accuracy claims reproduced under their stated setup. That does not mean the same accuracy transfers to Norwegian or a new industrial workflow. Public training overlap is unknown, and Laya lists news and email screening among its training tasks.

For business software, I would start with support queue suggestions, technical document intake or maintenance request categories. Measure corrections and net time saved before expanding automation.

The carousel includes model sizes, task accuracy, response times, technology approaches and the limits of the comparison. This is a snapshot of completed suites; the wider benchmark continues.
'''
    (OUT / 'linkedin-post.txt').write_text(post, encoding='utf-8')
    (OUT / 'alt-text.txt').write_text('\n\n'.join(f'{c["stem"]}\n{c["alt"]}' for c in CARDS), encoding='utf-8')
    (OUT / 'README.txt').write_text(
        'LinkedIn benchmark carousel — 1 October 2026\n\n'
        'Upload: upload/*.png, numbered 01–12, 1080 × 1350 pixels (4:5).\n'
        'Masters: root PNG files, 2160 × 2700. SVG files contain editable text.\n'
        'Preview: index.html or contact-sheet.jpg. Caption: linkedin-post.txt.\n'
        'Evidence: evidence.json contains the frozen metrics, source hashes and model revisions.\n'
        'Read each slide footnote; individual figures describe different test protocols.\n'
        'Task accuracy is never averaged into a misleading overall model score.\n'
        'Rebuild from the workspace with .venv/Scripts/python.exe -m laya_bench.linkedin_summary\n'
        'A rebuild intentionally takes a new completed-results snapshot.\n', encoding='utf-8')
    cards = ''.join(f'<article><h2>{html.escape(c["title"])}</h2><a href="{c["stem"]}.png"><img loading="lazy" src="upload/{c["stem"]}.png" alt="{html.escape(c["alt"])}"></a><p><a href="upload/{c["stem"]}.png">Upload PNG</a> · <a href="{c["stem"]}.png">High resolution</a> · <a href="{c["stem"]}.svg">Editable SVG</a></p></article>' for c in CARDS)
    coverage = ''
    for model, v in d['comparison']['models'].items():
        fresh = sum(s in m[model] for s in FRESH)
        pub = sum(s in m[model] for s in TIERS)
        nb = m[model].get('massive_nb')
        timing = d['latency']['models'].get(model)
        coverage += f'<tr><td>{html.escape(NAMES.get(model, v["label"]))}</td><td>{fresh}/4 complete</td><td>{sum(m[model][s]["correct"] for s in TIERS) if pub == 3 else "Pending"}{"/231" if pub == 3 else ""}</td><td>{str(nb["correct"])+"/2948" if nb else "Pending in current matrix"}</td><td>{str(round(timing["median_ms"],1))+" ms" if timing else "Not in matched timing run"}</td></tr>'
    sources = [
        ('Laya model card', 'https://huggingface.co/convaiinnovations/laya/blob/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/README.md'),
        ('Laya publisher benchmark protocol', 'https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/BENCHMARKS.md'),
        ('Public JevBench leaderboard', 'https://benchmarkheaven.com/jev-models'),
        ('MASSIVE source dataset', 'https://huggingface.co/datasets/AmazonScience/massive'),
    ]
    page_html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local decision models — LinkedIn carousel</title><style>body{margin:0;background:#edf2f2;color:#152d3b;font:16px/1.6 Segoe UI,Arial,sans-serif}main{max-width:1240px;margin:auto;padding:40px 24px}h1{font-size:42px;line-height:1.1;max-width:850px}h2{font-size:20px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}article{background:white;padding:16px}img{width:100%;height:auto;display:block}a{color:#187c76}.links{background:white;padding:20px;margin:24px 0}table{border-collapse:collapse;background:white;font-size:13px;width:100%}td,th{padding:10px;border-bottom:1px solid #d7e0e3;text-align:left}small{color:#526671}.scroll{overflow-x:auto}@media(max-width:700px){.grid{grid-template-columns:1fr}h1{font-size:32px}}</style><main><p>BENCHMARK SNAPSHOT / 1 OCTOBER 2026</p><h1>Local decision models in business software</h1><p>12 clean graphics covering size, task accuracy, Norwegian, response time, claim replication and practical value. Every headline uses a completed test suite. The wider GPU benchmark continues.</p><div class="links"><a href="linkedin-carousel-upload.zip">Download 12 upload-ready PNGs</a> · <a href="linkedin-carousel-complete.zip">Download PNG/SVG masters and evidence</a><br><a href="contact-sheet.jpg">See all slides</a> · <a href="linkedin-post.txt">Suggested LinkedIn post</a> · <a href="alt-text.txt">Alt text</a> · <a href="evidence.json">Data and source hashes</a></div><div class="grid">'''+cards+'</div><h2>Coverage at the frozen snapshot</h2><p>Accuracy is shown only for complete suites. Earlier Laya / CLM language runs appear on slide 6 and in evidence.json, while the table below reflects the current shared matrix.</p><div class="scroll"><table><tr><th>Model</th><th>Fresh tasks</th><th>Public JevBench</th><th>Bokmål</th><th>Median response</th></tr>'+coverage+'</table></div><h2>Primary references</h2><ul>'+''.join(f'<li><a href="{url}">{html.escape(label)}</a></li>' for label,url in sources)+'</ul><p>All checkpoint links, counts, Wilson intervals and the selected paired comparison are in evidence.json. Model size is a nominal publisher count, not memory consumption. Neither training-data independence nor production safety is established by these results.</p></main></html>'
    (OUT / 'index.html').write_text(page_html, encoding='utf-8')
    write_json(OUT / 'render-checks.json', {'images': QA, 'slide_count': len(CARDS), 'snapshot': d['comparison']['updated']})
    with zipfile.ZipFile(OUT / 'linkedin-carousel-upload.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for c in CARDS: z.write(OUT / 'upload' / f'{c["stem"]}.png', f'{c["stem"]}.png')
        for filename in ['linkedin-post.txt', 'alt-text.txt', 'README.txt']: z.write(OUT / filename, filename)
    with zipfile.ZipFile(OUT / 'linkedin-carousel-complete.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix != '.zip': z.write(p, str(p.relative_to(OUT)))


def build():
    d, m = load_evidence()
    small = [JEV, 'julia', 'gliner-decide-multi', 'laya-multilingual', 'gliner-decide',
             'von', 'laya', 'decider-08b', 'decider-2b', 'nev-2b', 'imajev-2b']
    large = [JEV, 'decider-4b', 'decision-4b-v12', 'kev-4b', 'intern-decision-4b',
             'tev1', 'wald-4b-v12', 'imajev-4b', 'nimble-9b', 'winnow-12b-q8',
             'cygnet-12b-nf4', 'jev-omni-12b-nf4']
    complete = {k for k,v in m.items() if all(s in v for s in FRESH)}
    assert set(small + large) == complete, 'Update accuracy panels for newly completed models.'
    overview(d, m)
    sizes(d, m)
    accuracy_page(3, small, 'Compact models can be\nstrong on specific tasks.',
                  'A strong result on one task can coexist\nwith a weak result on another.', '03-accuracy-compact', m)
    accuracy_page(4, large, 'More parameters do not\nwin every column.',
                  'Decider and Decision are different model families.\nThe workflow decides which result matters.', '04-accuracy-larger', m)
    public(d, m)
    language(d, m)
    latency(d, m)
    claims(d, m)
    technology(d, m)
    limits(d, m)
    business(d, m)
    close(d, m)
    package(d, m)
    print(f'Created {len(CARDS)} slides, upload bundle, editable masters and evidence at {OUT}')


if __name__ == '__main__':
    build()
