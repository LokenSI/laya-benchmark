"""One business-facing share card, generated from completed aggregate evidence."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from .common import ROOT, read_json, write_json

OUT = ROOT / 'results/share/decision2-2026-10-03'
PAGE = 'https://lokensi.github.io/laya-benchmark/'
INK, TEAL, BLUE, MUTED = '#152D3B', '#187C76', '#4A789A', '#526671'
PALE, BG = '#EDF3F3', '#FCFDFC'

def build():
    inputs = {
        'decision2': 'results/alternatives/decision2-summary.json',
        'verification': 'results/alternatives/decision2-verification.json',
        'comparison': 'results/jev_live/comparison.json',
        'stress': 'results/decision2-stress/summary.json',
        'full': 'results/full/summary.json',
    }
    evidence = {k: read_json(ROOT / v) for k, v in inputs.items()}
    verified = evidence['verification']
    assert verified['status'] == 'complete'
    assert verified['saved_case_predictions'] == 70304
    assert verified['saved_prediction_errors'] == 0
    nox = evidence['decision2']['models']['decision2-nox-4b']['shared_suites']['alternatives']['suites']['massive_nb']
    models = evidence['comparison']['models']
    measures = [
        ('Decider 4B v2.1', 'local / zero-shot', models['decider-4b']['suites']['massive_nb']),
        ('Jev 1.13.0', 'hosted API / zero-shot', models['jev-1.13.0']['suites']['massive_nb']),
        ('Decision 2.0 Nox 4B', 'local / zero-shot', nox),
        ('Laya Multilingual', 'local / zero-shot', models['laya-multilingual']['suites']['massive_nb']),
    ]
    for _, _, metric in measures:
        assert metric['complete'] and metric.get('attempted', metric.get('n')) == 2948
    supervised = evidence['full']['baselines']['massive_nb']['tfidf_linear_svm']['accuracy']
    assert all(x['status'] == 'complete' for x in evidence['stress']['native'].values())
    assert all(x['status'] == 'unsupported' for x in evidence['stress']['vllm'].values())
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Segoe UI', 'svg.fonttype': 'none', 'text.color': INK})
    fig = plt.figure(figsize=(10.8, 13.5), dpi=200, facecolor=BG)
    def text(x, y, value, size=16, color=INK, weight='normal', **kwargs):
        return fig.text(x, y, value, fontsize=size, color=color, weight=weight,
                        va='top', linespacing=1.28, **kwargs)
    text(.07, .953, 'LOCAL AI / PRACTICAL BUSINESS VALUE', 11, TEAL, 'semibold')
    text(.07, .908, 'Less sorting.\nMore time for the work.', 35, weight='semibold')
    text(.07, .807, 'Use AI to suggest where requests belong.\nKeep the final decision with your team.', 17, MUTED)
    text(.07, .728, 'NORWEGIAN REQUEST ROUTING', 11.5, TEAL, 'semibold')
    text(.07, .704, '2,948 Bokmål test cases · 18 assistant-service categories', 13, MUTED)
    for i, (label, kind, metric) in enumerate(measures):
        y = .660 - i*.079
        color = TEAL if i == 2 else BLUE
        text(.07, y, label, 15.5, weight='semibold')
        text(.07, y-.025, kind, 11, MUTED)
        fig.patches.append(Rectangle((.425, y-.037), .39, .020, transform=fig.transFigure,
                                     facecolor=PALE, edgecolor='none'))
        fig.patches.append(Rectangle((.425, y-.037), .39*metric['accuracy'], .020,
                                     transform=fig.transFigure, facecolor=color, edgecolor='none'))
        text(.93, y-.008, f"{metric['accuracy']:.1%}", 21, color, 'semibold', ha='right')
    text(.07, .328, f'Also test a simple trained classifier: {supervised:.1%} here.\nIt used labelled training data; the models above did not.', 12.5, MUTED)
    fig.patches.append(Rectangle((.055, .118), .89, .143, transform=fig.transFigure,
                                 facecolor=PALE, edgecolor='none'))
    text(.075, .245, 'PILOT THE WORKFLOW, THEN MEASURE THE VALUE', 11.5, TEAL, 'semibold')
    text(.075, .216, 'Local processing can keep messages in your environment.\nTrack time saved, corrections and the cost of mistakes.\nHigher load needs a serving plan — queueing grew in our tests.', 13.5)
    text(.07, .094, 'Public test results ≠ production accuracy or proven ROI.\nDecision 2.0: 4 models · 70,304 saved predictions · RTX 5070 Ti 16 GB\nUnofficial, independent testing. Use as is, without warranty.', 10.5, MUTED)
    text(.07, .041, 'lokensi.github.io/laya-benchmark', 14, TEAL, 'semibold')
    text(.93, .041, evidence['comparison']['updated'][:10], 10.5, MUTED, ha='right')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    bounds = [t.get_window_extent(renderer) for t in fig.texts]
    assert all(0 <= b.x0 < b.x1 <= width and 0 <= b.y0 < b.y1 <= height for b in bounds)
    for extension in ['png', 'svg']:
        fig.savefig(OUT / f'business-summary-master.{extension}', facecolor=BG)
    plt.close(fig)
    with Image.open(OUT / 'business-summary-master.png') as image:
        assert image.size == (2160, 2700)
        image.convert('RGB').resize((1080, 1350), Image.Resampling.LANCZOS).save(OUT / 'business-summary.png')
    observed = ', '.join(f'{label}: {metric["accuracy"]:.1%}' for label, _, metric in measures)
    alt = ('Business summary of local decision-model benchmarks. Norwegian Bokmål accuracy on '
           '2,948 public MASSIVE test cases across 18 assistant-service scenarios: ' + observed + '. '
           f'A supervised TF-IDF/linear SVM reached {supervised:.1%} with labelled training data. '
           'Suggested use: local category suggestions reviewed by staff; measure time saved and '
           'corrections. Queueing grew under load. Public results do not establish production '
           'accuracy or financial savings. Four Decision 2.0 models produced 70,304 saved predictions '
           'on an RTX 5070 Ti 16 GB. Full results at ' + PAGE)
    (OUT / 'alt-text.txt').write_text(alt+'\n', encoding='utf-8')
    write_json(OUT / 'evidence.json', {
        'pages_url': PAGE, 'scope': 'MASSIVE Bokmål 18-scenario accuracy; identical 2,948 public test inputs.',
        'revision': 'completion-2026-10-04', 'updated': evidence['comparison']['updated'],
        'metrics': [{'model': label, 'deployment': kind, 'accuracy': v['accuracy'], 'n': v.get('attempted', v.get('n'))} for label, kind, v in measures],
        'supervised_baseline_accuracy': supervised,
        'sources': {k: {'path': p, 'sha256': hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for k, p in inputs.items()},
        'png_pixels': [1080, 1350], 'master_pixels': [2160, 2700], 'text_bounds': 'passed',
        'limitations': ['Different training conditions for supervised baseline.', 'No paired equivalence claim.',
                        'No measured staff-time savings or ROI.', 'Standard-run process interruptions required resumable retries.',
                        'Native stress queue has one GPU executor and no continuous batching; direct stock vLLM unsupported in tested setup.'],
    })
    print(OUT / 'business-summary.png')

if __name__ == '__main__':
    build()
