"""One task-specific accuracy/latency/memory figure from matched timing runs."""
import hashlib
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image

from .common import ROOT, read_json, write_json, digest
from .alternatives_jev import read_rows
from .jev_live_report import score
from .metrics import wilson

OUT = ROOT / 'results/linkedin/tradeoffs'
INK, MUTED, GRID = '#142D3C', '#50616C', '#DCE3E6'
MODELS = ['laya', 'laya-multilingual', 'decider-08b', 'decider-2b',
          'decider-4b', 'von', 'gliner-decide', 'jev-1.13.0']
NAMES = ['Laya', 'Laya multilingual', 'Decider 0.8B', 'Decider 2B',
         'Decider 4B v2.1', 'Von', 'GLiNER Decide', 'Jev 1.13.0 API']
SHORT = ['Laya', 'Laya multi', 'D0.8', 'D2', 'D4', 'Von', 'GLiNER', 'Jev']
COLORS = ['#057B91', '#54A39B', '#9B78AB', '#7541A3', '#364CA2',
          '#B98526', '#C26842', '#26333D']
SUITES = ['fresh/news', 'fresh/emotion', 'fresh/spam', 'fresh/phishing']
TITLES = ['News classification', 'Emotion tagging', 'Spam screening', 'Phishing screening']
# Offsets move labels only. All bubbles stay at their measured coordinates.
OFFSETS = [
    [(12, 11), (-5, 31), (-34, -31), (-15, 30), (23, -10), (-8, -25), (20, -17), (15, -9)],
    [(9, 17), (-25, -25), (-29, -22), (-7, 24), (24, -1), (-6, 23), (23, -27), (15, 8)],
    [(20, 18), (-19, 24), (-34, -23), (-5, 39), (26, -27), (17, -23), (16, 13), (18, -23)],
    [(23, 17), (-23, 23), (-30, -34), (-14, 25), (23, 16), (-7, -22), (20, -22), (15, -14)],
]


def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    p = ROOT / 'results/jev_live/comparison.json'
    raw = p.read_bytes()
    report = json.loads(raw)
    protocol = read_json(ROOT / 'results/latency/protocol.json')
    timing_fixture = ROOT / 'data/prepared/latency.jsonl'
    assert digest(timing_fixture) == protocol['fixture_sha256']
    timed = {r['id']: r for r in read_rows(timing_fixture)}
    fresh = read_rows(ROOT / 'data/prepared/jev_fresh.jsonl')
    test = {s: [r for r in fresh if r['suite'] == s and r['split'] == 'test'] for s in SUITES}
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'accuracy_snapshot_utc': report['updated'],
              'accuracy_report_sha256': hashlib.sha256(raw).hexdigest(),
              'protocol': protocol, 'models': {}, 'measurements': [],
              'memory_definition': 'Peak CUDA memory reserved by the PyTorch allocator during the complete 80-message, three-pass run. Includes cached allocations; excludes driver/non-PyTorch overhead and other applications. It is one workload-wide peak per model, not a per-task peak or minimum card specification.',
              'chart_definitions': {'x': 'Median milliseconds over 20 short messages per task, repeated three times (60 calls).',
                                    'y': 'Accuracy on all 200 held-out examples for that task. Timing uses a subset, restricted to <=512 state characters.',
                                    'area': 'Linear in peak_reserved_gib for local models. Jev uses a fixed diamond because hosted GPU memory is unknown.',
                                    'vertical_line': 'Wilson 95% binomial accuracy interval, descriptive and unadjusted for multiple comparisons.'},
              'limits': ['Local RTX 5070 Ti 16 GB, Windows, batch 1, loaded model, eight warm-up calls excluded.',
                         'Jev HTTPS time includes network and hosted service; hosted hardware is unknown.',
                         'Training overlap is unknown; Laya lists news, spam and phishing among its training tasks.',
                         'Only models with complete matched latency and memory runs are plotted. Newer leaderboard additions are not timed in this experiment.']}
    for model, name, short, color in zip(MODELS, NAMES, SHORT, COLORS):
        tp = ROOT / f'results/latency/{model}.jsonl'
        timings = read_rows(tp)
        assert len(timings) == 240
        assert len({(r['id'], r['pass']) for r in timings}) == 240
        assert all(r['input_sha256'] == timed[r['id']]['input_sha256'] for r in timings)
        assert all(not r['error'] and not r['truncated'] for r in timings)
        memory = read_json(ROOT / f'results/latency/{model}-memory.json') if model != 'jev-1.13.0' else None
        meta = read_json(ROOT / f'results/latency/{model}-metadata.json')
        pred_path = ROOT / ('results/jev_live/predictions.jsonl' if model == 'jev-1.13.0' else
                           f'results/alternatives/jev_fresh/{model}/predictions.jsonl')
        preds = {r['id']: r for r in read_rows(pred_path)}
        result['models'][model] = {'name': name, 'short': short, 'color': color, 'memory': memory,
                                   'metadata': meta, 'timing_sha256': digest(tp), 'predictions_sha256': digest(pred_path)}
        for suite in SUITES:
            accuracy = report['models'][model]['suites'][suite]
            assert accuracy['complete'] and accuracy['n'] == 200
            scores = [score(r, preds[r['id']]) for r in test[suite]]
            correct = sum(r['correct'] for r in scores)
            assert correct == accuracy['correct']
            sample = [r for r in timings if timed[r['id']]['suite'] == suite]
            assert len(sample) == 60 and len({r['id'] for r in sample}) == 20
            ms = np.array([r['seconds'] * 1000 for r in sample])
            result['measurements'].append({'model': model, 'suite': suite,
                'accuracy_n': 200, 'correct': correct, 'accuracy_percent': correct/2,
                'wilson95_percent': [100*x for x in wilson(correct, 200)],
                'unanswered_or_invalid': sum(not r['valid'] for r in scores),
                'timing_unique_cases': 20, 'timed_calls': 60,
                'median_ms': float(np.median(ms)), 'p95_ms': float(np.quantile(ms, .95)),
                'peak_reserved_gib': memory['peak_reserved_gib'] if memory else None,
                'peak_allocated_gib': memory['peak_allocated_gib'] if memory else None})
    write_json(OUT / 'measurements.json', result)
    return result


def draw(data):
    plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11,
                         'text.color': INK, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED,
                         'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(16, 13), dpi=180, facecolor='white')
    fig.text(.062, .966, 'Accuracy vs latency, with GPU memory', fontsize=27, weight='semibold', va='top')
    fig.text(.062, .927, 'Four use cases  |  Higher and further left is better  |  Bubble area = peak GPU memory reserved by PyTorch',
             fontsize=13, color=MUTED, va='top')
    positions = [[.065, .586, .403, .284], [.560, .586, .403, .284],
                 [.065, .242, .403, .284], [.560, .242, .403, .284]]
    labels = []
    for i, (suite, title, pos) in enumerate(zip(SUITES, TITLES, positions)):
        ax = fig.add_axes(pos)
        ax.set_xscale('log')
        ax.set_xlim(12, 450)
        ax.set_ylim(40, 112)
        ax.set_xticks([20, 50, 100, 200, 400], ['20', '50', '100', '200', '400'])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
        ax.set_ylabel('Accuracy (%)', fontsize=12, labelpad=9)
        if i >= 2:
            ax.set_xlabel('Median response time (ms, log scale)', fontsize=12, labelpad=8)
        ax.set_title(title, loc='left', fontsize=18, weight='semibold', pad=14)
        ax.grid(color=GRID, linewidth=.7)
        ax.set_axisbelow(True)
        for side in ['top', 'right']: ax.spines[side].set_visible(False)
        for side in ['bottom', 'left']: ax.spines[side].set_color(GRID)
        ax.tick_params(length=0, pad=6)
        ax.axhline(100, color='#B9C6CD', lw=.9)
        for j, model in enumerate(MODELS):
            d = next(r for r in data['measurements'] if r['suite'] == suite and r['model'] == model)
            info = data['models'][model]
            x, y = d['median_ms'], d['accuracy_percent']
            low, high = d['wilson95_percent']
            ax.errorbar(x, y, yerr=[[y-low], [high-y]], fmt='none', ecolor=info['color'],
                        elinewidth=1.1, alpha=.45, capsize=2, zorder=2)
            if d['peak_reserved_gib'] is None:
                ax.scatter(x, y, s=105, marker='D', color='white', edgecolor=info['color'], linewidth=1.6, zorder=4)
            else:
                ax.scatter(x, y, s=d['peak_reserved_gib']*82, facecolor=info['color'],
                           edgecolor=info['color'], linewidth=1.3, alpha=.48, zorder=3)
            offset = OFFSETS[i][j]
            label = ax.annotate(info['short'], (x, y), xytext=offset, textcoords='offset points',
                                ha='center', va='center', fontsize=10.5, color=INK, weight='semibold',
                                bbox={'boxstyle': 'square,pad=.16', 'fc': 'white', 'ec': 'none', 'alpha': .92},
                                arrowprops={'arrowstyle': '-', 'color': info['color'], 'lw': .7}, zorder=6)
            labels.append(label)
    # A compact key provides the exact measured memory value for each bubble.
    legend_y = [.174, .123]
    legend_x = [.065, .297, .532, .767]
    for j, model in enumerate(MODELS):
        info = data['models'][model]
        x, y = legend_x[j % 4], legend_y[j // 4]
        key = fig.add_axes([x, y-.020, .017, .022])
        key.set_axis_off()
        key.scatter([.5], [.5], s=65, marker='D' if info['memory'] is None else 'o',
                    facecolor='white' if info['memory'] is None else info['color'],
                    edgecolor=info['color'], linewidth=1.2)
        fig.text(x+.022, y, info['name'], fontsize=12, weight='semibold', va='top')
        mem = f'{info["memory"]["peak_reserved_gib"]:.2f} GiB reserved' if info['memory'] else 'Hosted GPU memory undisclosed'
        fig.text(x+.022, y-.020, mem, fontsize=10.5, color=MUTED, va='top')
    fig.text(.065, .062, 'Accuracy: 200 test cases/task; vertical lines: 95% Wilson intervals. Latency: a 20-short-message subset/task × 3 passes, state ≤512 characters.',
             fontsize=9.5, color=MUTED)
    fig.text(.065, .045, 'Memory: workload-wide peak, includes allocator cache; excludes driver/non-PyTorch overhead. Local: RTX 5070 Ti 16 GB, Windows, batch 1, warm models.',
             fontsize=9.5, color=MUTED)
    fig.text(.065, .028, 'Jev includes network/service time. Public training overlap unknown; Laya lists news/spam/phishing as training tasks. Matched runs only. 1 Oct 2026.',
             fontsize=9.5, color=MUTED)
    fig.canvas.draw()
    # Catch cropped labels before publishing. Overlaps are reviewed visually.
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    for item in fig.texts + labels:
        b = item.get_window_extent(renderer)
        assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= width and b.y1 <= height, item.get_text()
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'accuracy-latency-vram.{ext}', facecolor='white')
    plt.close(fig)
    im = Image.open(OUT / 'accuracy-latency-vram.png')
    assert im.size == (2880, 2340)
    im.resize((1920, 1560), Image.Resampling.LANCZOS).save(OUT / 'accuracy-latency-vram-linkedin.png')


def table(data):
    import html
    rows = []
    for d in data['measurements']:
        name = data['models'][d['model']]['name']
        lo, hi = d['wilson95_percent']
        mem = f'{d["peak_reserved_gib"]:.2f}' if d['peak_reserved_gib'] is not None else 'Unknown (hosted)'
        rows.append(f'<tr><td>{html.escape(name)}</td><td>{html.escape(d["suite"].split("/")[1])}</td><td>{d["correct"]}/200 · {d["accuracy_percent"]:.1f}%</td><td>{lo:.1f}–{hi:.1f}%</td><td>{d["median_ms"]:.1f}</td><td>{d["p95_ms"]:.1f}</td><td>{mem}</td></tr>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Accuracy, latency and GPU memory</title><style>body{font:15px/1.6 Segoe UI,Arial,sans-serif;color:#142d3c;margin:30px auto;padding:0 24px;max-width:1500px}img{width:100%;height:auto}a{color:#057b91}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #dce3e6}aside{background:#f0f4f5;padding:18px}h1{font-size:28px}</style><h1>Accuracy, response time and GPU memory by use case</h1><p><a href="accuracy-latency-vram-linkedin.png">LinkedIn PNG</a> · <a href="accuracy-latency-vram.png">High-resolution PNG</a> · <a href="accuracy-latency-vram.svg">Editable SVG</a> · <a href="measurements.json">Measurements and provenance</a></p><img src="accuracy-latency-vram.png" alt="Four bubble plots show median latency on a logarithmic horizontal axis and task accuracy on the vertical axis. Bubble areas represent peak PyTorch-reserved GPU memory. Jev is shown as a diamond with hosted memory unknown."><aside><b>Reading the plot</b><p>Higher accuracy is better; lower latency is better; smaller local bubbles use less measured allocator-reserved GPU memory. Accuracy uses 200 examples per task, while timing uses a 20-short-message subset repeated three times. Vertical lines are descriptive 95% Wilson intervals.</p><p>Memory is the peak across the full four-task timing run, shared across the four panels. It includes PyTorch cache and excludes driver/non-PyTorch overhead. It is not total process VRAM or a minimum card capacity. Jev has no measured local GPU footprint; its server footprint is unknown. No zero-VRAM size is implied.</p><p>Only the seven local models with a complete matched timing and memory experiment are plotted. Newer leaders require that same experiment. Latency includes adapter processing locally and HTTPS transport for Jev; no general hardware-normalized speed claim is made.</p><p>The axes show medians; the table additionally reports 95th-percentile latency. Public training overlap is unknown. Laya lists news, spam and phishing among its training tasks.</p></aside><h2>Exact values</h2><table><tr><th>Model</th><th>Task</th><th>Accuracy</th><th>95% interval</th><th>Median ms</th><th>P95 ms</th><th>Peak reserved GiB</th></tr>'''+''.join(rows)+'</table></html>'
    (OUT / 'index.html').write_text(page, encoding='utf-8')


def build():
    data = collect()
    draw(data)
    table(data)
    print(f'Created one four-panel chart from {len(data["measurements"])} verified model/task measurements: {OUT}')


if __name__ == '__main__':
    build()
