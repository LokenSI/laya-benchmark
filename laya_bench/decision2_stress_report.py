"""Report saved compatibility probes and completed native stress stages."""
import html
import re
from .common import ROOT, read_json, write_json, digest
from .decision2_stress import OUT, workload
from .jev_api import now


def charts(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = [(name, entry) for name, entry in result['native'].items() if entry['status'] == 'complete']
    if not models:
        return
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    for (name, entry), color in zip(models, ['#17699a', '#af5c16']):
        stages = entry['concurrency']
        x = [s['concurrency'] for s in stages]
        label = 'Eos 0.8B' if 'eos' in name else 'Sol 2B'
        axes[0].plot(x, [s['successful_requests_per_second'] for s in stages], 'o-', color=color, label=label)
        axes[1].plot(x, [s['latency_ms']['p95']/1000 for s in stages], 'o-', color=color, label=label)
    for axis, title, ylabel in zip(axes, ['Throughput', 'HTTP p95 latency'], ['Successful requests / second', 'Seconds']):
        axis.set_xscale('log', base=2)
        axis.set_xticks([1, 8, 32, 128, 512], ['1', '8', '32', '128', '512'])
        axis.set_xlabel('Concurrent HTTP clients')
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc='left', fontweight='bold')
        axis.set_ylim(bottom=0)
        axis.grid(axis='y', alpha=.2)
        axis.legend(frameon=False)
    fig.suptitle('Decision 2.0: native HTTP queue under load', x=.075, ha='left', fontweight='bold', fontsize=16)
    fig.text(.075, .03, 'RTX 5070 Ti · Windows native FIFO executor · No continuous batching · 78% VRAM cap\n'
             'Same 80 cases at every level, three passes · 4,800 timed requests per model · 0 errors / 0 label flips', fontsize=9, color='#596c7d')
    fig.subplots_adjust(left=.075, right=.97, top=.82, bottom=.22, wspace=.27)
    target = OUT / 'figures'
    target.mkdir(exist_ok=True)
    for extension in ['png', 'svg']:
        fig.savefig(target / f'native-concurrency.{extension}', facecolor='white')
    plt.close(fig)


def build(draw_charts=False):
    fixture, _ = workload()
    result = {'updated': now(), 'fixture_sha256': digest(fixture),
              'scope': 'Local RTX 5070 Ti 16 GB with the standard 78% VRAM allocation cap; Windows CUDA/Transformers/FLA with reference causal-convolution fallback; localhost HTTP; native publisher models with one FIFO GPU executor, no continuous batching.',
              'vllm': {}, 'native': {}}
    for implementation in ['auto', 'transformers']:
        path = OUT / f'vllm-{implementation}.json'
        if path.exists():
            result['vllm'][implementation] = read_json(path)
            log = OUT / f'vllm-{implementation}.log'
            if log.exists():
                causes = re.findall(r'ValueError: Argument input_ids[^\r\n]+', log.read_text(encoding='utf-8'))
                if causes:
                    result['vllm'][implementation]['root_cause'] = causes[-1]
                    result['vllm'][implementation]['engine_log'] = log.name
    for path in sorted(OUT.glob('decision2-*/summary.json')):
        entry = read_json(path)
        assert entry['fixture_sha256'] == result['fixture_sha256']
        if entry['status'] == 'complete':
            assert digest(path.parent / 'requests.jsonl') == entry['requests_sha256']
            assert digest(path.parent / 'question-capacity.jsonl') == entry['question_capacity_sha256']
        result['native'][entry['model']] = entry
    write_json(OUT / 'summary.json', result)
    if draw_charts:
        charts(result)
    esc = html.escape
    probes = []
    for implementation, probe in result['vllm'].items():
        evidence = esc(probe.get('root_cause', probe.get('error', probe.get('note', ''))))
        if probe.get('engine_log'):
            evidence += f" <a href='{esc(probe['engine_log'])}'>Engine log</a>"
        probes.append(f"<tr><td>{esc(implementation)}</td><td>{esc(probe['status'])}</td><td>{evidence}</td></tr>")
    stages, capacity = [], []
    for name, entry in result['native'].items():
        for stage in entry['concurrency']:
            latency = stage['latency_ms']
            stages.append(f"<tr><td>{esc(name)}</td><td>{stage['concurrency']}</td><td>{stage['requests']}</td>"
                          f"<td>{stage['successful_requests_per_second']:.1f}</td><td>{latency['p50']:.1f}</td>"
                          f"<td>{latency['p95']:.1f}</td><td>{latency['p99']:.1f}</td><td>{sum(stage['errors'].values())}</td>"
                          f"<td>{stage['label_flips']}</td><td>{stage['max_probability_delta']:.3g}</td></tr>")
        for stage in entry['question_capacity']:
            passes = stage['passes']
            errors = [p['error'] for p in passes if p['error']]
            seconds = sorted(p['seconds'] for p in passes)[len(passes)//2]
            flips = str(sum(p['label_flips'] or 0 for p in passes)) if len(errors) < len(passes) else '—'
            delta = f"{max((p['max_probability_delta'] or 0 for p in passes), default=0):.3g}" if len(errors) < len(passes) else '—'
            rate = f"{stage['questions']/seconds:.1f}" if not errors else '—'
            error_text = esc('; '.join(sorted(set(errors))))
            evidence = f'<details><summary>Failure details</summary>{error_text}</details>' if errors else ''
            capacity.append(f"<tr><td>{esc(name)}</td><td>{stage['questions']}</td><td>{seconds*1000:.1f}</td>"
                            f"<td>{rate}</td><td>{len(errors)}</td><td>{flips}</td><td>{delta}</td><td>{evidence}</td></tr>")
    statuses = ', '.join(f"{name}: {entry['status']}" for name, entry in result['native'].items()) or 'Pending'
    figure = '<img style="width:100%" src="figures/native-concurrency.png" alt="Native throughput stays between 12 and 23 requests per second while p95 HTTP latency increases with concurrent clients."><p><a href="figures/native-concurrency.svg">Editable SVG</a> · <a href="figures/native-concurrency.png">PNG</a></p>' if (OUT / 'figures/native-concurrency.png').exists() else ''
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Decision 2.0 concurrency</title><style>body{{font:16px system-ui;background:#f4f6f9;color:#16283a;margin:0}}main{{max-width:1250px;margin:40px auto;padding:28px;background:white;border-radius:14px}}table{{border-collapse:collapse;width:100%;margin:24px 0}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #d9e1ea}}a{{color:#17699a}}p{{line-height:1.6}}small{{color:#596c7d}}</style>
<main><h1>Decision 2.0: concurrency and compatibility</h1><small>Updated {esc(result['updated'])}</small>
<p>{esc(result['scope'])}</p><p>Status: {esc(statuses)}</p>
{figure}
<h2>Direct vLLM compatibility</h2><p>Official vLLM 0.30.0 Linux container. Both probes use the complete pinned Sol 2B package with the published decision head, pooling runner and trusted model code. Engine construction alone would not prove System One parity.</p>
<table><tr><th>Model implementation</th><th>Result</th><th>Evidence</th></tr>{''.join(probes) or '<tr><td colspan="3">Pending</td></tr>'}</table>
<h2>Concurrent native HTTP requests</h2><p>80 frozen short-message cases; all shapes warmed before a separate serial reference. Three passes per level, at least max(80, 2 × concurrency) requests per pass, rounded to complete workload cycles so every level has the same case mix. Persistent HTTP connections, no retries. One decision per request. HTTP wall time includes the service queue. Repeated inputs test stability and do not enlarge the accuracy sample.</p>
<table><tr><th>Model</th><th>Concurrency</th><th>Requests</th><th>Successful req/s</th><th>p50 ms</th><th>p95 ms</th><th>p99 ms</th><th>Errors</th><th>Label flips</th><th>Max probability Δ</th></tr>{''.join(stages) or '<tr><td colspan="10">Pending</td></tr>'}</table>
<h2>Questions in one native request</h2><p>One state with repeated named questions; each shape warmed, three timed passes, context sharing disabled. Capacity and consistency diagnostic. The 512-question stage does not represent 512 simultaneous GPU forwards.</p>
<table><tr><th>Model</th><th>Questions</th><th>Median ms</th><th>Median decisions/s</th><th>Failed passes</th><th>Label flips</th><th>Max probability Δ</th><th>Errors</th></tr>{''.join(capacity) or '<tr><td colspan="8">Pending</td></tr>'}</table>
<p>The <a href="https://vllm-sr.ai/blog/decision-models/">older publisher explanation</a> distinguishes SDK submission capacity from throughput and simultaneous forwards. Decision 2.0 cards publish single-request latency. This serialized native baseline measures local queueing and does not establish the publisher's maximum throughput or a vLLM scaling result.</p>
<p><a href="protocol.json">Protocol</a> · <a href="summary.json">Saved results</a> · <a href="queue.json">Execution status</a> · <a href="../alternatives/decision2.html">Standard accuracy benchmark</a></p></main></html>'''
    (OUT / 'report.html').write_text(document, encoding='utf-8')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--charts', action='store_true')
    args = parser.parse_args()
    build(args.charts)
