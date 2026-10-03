"""Offline native Decision 2.0 HTTP stress test on the frozen latency workload.

The native baseline serializes GPU calls. It measures queueing and capacity,
not vLLM continuous batching or the publisher's maximum achievable throughput.
"""
import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

from .common import ROOT, read_json, write_json, digest, fingerprint
from .alternatives_run import validate_answer, expected_failure
from .jev_api import now

OUT = ROOT / 'results/decision2-stress'
LEVELS = [1, 8, 32, 128, 512]

def workload():
    path = ROOT / 'data/prepared/latency.jsonl'
    assert digest(path) == read_json(ROOT / 'results/latency/protocol.json')['fixture_sha256']
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(rows) == 80
    return path, rows

def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    return values[low] + (values[math.ceil(position)] - values[low]) * (position - low)

def compare(reference, result):
    if result.get('error') or reference.get('error'):
        return {'labels_changed': None, 'max_probability_delta': None}
    if set(result['probabilities']) != set(reference['probabilities']):
        raise ValueError('Response question roster changed')
    deltas = []
    for key, probabilities in reference['probabilities'].items():
        if set(probabilities) != set(result['probabilities'][key]):
            raise ValueError('Response option roster changed')
        deltas.extend(abs(value - result['probabilities'][key][option]) for option, value in probabilities.items())
    return {'labels_changed': result['pred'] != reference['pred'], 'max_probability_delta': max(deltas, default=0)}

def prepare():
    path, rows = workload()
    OUT.mkdir(exist_ok=True)
    write_json(OUT / 'protocol.json', {
        'created': now(), 'fixture_sha256': digest(path), 'unique_cases': len(rows),
        'concurrency_levels': LEVELS, 'passes': 3, 'requests_per_pass': '80 * ceil(max(80, 2*concurrency)/80); complete workload cycles at every level',
        'warmup': 'All 80 cases run once before serial reference; initial compilation excluded.',
        'client': 'Persistent localhost HTTP session; closed-loop bounded concurrency; no timed retries; timeout 180 seconds.',
        'native_service': 'One loaded publisher exact-path model; one GPU executor; bounded FIFO queue of 1024 requests. HTTP wall time includes queueing. This baseline does not perform continuous batching.',
        'gpu_allocation_cap': '78% of local VRAM, matching the standard benchmark; resource failures remain visible.',
        'question_capacity': '1, 4, 16, 64, 128, 512 named questions about one state; repeated existing question templates, share_context=False. Capacity/consistency diagnostic, not an accuracy benchmark.',
        'question_capacity_repeats': 3,
        'metrics': ['requests/s', 'decisions/s', 'p50/p95/p99 HTTP latency', 'native compute time',
                    'queue wait', 'failures', 'peak client concurrency', 'answer flips', 'probability drift', 'GPU peak allocated/reserved'],
        'claims': {'source': 'https://vllm-sr.ai/blog/decision-models/',
                   'scope': 'Decision 1.0 Kai/Lex: up to 128 requests and 512 decisions in one SDK submission; explicitly not simultaneous forwards or measured throughput. Decision 2.0 cards publish single-request latency.'},
        'limits': ['Local 16 GB RTX 5070 Ti; different hardware/runtime from publisher speed results.',
                   'Native service and direct-vLLM compatibility are reported separately.',
                   'Repeated cases measure stability; they do not increase the accuracy sample size.']})

async def serve(name, port):
    import torch
    from aiohttp import web
    from .alternatives_adapters import create
    torch.set_num_threads(8)
    torch.manual_seed(20261001)
    torch.cuda.set_per_process_memory_fraction(.78)
    torch.backends.cuda.matmul.allow_tf32 = False
    adapter = create(name)
    executor = ThreadPoolExecutor(max_workers=1)
    queue = asyncio.Queue(maxsize=1024)
    def infer(payload, queued):
        started = time.perf_counter()
        torch.cuda.synchronize()
        try:
            answer = adapter.batch([payload])[0]
            validate_answer(payload, answer)
        except Exception as error:
            if not expected_failure(error, torch):
                raise
            answer = {'error': type(error).__name__ + ': ' + str(error), 'pred': {}, 'probabilities': {}}
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        answer['service_timing'] = {'queue_seconds': started - queued,
                                    'compute_seconds': time.perf_counter() - started}
        return answer
    async def worker():
        while True:
            payload, queued, future = await queue.get()
            try:
                answer = await asyncio.get_running_loop().run_in_executor(executor, infer, payload, queued)
                if not future.done():
                    future.set_result(answer)
            except Exception as error:
                if not future.done():
                    future.set_exception(error)
            finally:
                queue.task_done()
    async def decide(request):
        payload = await request.json()
        if set(payload) != {'model', 'state', 'questions'} or payload['model'] != name:
            return web.json_response({'error': 'Invalid model or request shape'}, status=400)
        future = asyncio.get_running_loop().create_future()
        try:
            queue.put_nowait(({'state': payload['state'], 'questions': payload['questions']}, time.perf_counter(), future))
        except asyncio.QueueFull:
            return web.json_response({'error': 'Queue capacity exceeded'}, status=503)
        answer = await future
        return web.json_response(answer, status=422 if answer.get('error') else 200)
    async def status(request):
        return web.json_response({'model': name, 'queued_requests': queue.qsize(),
                                  'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                                  'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30})
    app = web.Application(client_max_size=32*1024*1024)
    app.router.add_post('/v1/systemone', decide)
    app.router.add_get('/status', status)
    task = asyncio.create_task(worker())
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', port)
    await site.start()
    write_json(OUT / f'{name}-server.json', {'model': name, 'port': port, 'started': now(),
                                           'adapter': adapter.metadata, 'gpu': torch.cuda.get_device_name(0),
                                           'scheduler': 'Single native GPU executor with FIFO queue; no continuous batching'})
    try:
        await asyncio.Event().wait()
    finally:
        task.cancel()
        await runner.cleanup()
        executor.shutdown()

async def run(name, port):
    import aiohttp
    path, rows = workload()
    protocol = read_json(OUT / 'protocol.json')
    assert digest(path) == protocol['fixture_sha256']
    directory = OUT / name
    directory.mkdir(exist_ok=False)
    baseline = {}
    summary = {'model': name, 'started': now(), 'status': 'running', 'fixture_sha256': digest(path),
               'transport': 'localhost HTTP', 'concurrency': [], 'question_capacity': []}
    url = f'http://127.0.0.1:{port}'
    def save():
        write_json(directory / 'summary.json', summary)
    save()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=512),
                                     timeout=aiohttp.ClientTimeout(total=180), trust_env=False) as session:
        async def request(row):
            payload = {'model': name, 'state': row['state'], 'questions': row['questions']}
            start = time.perf_counter()
            try:
                async with session.post(url + '/v1/systemone', json=payload) as response:
                    answer = await response.json()
                    if response.status != 200:
                        answer.setdefault('error', 'HTTP ' + str(response.status))
                    if not answer.get('error'):
                        validate_answer(row, answer)
                    return answer, time.perf_counter() - start
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                return {'error': type(error).__name__ + ': ' + str(error)}, time.perf_counter() - start
        # Compile/warm every input shape before any timed measurements.
        for row in rows:
            answer, _ = await request(row)
            if answer.get('error'):
                raise RuntimeError('Warmup failed: ' + answer['error'])
        for row in rows:
            answer, elapsed = await request(row)
            if answer.get('error'):
                raise RuntimeError('Serial reference failed: ' + answer['error'])
            baseline[row['id']] = answer
        write_json(directory / 'serial-reference.json', baseline)
        for concurrency in LEVELS:
            count = len(rows) * math.ceil(max(len(rows), 2 * concurrency) / len(rows))
            semaphore = asyncio.Semaphore(concurrency)
            observations = []
            pass_results = []
            for repeat in range(3):
                current = peak = 0
                async def execute(index):
                    nonlocal current, peak
                    row = rows[index % len(rows)]
                    async with semaphore:
                        current += 1
                        peak = max(peak, current)
                        try:
                            answer, elapsed = await request(row)
                            drift = compare(baseline[row['id']], answer)
                            return {'id': row['id'], 'input_sha256': row['input_sha256'],
                                    'concurrency': concurrency, 'pass': repeat+1, 'index': index,
                                    'seconds': elapsed, 'answer': answer, **drift}
                        finally:
                            current -= 1
                start = time.perf_counter()
                observed = await asyncio.gather(*(execute(i) for i in range(count)))
                duration = time.perf_counter() - start
                observations.extend(observed)
                pass_results.append({'pass': repeat+1, 'requests': count, 'seconds': duration,
                                     'requests_per_second': count/duration, 'peak_client_concurrency': peak})
                with (directory / 'requests.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(''.join(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n' for item in observed))
            errors = Counter(item['answer'].get('error') for item in observations if item['answer'].get('error'))
            timings = [item['seconds'] for item in observations]
            duration = sum(p['seconds'] for p in pass_results)
            entry = {'concurrency': concurrency, 'requests': len(observations), 'passes': pass_results,
                     'successful_requests_per_second': (len(observations)-sum(errors.values()))/duration,
                     'successful_decisions_per_second': (len(observations)-sum(errors.values()))/duration,
                     'latency_ms': {label: percentile(timings, fraction)*1000 for label, fraction in
                                    [('p50', .5), ('p95', .95), ('p99', .99)]},
                     'errors': dict(errors), 'label_flips': sum(item['labels_changed'] is True for item in observations),
                     'max_probability_delta': max((item['max_probability_delta'] or 0 for item in observations), default=0)}
            for field in ['queue_seconds', 'compute_seconds']:
                values = [item['answer']['service_timing'][field] for item in observations
                          if 'service_timing' in item['answer']]
                entry[field] = {label: percentile(values, fraction) for label, fraction in
                                [('p50', .5), ('p95', .95), ('p99', .99)]}
            summary['concurrency'].append(entry)
            save()
            print(name, 'concurrency', concurrency, entry, flush=True)
        # One state, increasingly many named questions. Existing templates are
        # deliberately repeated: this checks submission capacity and exact-path
        # stability, not useful-question diversity or generalization accuracy.
        row = rows[0]
        template = next(iter(row['questions'].values()))
        expected = next(iter(baseline[row['id']]['probabilities'].values()))
        expected_label = next(iter(baseline[row['id']]['pred'].values()))
        for count in [1, 4, 16, 64, 128, 512]:
            probe = {'state': row['state'], 'questions': {f'q{i}': template for i in range(count)}}
            await request(probe)  # Warm this question-batch shape; failures retained below.
            results = []
            for repeat in range(3):
                answer, elapsed = await request(probe)
                delta = None if answer.get('error') else max(abs(answer['probabilities'][key][option]-value)
                              for key in probe['questions'] for option, value in expected.items())
                results.append({'pass': repeat+1, 'seconds': elapsed, 'error': answer.get('error'),
                                'max_probability_delta': delta,
                                'label_flips': None if answer.get('error') else sum(answer['pred'][key] != expected_label for key in probe['questions']),
                                'decisions_per_second': count/elapsed if not answer.get('error') else 0})
                with (directory / 'question-capacity.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'questions': count, 'pass': repeat+1,
                                             'payload_sha256': fingerprint(probe),
                                             'answer': answer, **results[-1]},
                                            ensure_ascii=False, allow_nan=False)+'\n')
            summary['question_capacity'].append({'questions': count, 'passes': results})
            save()
            print(name, 'questions per request', count, results, flush=True)
        async with session.get(url + '/status') as response:
            summary['server_final'] = await response.json()
    summary.update(status='complete', finished=now(), requests_sha256=digest(directory / 'requests.jsonl'),
                   question_capacity_sha256=digest(directory / 'question-capacity.jsonl'))
    save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'serve', 'run'])
    parser.add_argument('--model')
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        try:
            asyncio.run((serve if args.action == 'serve' else run)(args.model, args.port))
        except Exception as error:
            if args.action == 'run':
                target = OUT / args.model / 'summary.json'
                if target.exists():
                    result = read_json(target)
                    result.update(status='failed', error=type(error).__name__+': '+str(error), finished=now())
                    write_json(target, result)
            raise
        finally:
            if args.action == 'run':
                from .decision2_stress_report import build
                build()
