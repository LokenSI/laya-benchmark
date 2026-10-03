"""GPU integration probes for new leaders, kept outside benchmark scores."""
import argparse
import time
import torch
from .common import ROOT, read_json, write_json
from .alternatives_adapters import create
from .alternatives_run import validate_answer


def run(name):
    torch.set_num_threads(8)
    torch.manual_seed(20261001)
    torch.cuda.set_per_process_memory_fraction(.78)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = [
        {'state': 'The ticket explicitly requests a password reset.', 'questions': {'q': {
            'type': 'choice', 'instructions': 'Route the request.', 'criteria': {'account': 'Password or account access', 'hardware': 'Broken physical equipment'}}}},
        {'state': 'Pumpen er ikke i drift. Den er stoppet for vedlikehold.', 'questions': {'q': {
            'type': 'noul', 'instructions': 'Er pumpen i drift?', 'criteria': {'true': 'Ja', 'false': 'Nei'}}}},
        {'state': {'priority': 'medium'}, 'questions': {'q': {'type': 'score', 'instructions': 'Use the recorded priority.', 'criteria': ['low', 'medium', 'high']}}},
        {'state': 'The ticket mentions a pump and a valve.', 'questions': {'q': {
            'type': 'multilabel', 'instructions': 'Select equipment explicitly mentioned.', 'criteria': {'pump': 'Pump', 'valve': 'Valve', 'motor': 'Motor'}}}},
    ]
    gold = [['account'], ['false'], ['1'], ['pump', 'valve']]
    started = time.perf_counter()
    adapter = create(name)
    results = []
    try:
        for row, expected in zip(rows, gold):
            answer = adapter.batch([row])[0]
            validate_answer(row, answer)
            results.append({'input': row, 'output': answer, 'expected': expected,
                            'correct': set(answer['pred']['q']) == set(expected)})
        write_json(ROOT / f'results/alternatives/leader-smoke/{name}.json', {
            'model': name, 'status': 'interface_passed', 'metadata': adapter.metadata,
            'seconds_including_load': time.perf_counter() - started, 'probes': results,
            'scope': 'Four authored integration probes; not production accuracy or a benchmark headline.'})
        print(name, 'interface passed; simple probe matches', sum(r['correct'] for r in results), '/4', flush=True)
    finally:
        if hasattr(adapter, 'close'):
            adapter.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model')
    run(p.parse_args().model)
