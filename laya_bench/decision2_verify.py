"""Verify complete saved native evaluations and HTTP stress observations."""
import json
from collections import Counter
from .common import ROOT, read_json, write_json, digest, fingerprint
from .decision2_prepare import MODELS
from .alternatives_run import validate_answer
from .decision2_stress import OUT as STRESS, workload, compare
from .jev_api import now


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def verify():
    out = ROOT / 'results/alternatives'
    result = {'verified': now(), 'status': 'complete', 'models': {}, 'stress': {},
              'saved_case_predictions': 0, 'saved_prediction_errors': 0,
              'runtime_events': [], 'note': 'Final prediction coverage and process stability are separate; interrupted attempts remain in supervisor logs.'}
    for name in MODELS:
        model = read_json(out / f'models/{name}.json')
        result['models'][name] = {}
        for kind in ['jev_verified', 'jev_fresh', 'alternatives', 'alternatives_claims', 'alternatives_typed']:
            fixture = ROOT / f'data/prepared/{kind}.jsonl'
            expected = {row['id']: row for row in records(fixture)}
            folder = 'runs' if kind == 'alternatives' else kind
            directory = out / folder / name
            predpath = directory / 'predictions.jsonl'
            predictions = records(predpath)
            meta = read_json(directory / 'metadata.json')
            assert len(predictions) == len(expected) == len({p['id'] for p in predictions})
            assert meta['fixture_sha256'] == digest(fixture)
            assert meta['predictions_sha256'] == digest(predpath)
            assert meta['adapter']['revision'] == model['revision']
            assert meta['adapter']['verified_files'] == model['verified_files']
            for prediction in predictions:
                row = expected[prediction['id']]
                assert prediction['input_sha256'] == row['input_sha256']
                assert prediction['gold'] == row['gold'] and prediction['model'] == name
                correct = set(prediction['pred']) == set(row['gold']) and all(
                    set(prediction['pred'][key]) == set(gold) for key, gold in row['gold'].items())
                assert prediction['correct'] == correct
                if not prediction.get('error'):
                    validate_answer(row, prediction)
            errors = sum(bool(p.get('error')) for p in predictions)
            result['models'][name][kind] = {'cases': len(predictions), 'errors': errors,
                                           'fixture_sha256': digest(fixture),
                                           'predictions_sha256': digest(predpath)}
            result['saved_case_predictions'] += len(predictions)
            result['saved_prediction_errors'] += errors
    for filename in ['queue-decision2.json', 'queue-decision2-retry.json']:
        queue = read_json(out / filename)
        for job in queue['jobs']:
            if job.get('exit_code', 0):
                result['runtime_events'].append({'queue': filename, **job})
    _, rows = workload()
    for path in sorted(STRESS.glob('decision2-*/summary.json')):
        summary = read_json(path)
        assert summary['status'] == 'complete'
        directory = path.parent
        assert digest(directory / 'requests.jsonl') == summary['requests_sha256']
        assert digest(directory / 'question-capacity.jsonl') == summary['question_capacity_sha256']
        baseline = read_json(directory / 'serial-reference.json')
        requests = records(directory / 'requests.jsonl')
        for item in requests:
            row = rows[item['index'] % len(rows)]
            assert item['id'] == row['id'] and item['input_sha256'] == row['input_sha256']
            if not item['answer'].get('error'):
                validate_answer(row, item['answer'])
            drift = compare(baseline[row['id']], item['answer'])
            assert all(item[key] == value for key, value in drift.items())
        for stage in summary['concurrency']:
            observed = [r for r in requests if r['concurrency'] == stage['concurrency']]
            assert len(observed) == stage['requests']
            assert dict(Counter(r['answer']['error'] for r in observed if r['answer'].get('error'))) == stage['errors']
            assert sum(r['labels_changed'] is True for r in observed) == stage['label_flips']
            for repeat in stage['passes']:
                assert sum(r['pass'] == repeat['pass'] for r in observed) == repeat['requests']
                assert repeat['peak_client_concurrency'] == stage['concurrency']
        capacity = records(directory / 'question-capacity.jsonl')
        assert len(capacity) == 18
        template = next(iter(rows[0]['questions'].values()))
        for item in capacity:
            probe = {'state': rows[0]['state'], 'questions': {f'q{i}': template for i in range(item['questions'])}}
            assert item['payload_sha256'] == fingerprint(probe)
            if not item['answer'].get('error'):
                validate_answer(probe, item['answer'])
        result['stress'][summary['model']] = {'requests': len(requests),
                                             'errors': sum(bool(r['answer'].get('error')) for r in requests),
                                             'label_flips': sum(r['labels_changed'] is True for r in requests),
                                             'requests_sha256': summary['requests_sha256'],
                                             'question_capacity_sha256': summary['question_capacity_sha256']}
    write_json(out / 'decision2-verification.json', result)
    print(json.dumps({key: result[key] for key in ['status', 'saved_case_predictions', 'saved_prediction_errors', 'stress']}, indent=2))
    return result


if __name__ == '__main__':
    verify()
