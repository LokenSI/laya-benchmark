"""Final local evidence checks before publishing aggregate completion results."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laya_bench import completion, completion_timing
from laya_bench.common import ROOT, digest, read_json
from laya_bench.completion_state import atomic_json, load_records, record_hash, runtime_failure


def verify():
    with completion.controller_lock():
        completion.assert_no_worker()
        check = completion.audit()
        assert check['missing'] == 0, 'Missing cases prevent final publication'
        assert all(v['protected_predictions_unchanged'] for v in check['models'].values())
        frozen = completion.fixtures()
        bounded_failures = recovered = archives = 0
        for name in check['models']:
            for fixture, rows in frozen.items():
                expected = {row['id']: row for row in rows}
                current = load_records(completion.prediction_path(name, fixture), expected)
                path = completion.OUT/'work'/name/fixture/'runtime-retries.json'
                state = read_json(path) if path.exists() else {'failures': {}, 'blocked': [], 'attempts': []}
                for key, row in current.items():
                    if not runtime_failure(row):
                        continue
                    failures = [a for a in state['attempts'] if key in a['failed_case_ids']]
                    assert key in state['blocked'] and state['failures'].get(key, 0) >= 3
                    assert len({a['directory'] for a in failures}) >= 3, (name, fixture, key)
                    for attempt in failures:
                        directory = (ROOT/attempt['directory']).resolve()
                        assert directory.is_relative_to((completion.OUT/'workers'/name/fixture/'runtime-retries').resolve())
                        worker = read_json(directory/'worker.json')
                        assert worker['status'] in {'finished', 'crashed'}
                        assert (directory/'worker.log').exists()
                        tried = load_records(directory/'retry-predictions.jsonl', expected)
                        if key in tried:
                            assert tried[key].get('error'), 'Valid retry was left unresolved'
                        else:
                            assert worker['exit_code'] != 0
                            assert key in read_json(directory/'case-ids.json')
                    bounded_failures += 1
                for attempt in state['attempts']:
                    if not attempt['recovered']:
                        continue
                    directory = ROOT/attempt['directory']
                    originals = list(directory.glob('before-runtime-retry-*.jsonl'))
                    assert len(originals) == 1, directory
                    original = load_records(originals[0], expected)
                    retry = load_records(directory/'retry-predictions.jsonl', expected)
                    replaced = {key: row for key, row in retry.items() if not row.get('error')}
                    assert len(replaced) == attempt['recovered']
                    for key, row in replaced.items():
                        assert runtime_failure(original[key])
                        assert record_hash(row) == record_hash(current[key])
                    for key, row in original.items():
                        if not runtime_failure(row):
                            assert record_hash(row) == record_hash(current[key])
                    recovered += len(replaced)
                    archives += 1
        assert bounded_failures == check['runtime_failures']
        protocol = read_json(completion_timing.OUT/'protocol.json')
        original_root = completion_timing.ORIGINAL
        assert digest(original_root/'protocol.json') == protocol['parent_protocol_sha256']
        assert digest(original_root/'fixture.jsonl') == digest(completion_timing.OUT/'fixture.jsonl')
        historical_blocks = 0
        for directory in original_root.iterdir():
            if not directory.is_dir() or not (directory/'summary.json').exists():
                continue
            original = read_json(directory/'summary.json')
            current = read_json(completion_timing.OUT/directory.name/'summary.json')
            for suite, block in original['groups'].items():
                if not block.get('complete'):
                    continue
                filename = suite.replace('/', '--')+'.jsonl'
                assert digest(directory/filename) == block['predictions_sha256']
                assert digest(completion_timing.OUT/directory.name/filename) == block['predictions_sha256']
                assert current['groups'][suite] == block
                historical_blocks += 1
        timings = {name: completion_timing.verify(name) for name in completion.ORDER}
        for path in (completion.OUT/'restored').glob('*.json'):
            if path.stem.endswith('-prune'):
                continue
            value = read_json(path)
            if value['model'] not in completion.KEEP:
                assert value.get('pruned'), f'Temporary checkpoint not pruned: {path.stem}'
                assert not any(Path(item['path']).exists() for item in value['files'])
        jev = read_json(completion.OUT/'jev-verification.json')
        assert jev['remaining_errors'] == 0 and jev['protected_predictions_unchanged']
        assert digest(ROOT/'results/jev_live/predictions.jsonl') == jev['predictions_sha256']
        retained = {}
        for name in ['decider-4b', 'decision2-nox-4b']:
            metadata = read_json(ROOT/'results/alternatives/models'/f'{name}.json')
            weights = list(Path(metadata['path']).rglob('*.safetensors'))
            assert weights and all(path.stat().st_size > 0 for path in weights)
            retained[name] = len(weights)
        for environment in ['.venv', '.venv-decision']:
            assert (ROOT/environment/'Scripts/python.exe').is_file()
        result = {
            'updated': completion.now(), 'status': 'verified_with_disclosed_failures',
            'local_models': len(check['models']), 'cases_per_model': check['cases_per_model'],
            'local_records': sum(v['recorded'] for v in check['models'].values()),
            'missing_cases': check['missing'], 'protected_predictions_unchanged': True,
            'remaining_runtime_failures_with_at_least_three_attempts': bounded_failures,
            'runtime_failures_recovered': recovered, 'original_retry_archives_verified': archives,
            'historical_timing_blocks_unchanged': historical_blocks,
            'completion_queue_timing': {name: {'status': v['status'], 'complete_groups': len(v['groups']),
                                              'unavailable_groups': v['unavailable_groups']} for name, v in timings.items()},
            'free_disk_gib': completion.disk_check()/2**30,
            'temporary_restored_weights_removed': True,
            'retained_model_weight_files': retained, 'jev_and_local_environments_retained': True,
            'audit_sha256': digest(completion.OUT/'audit.json'),
            'prediction_sha256': {name: {fixture: block['predictions_sha256']
                                         for fixture, block in model['fixtures'].items()}
                                  for name, model in check['models'].items()},
            'jev_predictions_sha256': jev['predictions_sha256'],
            'verifier_sha256': digest(Path(__file__)),
            'limits': 'Saved runtime failures, native rejections and abstentions are not successful answers. '
                      'JevK5 developer-tool timing is unavailable; seven historical Winnow timing gaps are outside the restoration queue. '
                      'This verifies execution evidence, not label quality, training independence or production suitability.',
        }
        atomic_json(completion.OUT/'final-verification.json', result)
        print(json.dumps(result, indent=2))
        return result


if __name__ == '__main__':
    verify()
