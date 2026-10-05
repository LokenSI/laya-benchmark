"""Append controlled timing coverage without changing the original frozen study."""
import argparse
import json
import shutil
import time

from .common import ROOT, read_json, digest
from .completion_state import atomic_json
from . import completion
from .tradeoff_expansion import OUT as ORIGINAL, SUITES, TIERS

OUT = completion.OUT/'timing'


def verified_unavailable(name, protocol):
    """Accept only explicitly reviewed, hash-backed repeated timing failures."""
    path = completion.OUT/'timing-failures'/f'{name}.json'
    if not path.exists():
        return {}
    evidence = read_json(path)
    assert evidence['model'] == name
    assert evidence['fixture_sha256'] == protocol['fixture_sha256']
    assert evidence['model_metadata_sha256'] == digest(ROOT/f'results/alternatives/models/{name}.json')
    unavailable = evidence['groups']
    assert set(unavailable) <= set(protocol['models'][name])
    for suite, item in unavailable.items():
        assert item['reason'] and item['failure_signature']
        assert len(item['attempts']) >= 3
        assert len({a['directory'] for a in item['attempts']}) == len(item['attempts'])
        for attempt in item['attempts']:
            directory = (ROOT/attempt['directory']).resolve()
            assert directory.is_relative_to((completion.OUT/'workers'/name/'timing').resolve())
            assert read_json(directory/'worker.json')['exit_code'] != 0
            log = directory/'worker.log'
            assert digest(log) == attempt['worker_log_sha256']
            assert item['failure_signature'] in log.read_text(encoding='utf-8', errors='replace')
    return unavailable


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ORIGINAL/'fixture.jsonl'
    protocol = read_json(ORIGINAL/'protocol.json')
    assert digest(source) == protocol['fixture_sha256']
    if (OUT/'fixture.jsonl').exists():
        assert digest(OUT/'fixture.jsonl') == digest(source)
    else:
        shutil.copyfile(source, OUT/'fixture.jsonl')
    report = read_json(ROOT/'results/jev_live/comparison.json')
    original_models = set(protocol['models'])
    names = list(protocol['models']) + ['decision2-nox-4b']
    protocol['models'] = {}
    for name in names:
        groups = []
        for suite in SUITES:
            sources = TIERS if suite == 'public_jevbench' else [suite]
            if all(report['models'][name]['suites'].get(s, {}).get('complete') for s in sources):
                groups.append(suite)
        protocol['models'][name] = groups
        # Historical complete measurements are immutable; extension workers only
        # add newly eligible blocks. Preserve raw files so all plots are auditable.
        directory = OUT/name
        if name in original_models and (ORIGINAL/name/'summary.json').exists():
            directory.mkdir(exist_ok=True)
            for path in (ORIGINAL/name).glob('*'):
                if path.suffix not in ['.json', '.jsonl']:
                    continue
                target = directory/path.name
                if not target.exists():
                    shutil.copyfile(path, target)
    protocol.update(completion_updated=completion.now(),
                    parent_protocol_sha256=digest(ORIGINAL/'protocol.json'),
                    extension='Identical fixed cases, warmups, passes and measurement policy. Eligibility expanded only when full accuracy groups complete. Historical raw timing files retained unchanged.')
    if (OUT/'protocol.json').exists():
        archive = OUT/'history'/str(time.time_ns())
        archive.mkdir(parents=True)
        for filename in ['protocol.json', 'accuracy-snapshot.json']:
            shutil.copyfile(OUT/filename, archive/filename)
    atomic_json(OUT/'protocol.json', protocol)
    atomic_json(OUT/'accuracy-snapshot.json', report)
    return protocol


def verify(name):
    protocol = read_json(OUT/'protocol.json')
    assert digest(OUT/'fixture.jsonl') == protocol['fixture_sha256']
    fixture = [json.loads(line) for line in (OUT/'fixture.jsonl').read_text(encoding='utf-8').splitlines()]
    by_id = {r['id']: r for r in fixture}
    state = read_json(OUT/name/'summary.json')
    unavailable = verified_unavailable(name, protocol)
    completed = []
    for suite in protocol['models'][name]:
        block = state['groups'].get(suite, {})
        if not block.get('complete') and suite in unavailable:
            expected_inputs = {r['id']: r['input_sha256'] for r in fixture if r['timing_suite'] == suite}
            assert unavailable[suite]['case_input_hashes'] == expected_inputs
            continue
        assert block.get('complete'), f'Missing timing block: {name}/{suite}'
        path = OUT/name/(suite.replace('/', '--')+'.jsonl')
        assert digest(path) == block['predictions_sha256']
        raw = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        wanted = {(r['id'], repeat) for r in fixture if r['timing_suite'] == suite for repeat in [1,2,3]}
        assert len(raw) == len(wanted) == block['calls']
        assert {(r['id'], r['pass']) for r in raw} == wanted
        assert all(r['input_sha256'] == by_id[r['id']]['input_sha256'] for r in raw)
        completed.append(suite)
    withheld = {suite: item['reason'] for suite, item in unavailable.items() if suite not in completed}
    if withheld != state.get('unavailable_groups', {}):
        archive = completion.OUT/'history'/f'{name}-timing-before-unavailable-{time.time_ns()}.json'
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUT/name/'summary.json', archive)
        state['unavailable_groups'] = withheld
        atomic_json(OUT/name/'summary.json', state)
    value = {'model': name, 'verified': completion.now(), 'groups': completed,
             'status': 'partial_unavailable' if withheld else 'complete',
             'unavailable_groups': withheld, 'all_requested_groups_accounted': True,
             'fixture_sha256': protocol['fixture_sha256'], 'summary_sha256': digest(OUT/name/'summary.json')}
    atomic_json(completion.OUT/'verified'/f'{name}-timing.json', value)
    return value


def measure(name):
    protocol = prepare()
    unavailable = verified_unavailable(name, protocol)
    state_path = OUT/name/'summary.json'
    state = read_json(state_path) if state_path.exists() else {'groups': {}}
    pending = [s for s in protocol['models'][name] if not state['groups'].get(s, {}).get('complete')]
    if not pending or set(pending) <= set(unavailable):
        return verify(name)
    for _ in range(3):
        attempt = completion.OUT/'workers'/name/'timing'/str(time.time_ns())
        code = completion.worker([completion.python_for(name), '-X', 'utf8', '-m',
                                  'laya_bench.tradeoff_expansion', 'local', '--model', name,
                                  '--output-root', str(OUT)], attempt)
        if code == 0:
            return verify(name)
    raise RuntimeError(f'Controlled timing failed in three fresh processes: {name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'measure', 'verify'])
    parser.add_argument('--model')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'verify': print(verify(args.model))
    else:
        with completion.controller_lock(): measure(args.model)
