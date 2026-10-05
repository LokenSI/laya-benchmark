"""Complete frozen benchmark evidence in isolated, resumable Windows workers.

Audit and prepare never import torch or instantiate a model. Completion comes
from validated prediction IDs, not historical supervisor state.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .common import ROOT, digest, fingerprint, read_json
from .completion_state import (atomic_json, load_records, merge_retries,
                               record_hash, runtime_failure, select_pending)

OUT = ROOT / 'results/completion'
FIXTURES = ['alternatives', 'alternatives_claims', 'alternatives_typed', 'jev_verified', 'jev_fresh']
ORDER = ['wald-4b-v12', 'plumb-4b', 'cygnet-12b-nf4', 'jevk5',
         'decision-4b-v12', 'imajev-2b', 'imajev-4b', 'intern-decision-4b',
         'jev-omni-12b-nf4', 'clm-int8', 'gliner-decide-multi',
         'gliner-decide', 'nimble-9b', 'tev1']
KEEP = {'decider-4b', 'decision2-nox-4b', 'jev-1.13.0'}
FLOOR = 50 * 2**30


def now():
    return datetime.now(timezone.utc).isoformat()


def python_for(name):
    env = '.venv' if name in ['gliner-decide', 'laya', 'laya-multilingual', 'clm-int8'] else '.venv-decision'
    return str(ROOT / env / 'Scripts/python.exe')


def prediction_path(name, fixture):
    group = 'runs' if fixture == 'alternatives' else fixture
    return ROOT / f'results/alternatives/{group}/{name}/predictions.jsonl'


def fixtures():
    result = {}
    for name in FIXTURES:
        path = ROOT / f'data/prepared/{name}.jsonl'
        protocol = ROOT / ('results/alternatives/protocol.json' if name == 'alternatives'
                           else f'results/alternatives/{name}-protocol.json')
        assert digest(path) == read_json(protocol)['fixture_sha256'], name
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert len(rows) == len({r['id'] for r in rows}), name
        for row in rows:
            assert row['input_sha256'] == fingerprint({'state': row['state'], 'questions': row['questions']}), row['id']
        result[name] = rows
    assert sum(map(len, result.values())) == 17576
    return result


def classify(row):
    if runtime_failure(row):
        return 'runtime_failures'
    if row.get('error'):
        # Saved ValueErrors must remain visible, including native option limits.
        return 'native_rejections' if row['error'].startswith('ValueError:') else 'other_failures'
    if any(row.get('abstained', {}).values()):
        return 'abstentions'
    return 'answered'


def audit_model(name, frozen, repair=False, baseline=False):
    from .alternatives_run import validate_answer
    summary = {'model': name, 'fixtures': {}, 'expected': 17576}
    baseline_path = OUT / 'baselines' / f'{name}.json'
    old = read_json(baseline_path) if baseline_path.exists() else None
    protected = {}
    for fixture, rows in frozen.items():
        expected = {r['id']: r for r in rows}
        path = prediction_path(name, fixture)
        saved = load_records(path, expected, repair_tail=repair)
        for key, row in saved.items():
            if not row.get('error'):
                validate_answer(expected[key], row)
            gold = expected[key]['gold']
            correct = set(row['pred']) == set(gold) and all(set(row['pred'][k]) == set(v) for k, v in gold.items())
            assert row['correct'] == correct, f'Saved correctness mismatch: {name}/{key}'
        counts = Counter(classify(r) for r in saved.values())
        protected[fixture] = {key: record_hash(row) for key, row in saved.items() if not runtime_failure(row)}
        if old:
            for key, value in old['records'][fixture].items():
                assert key in saved and record_hash(saved[key]) == value, f'Protected result changed: {name}/{fixture}/{key}'
        summary['fixtures'][fixture] = {
            'expected': len(rows), 'recorded': len(saved), 'missing': len(rows)-len(saved),
            **{k: counts[k] for k in ['answered', 'abstentions', 'native_rejections', 'runtime_failures', 'other_failures']},
            'accounted_complete': len(saved) == len(rows),
            'fixture_sha256': digest(ROOT / f'data/prepared/{fixture}.jsonl'),
            'predictions_sha256': digest(path) if path.exists() else None,
        }
    metadata = ROOT / f'results/alternatives/models/{name}.json'
    if old:
        assert digest(metadata) == old['model_metadata_sha256'], f'Pinned model metadata changed: {name}'
    elif baseline:
        atomic_json(baseline_path, {'created': now(), 'model_metadata_sha256': digest(metadata), 'records': protected})
    for key in ['recorded', 'missing', 'answered', 'abstentions', 'native_rejections', 'runtime_failures', 'other_failures']:
        summary[key] = sum(v[key] for v in summary['fixtures'].values())
    summary['accounted_complete'] = summary['missing'] == 0
    summary['protected_predictions_unchanged'] = bool(old or baseline)
    return summary


def audit(models=None, baseline=False, repair=False):
    frozen = fixtures()
    names = models or [p.stem for p in sorted((ROOT / 'results/alternatives/models').glob('*.json'))]
    summary = {'created': now(), 'cases_per_model': 17576,
               'models': {name: audit_model(name, frozen, repair, baseline) for name in names}}
    summary['missing'] = sum(v['missing'] for v in summary['models'].values())
    summary['runtime_failures'] = sum(v['runtime_failures'] for v in summary['models'].values())
    atomic_json(OUT / 'audit.json', summary)
    if baseline and not (OUT/'initial-audit.json').exists():
        atomic_json(OUT/'initial-audit.json', summary)
    return summary


def disk_check(required=0):
    free = shutil.disk_usage(ROOT).free
    if free - required < FLOOR:
        raise RuntimeError(f'Disk floor: {free/2**30:.1f} GiB free; need {required/2**30:.1f} GiB plus 50 GiB reserve')
    return free


def restore_entries(name):
    meta = read_json(ROOT / f'results/alternatives/models/{name}.json')
    cleanup = read_json(ROOT / 'results/disk-audit/top3-cleanup-plan-2026-10-04.json')
    pins = [meta] + meta.get('bases', [])
    if name == 'clm-int8':
        pins.append(meta['encoder'])
    entries = []
    for pin in pins:
        root = Path(pin.get('path') or ROOT / '.cache/huggingface/hub' /
                    ('models--' + pin['repo'].replace('/', '--')) / 'snapshots' / pin['revision'])
        assert root.resolve().is_relative_to((ROOT / '.cache/huggingface/hub').resolve())
        assert root.name == pin['revision'] and len(pin['revision']) == 40
        for item in cleanup['weight_files']:
            path = Path(item['path'])
            if path.is_relative_to(root):
                entries.append({**item, 'hf_repo': pin['repo'], 'revision': pin['revision'],
                                'filename': path.relative_to(root).as_posix()})
    if not entries and name not in KEEP:
        raise ValueError(f'No pinned restoration manifest for {name}')
    return entries


def prepare(name):
    from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
    import huggingface_hub.file_download as fd
    fd.are_symlinks_supported = lambda cache_dir=None: False
    # A failed prior model must be reviewed before another checkpoint is restored.
    for previous in (OUT/'restored').glob('*.json'):
        if previous.stem.endswith('-prune') or previous.stem == name:
            continue
        record = read_json(previous)
        if record.get('model') in KEEP or record.get('pruned'):
            continue
        if any(Path(entry['path']).exists() for entry in record.get('files', [])):
            raise RuntimeError(f'Previous temporary model still restored: {record["model"]}; verify and prune it first')
    entries = restore_entries(name)
    needed = sum(e['bytes'] for e in entries if not Path(e['path']).exists())
    disk_check(needed + 512*2**20)
    state_path = OUT / 'restored' / f'{name}.json'
    state = read_json(state_path) if state_path.exists() else {'model': name, 'started': now(), 'files': []}
    known = {e['path']: e for e in state['files']}
    for entry in entries:
        path = Path(entry['path'])
        if not path.exists():
            disk_check(entry['bytes'])
            print('Restore', name, entry['filename'], flush=True)
            saved = Path(hf_hub_download(entry['hf_repo'], entry['filename'], revision=entry['revision']))
            assert saved.resolve() == path.resolve()
        assert path.stat().st_size == entry['bytes'], f'Unexpected weight size: {path}'
        checksum = digest(path)
        if entry['path'] in known:
            assert checksum == known[entry['path']]['sha256'], f'Restored weight hash changed: {path}'
        # Strong file hashes are retained independently of temporary weights.
        expected_sha = known.get(entry['path'], {}).get('publisher_sha256')
        if not expected_sha:
            remote = get_hf_file_metadata(hf_hub_url(entry['hf_repo'], entry['filename'], revision=entry['revision']))
            assert remote.commit_hash == entry['revision']
            assert remote.size == entry['bytes']
            expected_sha = remote.etag
        assert len(expected_sha) == 64 and checksum == expected_sha, f'Weight differs from pinned publisher hash: {path}'
        known[entry['path']] = {**entry, 'sha256': checksum, 'publisher_sha256': expected_sha}
        state['files'] = list(known.values())
        atomic_json(state_path, state)
    if name in ['clm-int8', 'nimble-9b', 'cygnet-12b-nf4', 'jev-omni-12b-nf4']:
        target = ROOT / '.cache/compat-bnb-0492'
        if not (target / 'bitsandbytes-0.49.2.dist-info/METADATA').exists():
            subprocess.run([str(ROOT/'.venv/Scripts/python.exe'), '-m', 'pip', 'install',
                            '--no-deps', '--no-cache-dir', '--only-binary=:all:', '--target', str(target),
                            'bitsandbytes==0.49.2'], cwd=ROOT, check=True)
        state['bitsandbytes'] = '0.49.2'
    if name == 'clm-int8':
        meta = read_json(ROOT/'results/alternatives/models/clm-int8.json')
        head = next(Path(e['path']) for e in entries if e['filename'] == 'CLM_v0.1-8B.pt')
        assert digest(head) == meta['heads']['sha256']
    state.update(finished=now(), free_gib=disk_check()/2**30)
    atomic_json(state_path, state)
    return state


@contextmanager
def controller_lock():
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'controller.lock').open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Another completion controller owns the GPU queue') from exc
        try:
            yield
        finally:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def assert_no_worker():
    import psutil
    markers = ['laya_bench.alternatives_run', 'laya_bench.tradeoff_expansion',
               'laya_bench.decision2_stress', 'laya_bench.alternatives_queue']
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.pid == os.getpid() or 'python' not in (process.info['name'] or '').lower():
            continue
        args = process.info['cmdline'] or []
        if any(marker in args for marker in markers):
            raise RuntimeError(f'Existing benchmark GPU worker/supervisor: PID {process.pid}')


def worker(command, attempt):
    """One child at a time; preserve stdout, command, exit status and active case."""
    assert_no_worker()
    disk_check()
    attempt.mkdir(parents=True, exist_ok=True)
    config = {'command': command, 'started': now(), 'status': 'running'}
    config['code_sha256'] = {name: digest(ROOT/'laya_bench'/name) for name in
                            ['alternatives_run.py', 'alternatives_adapters.py', 'leaders_adapters.py', 'decision2_adapter.py']}
    config['execution_policy'] = {'batch_size': 1,
                                  'torch_memory_fraction': .78, 'seed': 20261001,
                                  'tf32': False, 'threads': 8}
    if 'laya_bench.alternatives_run' in command:
        config['execution_policy']['max_new_cases'] = 256
    else:
        config['execution_policy'].update(timing_passes=3, warmups_per_task=8)
    atomic_json(attempt/'worker.json', config)
    with (attempt/'worker.log').open('ab') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        config['pid'] = process.pid
        atomic_json(attempt/'worker.json', config)
        # No terminal output pipe is held by the child. The controller is resumable.
        try:
            code = process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            process.terminate(); process.wait(timeout=30)
            code = 'timeout_3600s'
    config.update(exit_code=code, finished=now(), status='finished' if code == 0 else 'crashed')
    atomic_json(attempt/'worker.json', config)
    return code


def complete_fixture(name, fixture, rows, retry=False):
    expected = {r['id']: r for r in rows}
    dest = prediction_path(name, fixture)
    mode = 'runtime-retries' if retry else 'missing'
    progress_path = OUT/'work'/name/fixture/f'{mode}.json'
    progress = read_json(progress_path) if progress_path.exists() else {'failures': {}, 'blocked': [], 'attempts': []}
    focus = None
    while True:
        saved = load_records(dest, expected, repair_tail=True)
        pending = select_pending(rows, saved, retry_runtime=retry)
        pending = [r for r in pending if r['id'] not in progress['blocked']]
        if not pending:
            break
        if focus and any(r['id'] == focus for r in pending):
            selected = [expected[focus]]
        else:
            selected = pending[:256]
        ids = [r['id'] for r in selected]
        attempt = OUT/'workers'/name/fixture/mode/str(time.time_ns())
        atomic_json(attempt/'case-ids.json', ids)
        command = [python_for(name), '-X', 'utf8', '-m', 'laya_bench.alternatives_run', name,
                   '--fixture', fixture, '--batch-size', '1', '--max-new-cases', '256',
                   '--case-ids', str(attempt/'case-ids.json'), '--attempt-dir', str(attempt)]
        if retry:
            command.append('--retry-runtime-errors')
        recovery = OUT/'integration-recoveries'/name/f'{fixture}.json'
        if recovery.exists():
            command.extend(['--native-question-recovery', str(recovery)])
        print(name, fixture, mode, len(ids), 'cases;', len(pending), 'remaining', flush=True)
        code = worker(command, attempt)
        if retry:
            recovered = merge_retries(dest, attempt/'retry-predictions.jsonl', expected, attempt)
        else:
            after = load_records(dest, expected, repair_tail=True)
            recovered = list(set(after)-set(saved))
        unresolved = [key for key in ids if key not in recovered]
        # For retries an OOM is a completed attempt, but remains eligible up to 3.
        tried = load_records(attempt/'retry-predictions.jsonl', expected, repair_tail=True) if retry else {}
        active_path = attempt/'active-case.json'
        active = read_json(active_path)['ids'][0] if active_path.exists() else ids[0]
        failed = [key for key in unresolved if key in tried or key == active]
        if code != 0 and not failed and unresolved:
            failed = [unresolved[0]]
        if not recovered and not failed:
            failed = [unresolved[0]] if unresolved else []
        for key in failed:
            progress['failures'][key] = progress['failures'].get(key, 0) + 1
            if progress['failures'][key] >= 3 and key not in progress['blocked']:
                progress['blocked'].append(key)
        focus = next((key for key in failed if key not in progress['blocked']), None)
        progress['attempts'].append({'directory': str(attempt.relative_to(ROOT)), 'exit_code': code,
                                     'recovered': len(recovered), 'failed_case_ids': failed, 'finished': now()})
        atomic_json(progress_path, progress)
        if code != 0 and not active_path.exists() and len(progress['attempts']) >= 3:
            last = progress['attempts'][-3:]
            if all(v['exit_code'] != 0 and not v['recovered'] for v in last):
                raise RuntimeError(f'Three fresh workers failed before inference: {name}/{fixture}; see retained worker logs')
    return progress


def refresh_reports():
    OUT.mkdir(parents=True, exist_ok=True)
    for module in ['alternatives_report', 'jev_live_report']:
        with (OUT/f'{module}.log').open('ab') as stream:
            subprocess.run([str(ROOT/'.venv/Scripts/python.exe'), '-X', 'utf8', '-m',
                            'laya_bench.'+module], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def prune(name):
    """Remove only verified restored weights, using literal native PowerShell paths."""
    if name in KEEP:
        return
    verification = read_json(OUT/'verified'/f'{name}.json')
    if not verification['accounted_complete'] or not verification['protected_predictions_unchanged']:
        raise RuntimeError(f'Refusing to prune unverified model {name}')
    from .completion_timing import verify as verify_timing
    timing = verify_timing(name)
    assert timing['all_requested_groups_accounted']
    assert timing['summary_sha256'] == digest(OUT/'timing'/name/'summary.json')
    state = read_json(OUT/'restored'/f'{name}.json')
    assert all(e['sha256'] == e.get('publisher_sha256') for e in state['files']), 'Pinned publisher hashes must be verified before pruning'
    manifest = OUT/'restored'/f'{name}-prune.json'
    atomic_json(manifest, {'root': str(ROOT/'.cache/huggingface/hub'), 'files': state['files']})
    shell = shutil.which('pwsh.exe')
    if not shell:
        raise RuntimeError('PowerShell 7 is required for the verified literal-path cleanup script')
    subprocess.run([shell, '-NoProfile', '-File',
                    str(ROOT/'scripts/prune_completion_weights.ps1'), '-Manifest', str(manifest)],
                   cwd=ROOT, check=True)
    state.update(pruned=now(), free_gib=disk_check()/2**30)
    atomic_json(OUT/'restored'/f'{name}.json', state)


def run(models):
    with controller_lock():
        assert_no_worker()
        frozen = fixtures()
        # Freeze ALL pre-completion records before the first worker runs.
        audit(baseline=True, repair=True)
        state = {'started': now(), 'pid': os.getpid(), 'models': models, 'completed': [], 'failures': {}}
        if (OUT/'controller.json').exists():
            previous = OUT/'history'/f'controller-{time.time_ns()}.json'
            previous.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(OUT/'controller.json', previous)
        atomic_json(OUT/'controller.json', state)
        for name in models:
            state.update(current_model=name, status='preparing')
            atomic_json(OUT/'controller.json', state)
            try:
                current = audit_model(name, frozen)
                # Verified final models are skipped without restoring their weights.
                timed = OUT/'verified'/f'{name}-timing.json'
                restored = OUT/'restored'/f'{name}.json'
                exhausted = True
                for fixture, rows in frozen.items():
                    saved = load_records(prediction_path(name, fixture), {r['id']:r for r in rows})
                    retry_state = OUT/'work'/name/fixture/'runtime-retries.json'
                    blocked = set(read_json(retry_state)['blocked']) if retry_state.exists() else set()
                    if any(runtime_failure(row) and key not in blocked for key, row in saved.items()):
                        exhausted = False
                if current['accounted_complete'] and exhausted and timed.exists() and restored.exists() and read_json(restored).get('pruned'):
                    state['completed'].append(name)
                    continue
                prepare(name)
                for fixture, rows in frozen.items():
                    state.update(status='running', fixture=fixture)
                    atomic_json(OUT/'controller.json', state)
                    complete_fixture(name, fixture, rows)
                for fixture, rows in frozen.items():
                    complete_fixture(name, fixture, rows, retry=True)
                result = audit_model(name, frozen)
                atomic_json(OUT/'verified'/f'{name}.json', result)
                refresh_reports()
                if not result['accounted_complete']:
                    raise RuntimeError(f'{result["missing"]} cases still missing after bounded retries; retained diagnostics require review')
                if result['accounted_complete']:
                    state.update(status='measuring', fixture=None)
                    atomic_json(OUT/'controller.json', state)
                    from .completion_timing import measure
                    measure(name)
                    state['status']='pruning'
                    atomic_json(OUT/'controller.json', state)
                    prune(name)
                state['completed'].append(name)
                print('Verified', name, result['recorded'], 'records;', result['missing'], 'missing;',
                      result['runtime_failures'], 'runtime failures', flush=True)
            except Exception as exc:
                state['failures'][name] = {'type': type(exc).__name__, 'message': str(exc), 'time': now()}
                print('Needs attention:', name, type(exc).__name__, str(exc), flush=True)
                restored = OUT/'restored'/f'{name}.json'
                if restored.exists() and any(Path(e['path']).exists() for e in read_json(restored).get('files', [])):
                    state['deferred'] = models[models.index(name)+1:]
                    break
            atomic_json(OUT/'controller.json', state)
        state.update(status='needs_attention' if state['failures'] else 'finished', finished=now())
        atomic_json(OUT/'controller.json', state)
        audit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'prepare', 'run'])
    parser.add_argument('--models', nargs='+')
    args = parser.parse_args()
    if args.action == 'audit':
        value = audit(args.models)
        print(json.dumps({k:v for k,v in value.items() if k != 'models'}, indent=2))
        for name, v in value['models'].items():
            print(name, 'missing=', v['missing'], 'runtime=', v['runtime_failures'], 'native=', v['native_rejections'])
    elif args.action == 'prepare':
        if not args.models:
            parser.error('prepare requires --models with one model')
        if len(args.models) != 1:
            parser.error('Restore only one model at a time')
        with controller_lock():
            prepare(args.models[0])
    else:
        run(args.models or ORDER)


if __name__ == '__main__':
    main()
