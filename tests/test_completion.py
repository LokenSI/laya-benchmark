import json
import subprocess
import sys
from pathlib import Path

import pytest

from laya_bench import completion
from laya_bench.completion_state import (atomic_json, load_records, merge_retries,
                                        record_hash, runtime_failure, select_pending)


def case(key):
    return {'id': key, 'input_sha256': 'hash-' + key, 'gold': {'x': ['a']},
            'questions': {'x': {'type': 'choice', 'criteria': {'a': 'A', 'b': 'B'}}}}


def result(key, error=None):
    value = {**case(key), 'pred': {'x': ['a']}, 'probabilities': {'x': {'a': .8, 'b': .2}}, 'correct': True}
    if error:
        value.update(error=error, pred={}, probabilities={}, correct=False)
    return value


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def test_limit_applies_after_excluding_completed_and_selected_ids():
    rows = [case(str(i)) for i in range(8)]
    saved = {str(i): result(str(i)) for i in range(4)}
    assert [r['id'] for r in select_pending(rows, saved, 2)] == ['4', '5']
    assert [r['id'] for r in select_pending(rows, saved, 2, case_ids=['2', '6', '7'])] == ['6', '7']
    with pytest.raises(ValueError):
        select_pending(rows, saved, case_ids=['unknown'])


@pytest.mark.parametrize('corruption', ['duplicate', 'input', 'gold', 'unknown'])
def test_rejects_invalid_evidence(tmp_path, corruption):
    row = result('a')
    rows = [row]
    if corruption == 'duplicate': rows.append(row.copy())
    elif corruption == 'input': row['input_sha256'] = 'wrong'
    elif corruption == 'gold': row['gold'] = {'x': ['b']}
    else: row['id'] = 'unknown'
    path = tmp_path/'rows.jsonl'
    write_rows(path, rows)
    with pytest.raises(ValueError):
        load_records(path, {'a': case('a')}, repair_tail=True)


def test_interrupted_tail_is_archived_and_completed_prefix_preserved(tmp_path):
    path = tmp_path/'rows.jsonl'
    prefix = json.dumps(result('a')).encode() + b'\n'
    original = prefix + b'{"id": "b'
    path.write_bytes(original)
    with pytest.raises(ValueError): load_records(path)
    assert list(load_records(path, repair_tail=True)) == ['a']
    assert path.read_bytes() == prefix
    assert next(tmp_path.glob('*.interrupted-*')).read_bytes() == original


def test_valid_unterminated_record_is_not_lost_and_bad_interior_is_fatal(tmp_path):
    path = tmp_path/'rows.jsonl'
    path.write_text(json.dumps(result('a')), encoding='utf-8')
    assert list(load_records(path, repair_tail=True)) == ['a']
    assert path.read_bytes().endswith(b'\n')
    path.write_bytes(b'broken\n' + path.read_bytes())
    with pytest.raises(ValueError): load_records(path, repair_tail=True)


def test_retry_only_infrastructure_preserves_incorrect_abstained_and_rejected(tmp_path):
    rows = [case(str(i)) for i in range(5)]
    saved = {str(i): result(str(i)) for i in range(5)}
    saved['0'] = result('0', 'OutOfMemoryError: CUDA out of memory')
    saved['1'] = result('1', 'RuntimeError: TorchScript CUDA out of memory')
    saved['2'] = result('2', 'ValueError: at most 26 options')
    saved['3'].update(pred={'x': ['b']}, correct=False)
    saved['4'].update(pred={'x': []}, correct=False, abstained={'x': True})
    assert [r['id'] for r in select_pending(rows, saved, retry_runtime=True)] == ['0', '1']
    original = tmp_path/'predictions.jsonl'
    retry = tmp_path/'retry.jsonl'
    write_rows(original, list(saved.values()))
    before = original.read_bytes()
    write_rows(retry, [result('0'), saved['1']])
    assert merge_retries(original, retry, {r['id']:r for r in rows}, tmp_path/'archive') == ['0']
    after = load_records(original)
    for key in ['1', '2', '3', '4']:
        assert record_hash(after[key]) == record_hash(saved[key])
    assert next((tmp_path/'archive').glob('*.jsonl')).read_bytes() == before
    write_rows(retry, [result('3')])
    with pytest.raises(ValueError): merge_retries(original, retry, {r['id']:r for r in rows}, tmp_path/'archive')


def test_crash_after_partial_progress_retries_case_then_continues(tmp_path, monkeypatch):
    rows = [case(str(i)) for i in range(5)]
    dest = tmp_path/'predictions.jsonl'
    monkeypatch.setattr(completion, 'OUT', tmp_path/'completion')
    monkeypatch.setattr(completion, 'ROOT', tmp_path)
    monkeypatch.setattr(completion, 'prediction_path', lambda *args: dest)
    invocations = []
    def fake_worker(command, attempt):
        ids = json.loads((attempt/'case-ids.json').read_text())
        invocations.append(ids)
        existing = load_records(dest)
        for key in ids:
            atomic_json(attempt/'active-case.json', {'ids': [key]})
            if key == '2':
                write_rows(dest, list(existing.values()))
                return 99
            existing[key] = result(key)
        write_rows(dest, list(existing.values()))
        return 0
    monkeypatch.setattr(completion, 'worker', fake_worker)
    state = completion.complete_fixture('fake', 'test', rows)
    assert state['blocked'] == ['2']
    assert state['failures']['2'] == 3
    assert invocations == [['0','1','2','3','4'], ['2'], ['2'], ['3','4']]
    assert set(load_records(dest)) == {'0','1','3','4'}
    completion.complete_fixture('fake', 'test', rows)
    assert len(invocations) == 4


def test_recorded_oom_retried_three_times_and_original_retained(tmp_path, monkeypatch):
    dest = tmp_path/'predictions.jsonl'
    original = result('a', 'RuntimeError: bad allocation')
    write_rows(dest, [original])
    monkeypatch.setattr(completion, 'OUT', tmp_path/'completion')
    monkeypatch.setattr(completion, 'ROOT', tmp_path)
    monkeypatch.setattr(completion, 'prediction_path', lambda *args: dest)
    def fake_worker(command, attempt):
        write_rows(attempt/'retry-predictions.jsonl', [original])
        return 0
    monkeypatch.setattr(completion, 'worker', fake_worker)
    state = completion.complete_fixture('fake', 'test', [case('a')], retry=True)
    assert len(state['attempts']) == 3
    assert state['blocked'] == ['a']
    assert load_records(dest)['a'] == original


def test_completion_reports_do_not_load_optional_pandas_binaries():
    code = ('import sys; import laya_bench.completion_report; import laya_bench.jev_live_report; '
            'assert "pandas" not in sys.modules; assert "sklearn" not in sys.modules')
    subprocess.run([sys.executable, '-c', code], check=True)


def test_explicit_native_recovery_preserves_native_inputs_and_decoding():
    from laya_bench.alternatives_adapters import JevK5
    from laya_bench.alternatives_run import evaluate_batch
    question = {'type': 'choice', 'instructions': 'Choose exactly one.',
                'criteria': {str(i): 'Option '+str(i) for i in range(18)}}
    row = {'id': 'case', 'state': {'evidence': 'unaltered'}, 'questions': {
        'choice': question, 'tags': {'type': 'multilabel', 'instructions': 'Tag',
                                   'criteria': {'x': 'X', 'y': 'Y'}}}}
    calls = []
    class Native:
        def probabilities(self, state, q):
            calls.append((state, q))
            if q is question:
                return {str(i): float(i == 17) for i in range(18)}, 123
            return {'true': .8 if 'label x?' in q['instructions'] else .2,
                    'false': .2 if 'label x?' in q['instructions'] else .8}, 100
    adapter = JevK5.__new__(JevK5)
    adapter.model = Native()
    def failing_batch(rows):
        raise AssertionError('Native/batched probability mismatch')
    adapter.batch = failing_batch
    # An unlisted case still fails the original check; there is no automatic bypass.
    with pytest.raises(AssertionError, match='probability mismatch'):
        evaluate_batch(adapter, [row], set())
    answer = evaluate_batch(adapter, [row], {'case'})[0]
    assert answer['pred'] == {'choice': ['17'], 'tags': ['x']}
    assert answer['inference_path'] == 'publisher-native-serial'
    assert len(calls) == 3 and all(state is row['state'] for state, _ in calls)
    assert calls[0][1] is question  # All 18 options go to native knockout unchanged.
    with pytest.raises(AssertionError, match='batch size one'):
        evaluate_batch(adapter, [row, row], {'case'})


def test_native_recovery_manifest_pins_model_fixture_and_input(tmp_path):
    from laya_bench.alternatives_run import native_recovery_ids
    manifest = tmp_path/'recovery.json'
    value = {'model': 'plumb-4b', 'fixture': 'test',
             'method': 'publisher-native-serial', 'reason': 'Retained parity failure',
             'cases': {'a': 'hash-a'}}
    atomic_json(manifest, value)
    assert native_recovery_ids(manifest, 'plumb-4b', 'test', {'a': case('a')}) == {'a'}
    for name, fixture in [('jevk5', 'test'), ('plumb-4b', 'different')]:
        with pytest.raises(AssertionError):
            native_recovery_ids(manifest, name, fixture, {'a': case('a')})
    value['cases']['a'] = 'changed-input'
    atomic_json(manifest, value)
    with pytest.raises(AssertionError):
        native_recovery_ids(manifest, 'plumb-4b', 'test', {'a': case('a')})


def test_incomplete_model_is_not_marked_complete_or_replaced(tmp_path, monkeypatch):
    from contextlib import nullcontext
    out = tmp_path/'completion'
    weight = tmp_path/'temporary.safetensors'
    weight.write_bytes(b'fixture')
    atomic_json(out/'restored/first.json', {'files': [{'path': str(weight)}]})
    monkeypatch.setattr(completion, 'OUT', out)
    monkeypatch.setattr(completion, 'controller_lock', nullcontext)
    monkeypatch.setattr(completion, 'assert_no_worker', lambda: None)
    monkeypatch.setattr(completion, 'fixtures', lambda: {'test': [case('a')]})
    monkeypatch.setattr(completion, 'prediction_path', lambda *args: tmp_path/'absent.jsonl')
    monkeypatch.setattr(completion, 'audit', lambda **kwargs: {})
    monkeypatch.setattr(completion, 'audit_model', lambda *args: {
        'accounted_complete': False, 'missing': 1})
    prepared = []
    monkeypatch.setattr(completion, 'prepare', prepared.append)
    monkeypatch.setattr(completion, 'complete_fixture', lambda *args, **kwargs: None)
    monkeypatch.setattr(completion, 'refresh_reports', lambda: None)
    completion.run(['first', 'second'])
    state = json.loads((out/'controller.json').read_text())
    assert prepared == ['first'] and state['completed'] == []
    assert state['status'] == 'needs_attention' and state['deferred'] == ['second']


def test_unavailable_timing_requires_three_hash_verified_failures(tmp_path, monkeypatch):
    from laya_bench import completion_timing as timing
    from laya_bench.common import digest
    out = tmp_path/'results/completion'
    monkeypatch.setattr(completion, 'OUT', out)
    monkeypatch.setattr(timing, 'ROOT', tmp_path)
    model = tmp_path/'results/alternatives/models/fake.json'
    atomic_json(model, {'revision': 'pinned'})
    attempts = []
    for index in range(3):
        directory = out/'workers/fake/timing'/str(index)
        atomic_json(directory/'worker.json', {'exit_code': 1})
        (directory/'worker.log').write_text('AssertionError: native parity mismatch')
        attempts.append({'directory': str(directory.relative_to(tmp_path)),
                         'worker_log_sha256': digest(directory/'worker.log')})
    path = out/'timing-failures/fake.json'
    value = {'model': 'fake', 'fixture_sha256': 'frozen',
             'model_metadata_sha256': digest(model), 'groups': {'task': {
                 'reason': 'Original adapter failed three times',
                 'failure_signature': 'native parity mismatch', 'attempts': attempts}}}
    atomic_json(path, value)
    protocol = {'fixture_sha256': 'frozen', 'models': {'fake': ['task']}}
    assert 'task' in timing.verified_unavailable('fake', protocol)
    value['groups']['task']['attempts'] = attempts[:2]
    atomic_json(path, value)
    with pytest.raises(AssertionError): timing.verified_unavailable('fake', protocol)
    value['groups']['task']['attempts'] = attempts
    atomic_json(path, value)
    (out/'workers/fake/timing/0/worker.log').write_text('Modified evidence')
    with pytest.raises(AssertionError): timing.verified_unavailable('fake', protocol)


def test_unavailable_timing_is_disclosed_without_fabricated_measurements(tmp_path, monkeypatch):
    from laya_bench import completion_timing as timing
    from laya_bench.common import digest
    out = tmp_path/'completion'
    data = out/'timing'
    monkeypatch.setattr(completion, 'OUT', out)
    monkeypatch.setattr(timing, 'OUT', data)
    rows = [{'id': 'a', 'input_sha256': 'a-hash', 'timing_suite': 'done'},
            {'id': 'b', 'input_sha256': 'b-hash', 'timing_suite': 'unavailable'}]
    write_rows(data/'fixture.jsonl', rows)
    atomic_json(data/'protocol.json', {'fixture_sha256': digest(data/'fixture.jsonl'),
                                      'models': {'fake': ['done', 'unavailable']}})
    raw = [{'id': 'a', 'input_sha256': 'a-hash', 'pass': p} for p in [1,2,3]]
    write_rows(data/'fake/done.jsonl', raw)
    original_bytes = (data/'fake/done.jsonl').read_bytes()
    block = {'complete': True, 'calls': 3, 'predictions_sha256': digest(data/'fake/done.jsonl')}
    atomic_json(data/'fake/summary.json', {'groups': {'done': block}})
    monkeypatch.setattr(timing, 'verified_unavailable', lambda *args: {})
    with pytest.raises(AssertionError, match='Missing timing block'):
        timing.verify('fake')
    evidence = {'unavailable': {'case_input_hashes': {'b': 'b-hash'}, 'reason': 'Retained repeated failures'}}
    monkeypatch.setattr(timing, 'verified_unavailable', lambda *args: evidence)
    value = timing.verify('fake')
    assert value['status'] == 'partial_unavailable' and value['groups'] == ['done']
    assert value['unavailable_groups'] == {'unavailable': 'Retained repeated failures'}
    state = json.loads((data/'fake/summary.json').read_text())
    assert state['groups'] == {'done': block}
    assert not (data/'fake/unavailable.jsonl').exists()
    assert (data/'fake/done.jsonl').read_bytes() == original_bytes
    evidence['unavailable']['case_input_hashes']['b'] = 'changed'
    with pytest.raises(AssertionError): timing.verify('fake')
