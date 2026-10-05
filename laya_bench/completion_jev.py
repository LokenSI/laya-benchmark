"""Retry only the two saved transport failures against the frozen Jev version."""
import json
import os
import time

from .common import ROOT, read_json, digest
from .completion_state import atomic_json, load_records, record_hash


def run():
    from .jev_api import Budget, Client, append
    from .jev_live import evaluate
    from .alternatives_run import validate_answer
    root = ROOT/'results/jev_live'
    protocol = read_json(root/'protocol.json')
    pin = read_json(root/'model-pin.json')['model']
    assert pin == protocol['model']['model'] == 'jev-1.13.0'
    fixture = ROOT/'data/prepared/jev_live.jsonl'
    assert digest(fixture) == protocol['fixture_sha256']
    rows = {r['id']: r for r in map(json.loads, fixture.read_text(encoding='utf-8').splitlines())}
    dest = root/'predictions.jsonl'
    saved = load_records(dest, rows)
    assert len(saved) == len(rows) == 17576
    assert all(r['model'] == pin for r in saved.values())
    targets = [key for key, row in saved.items() if row.get('error') == 'HTTP 520']
    assert len(targets) <= 2, 'Unexpected API failure set; inspect before retrying'
    if not targets:
        print('Jev has no remaining recorded HTTP 520 failures', flush=True)
        return
    protected = {key: record_hash(row) for key, row in saved.items() if key not in targets}
    directory = ROOT/'results/completion/jev'/str(time.time_ns())
    directory.mkdir(parents=True)
    archive = directory/'original-predictions.jsonl'
    archive.write_bytes(dest.read_bytes())
    replacements = {}
    budget = Budget()
    try:
        client = Client(budget)
        for key in targets:
            for attempt in range(3):
                result = evaluate(rows[key], client, pin)
                append(directory/'retry-attempts.jsonl', {'retry_number': attempt+1, **result})
                if not result.get('error'):
                    validate_answer(rows[key], result)
                    assert result['request_ordered_sha256'] == saved[key]['request_ordered_sha256']
                    replacements[key] = result
                    break
                if result.get('error') in ['HTTP 401', 'HTTP 402', 'HTTP 403']:
                    raise RuntimeError('Jev account rejected the retry: '+result['error'])
        # Detect a concurrent edit before replacing any record.
        assert digest(dest) == digest(archive)
        temporary = dest.with_suffix('.retry.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            for key, row in saved.items():
                stream.write(json.dumps(replacements.get(key, row), ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(dest)
        final = load_records(dest, rows)
        assert all(record_hash(final[key]) == sha for key, sha in protected.items())
        summary = {'model': pin, 'targeted_errors': len(targets), 'recovered': len(replacements),
                   'remaining_errors': sum(bool(r.get('error')) for r in final.values()),
                   'protected_predictions_unchanged': True, 'original_sha256': digest(archive),
                   'predictions_sha256': digest(dest), 'original_attempts_preserved': True,
                   'fixture_sha256': digest(fixture), 'budget': budget.summary()}
        atomic_json(directory/'summary.json', summary)
        atomic_json(ROOT/'results/completion/jev-verification.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        budget.close()


if __name__ == '__main__':
    run()
