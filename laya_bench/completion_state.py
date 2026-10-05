"""CPU-only, strict evidence handling for resumable completion workers."""
import hashlib
import json
import os
import time
from pathlib import Path


def runtime_failure(row):
    error = row.get('error', '')
    return error.startswith('OutOfMemoryError') or (
        error.startswith('RuntimeError') and
        ('CUDA out of memory' in error or error == 'RuntimeError: bad allocation'))


def record_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_records(path, expected=None, repair_tail=False):
    """Only an incomplete final record can be archived/repaired automatically."""
    path = Path(path)
    if not path.exists():
        return {}
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    records = {}
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b'\n') or not repair_tail:
                raise ValueError(f'Malformed evidence record: {path}:{index + 1}')
            archive = path.with_name(path.name + f'.interrupted-{time.time_ns()}')
            archive.write_bytes(payload)
            temporary = path.with_name(path.name + '.repair.tmp')
            temporary.write_bytes(b''.join(lines[:index]))
            temporary.replace(path)
            break
        key = row['id']
        if key in records:
            raise ValueError(f'Duplicate prediction ID: {path}: {key}')
        if expected is not None:
            if key not in expected or row.get('input_sha256') != expected[key]['input_sha256']:
                raise ValueError(f'Prediction input mismatch: {path}: {key}')
            if row.get('gold') != expected[key]['gold']:
                raise ValueError(f'Prediction answer-key mismatch: {path}: {key}')
        records[key] = row
    if repair_tail and path.exists():
        with path.open('ab') as stream:
            if records and path.stat().st_size and not path.read_bytes().endswith(b'\n'):
                stream.write(b'\n')
    return records


def select_pending(rows, saved, max_new_cases=None, retry_runtime=False, case_ids=None):
    if max_new_cases is not None and max_new_cases < 1:
        raise ValueError('max_new_cases must be positive')
    allowed = set(case_ids) if case_ids is not None else None
    expected = {row['id'] for row in rows}
    if allowed is not None and not allowed <= expected:
        raise ValueError('Requested case IDs are not in the selected fixture')
    pending = [row for row in rows if (allowed is None or row['id'] in allowed) and
               (runtime_failure(saved.get(row['id'], {})) if retry_runtime
                else row['id'] not in saved)]
    return pending[:max_new_cases] if max_new_cases is not None else pending


def merge_retries(destination, retry_path, expected, archive_directory):
    """Replace only old infrastructure failures with successful, validated results."""
    from .alternatives_run import validate_answer
    destination, retry_path = Path(destination), Path(retry_path)
    current = load_records(destination, expected)
    retried = load_records(retry_path, expected, repair_tail=True)
    replacements = {}
    for key, row in retried.items():
        if key not in current or not runtime_failure(current[key]):
            raise ValueError(f'Retry tried to replace a non-runtime result: {key}')
        if not row.get('error'):
            validate_answer(expected[key], row)
            gold = expected[key]['gold']
            correct = set(row['pred']) == set(gold) and all(set(row['pred'][k]) == set(g) for k, g in gold.items())
            if row.get('correct') != correct:
                raise ValueError(f'Retry correctness mismatch: {key}')
            replacements[key] = row
    if not replacements:
        return []
    archive_directory = Path(archive_directory)
    archive_directory.mkdir(parents=True, exist_ok=True)
    archive = archive_directory / f'before-runtime-retry-{time.time_ns()}.jsonl'
    archive.write_bytes(destination.read_bytes())
    temporary = destination.with_name(destination.name + '.retry.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for key, row in current.items():
            stream.write(json.dumps(replacements.get(key, row), ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return list(replacements)


def no_progress_count(previous_ids, current_ids, previous_count):
    return 0 if set(current_ids) - set(previous_ids) else previous_count + 1
