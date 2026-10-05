import json
import os

import pytest

from laya_bench import common


def test_atomic_json_preserves_serialization(tmp_path):
    report = tmp_path / 'report.json'
    previous = tmp_path / 'previous-format.json'
    value = {'Norwegian': 'blåbær', 'nested': [1, {'value': .25}]}
    previous.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    report.write_text('old report', encoding='utf-8')
    common.write_json(report, value)
    assert report.read_bytes() == previous.read_bytes()
    assert not list(tmp_path.glob('*.tmp'))


def test_failed_replacement_keeps_existing_report(tmp_path, monkeypatch):
    report = tmp_path / 'report.json'
    report.write_bytes(b'{"old": true}')
    def fail(source, destination):
        assert report.read_bytes() == b'{"old": true}'
        raise OSError('replacement denied')
    monkeypatch.setattr(common.os, 'replace', fail)
    with pytest.raises(OSError, match='replacement denied'):
        common.write_json(report, {'new': True})
    assert report.read_bytes() == b'{"old": true}'
    assert not list(tmp_path.glob('*.tmp'))


def test_transient_windows_sharing_error_is_retried(tmp_path, monkeypatch):
    report = tmp_path / 'report.json'
    report.write_bytes(b'{"old": true}')
    replace = os.replace
    calls = []
    def locked_once(source, destination):
        calls.append(1)
        if len(calls) == 1:
            error = OSError('sharing violation')
            error.winerror = 32
            raise error
        replace(source, destination)
    monkeypatch.setattr(common.os, 'replace', locked_once)
    monkeypatch.setattr(common.time, 'sleep', lambda duration: None)
    common.write_json(report, {'new': True})
    assert len(calls) == 2
    assert json.loads(report.read_text()) == {'new': True}


def test_invalid_json_keeps_existing_report(tmp_path):
    report = tmp_path / 'report.json'
    report.write_bytes(b'{"old": true}')
    with pytest.raises(ValueError):
        common.write_json(report, {'invalid': float('nan')})
    assert report.read_bytes() == b'{"old": true}'
    assert not list(tmp_path.glob('*.tmp'))
