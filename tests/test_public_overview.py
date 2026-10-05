"""The public task ranking must retain failures, ties and missing measurements."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.public_overview import JEV, assemble, leaders, metrics, render

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def evidence():
    return (load('results/jev_live/comparison.json'),
            load('results/linkedin/completion-tradeoffs/measurements.json'),
            load('results/completion/summary.json'))


def test_all_models_are_present_and_leaders_are_not_the_previous_shortlist(evidence):
    data = assemble(*evidence)
    expected = set(evidence[0]['models'])
    assert len(expected) == 32
    assert all({r['id'] for r in task['rows']} == expected for task in data['tasks'])
    task = next(t for t in data['tasks'] if t['id'] == 'massive_nb')
    assert [r['id'] for r in leaders(task['rows'], 'accuracy', highest=True)] == ['imajev-4b']
    task = next(t for t in data['tasks'] if t['id'] == 'fresh/spam')
    assert {r['id'] for r in leaders(task['rows'], 'accuracy', highest=True)} == {'laya', 'laya-multilingual'}


def test_accuracy_retains_unanswered_cases_and_withholds_incomplete_groups():
    block = {'complete': True, 'n': 10, 'expected': 10, 'attempted': 10, 'correct': 6, 'valid': 8}
    model = {'suites': {'task': block}}
    assert metrics(model, 'task')['accuracy'] == .6
    block['attempted'] = 9
    assert metrics(model, 'task') is None


def test_resource_ranking_excludes_hosted_and_missing_values():
    rows = [{'id': JEV, 'vram_gib': 0}, {'id': 'unmeasured', 'vram_gib': None},
            {'id': 'a', 'vram_gib': 2}, {'id': 'b', 'vram_gib': 2}]
    assert [r['id'] for r in leaders(rows, 'vram_gib')] == ['a', 'b']


def test_low_timing_coverage_is_withheld_but_accuracy_is_retained(evidence):
    comparison, measured, completion = evidence
    measured = deepcopy(measured)
    row = next(r for r in measured['rows'] if r['model'] == 'decider-4b' and r['suite'] == 'massive_nb')
    row['answered'] = row['calls'] // 2
    data = assemble(comparison, measured, completion)
    task = next(t for t in data['tasks'] if t['id'] == 'massive_nb')
    row = next(r for r in task['rows'] if r['id'] == 'decider-4b')
    assert row['accuracy'] > .8 and row['median_ms'] is None and row['vram_gib'] is None
    assert 'Withheld' in row['timing_note']


def test_mismatched_timing_accuracy_is_rejected(evidence):
    comparison, measured, completion = evidence
    measured = deepcopy(measured)
    measured['rows'][0]['correct'] -= 1
    with pytest.raises(AssertionError, match='Timing/accuracy mismatch'):
        assemble(comparison, measured, completion)


def test_missing_timing_never_uses_recovery_run_latency(evidence):
    data = assemble(*evidence)
    task = next(t for t in data['tasks'] if t['id'] == 'kev_claim/devtools-v1')
    row = next(r for r in task['rows'] if r['id'] == 'jevk5')
    assert row['accuracy'] is not None and row['median_ms'] is None and row['vram_gib'] is None


def test_static_page_exposes_all_models_without_javascript(evidence):
    page = render(*evidence, baseline=.891)
    table = page.split('<table id="comparison">', 1)[1].split('</table>', 1)[0]
    assert table.count('<tr>') == 33
    assert 'Imajev 4B' in table and 'Jev 1.13.0 API' in table
    assert 'Earlier focused comparisons' not in page
    assert 'focused-comparison' not in page
    assert '@@' not in page
