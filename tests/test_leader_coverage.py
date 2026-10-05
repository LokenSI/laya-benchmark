from laya_bench.completion import FIXTURES
from laya_bench.leaders_report import shared_coverage


def test_shared_coverage_excludes_separate_publisher_claim_fixture():
    groups = {name: {'complete': True, 'attempted': n, 'errors': 0}
              for name, n in zip(FIXTURES, [12341, 3271, 400, 300, 1264])}
    groups['intern_claims'] = {'complete': False, 'attempted': 999, 'errors': 25}
    report = {'models': {'model': {'fixtures': groups}}}
    assert shared_coverage(report, 'model') == {'complete': True, 'recorded': 17576, 'saved_failures': 0}
    groups['jev_fresh']['complete'] = False
    groups['jev_fresh']['attempted'] -= 1
    assert shared_coverage(report, 'model')['complete'] is False
    assert shared_coverage(report, 'model')['recorded'] == 17575
