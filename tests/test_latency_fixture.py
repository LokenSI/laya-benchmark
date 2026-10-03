import json
from collections import Counter
from laya_bench.common import ROOT,read_json,digest
from laya_bench.alternatives_jev import read_rows

def test_latency_uses_exact_short_test_inputs_with_balanced_task_counts():
    path=ROOT/'data/prepared/latency.jsonl'
    protocol=read_json(ROOT/'results/latency/protocol.json')
    assert digest(path)==protocol['fixture_sha256']
    source={r['id']:r for r in read_rows(ROOT/'data/prepared/jev_fresh.jsonl')}
    rows=read_rows(path)
    assert len(rows)==len({r['id'] for r in rows})==80
    assert set(Counter(r['suite'] for r in rows).values())=={20}
    for r in rows:
        assert r==source[r['id']] and r['split']=='test'
        assert len(json.dumps(r['state'],ensure_ascii=False))<=512
