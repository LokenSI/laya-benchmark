import json
import pytest
from laya_bench.common import ROOT,read_json,digest
from laya_bench.jev_api import Budget,decode,CAP_USD,PRICE_PER_TOKEN
from laya_bench.alternatives_jev import read_rows
from laya_bench.alternatives_adapters import typed_questions,decode_typed
from laya_bench.jev_live_data import normalize,leaves

def test_budget_reserves_unknown_attempts_and_accounts_actual_usage(tmp_path):
    b=Budget(tmp_path)
    try:
        p={'state':'text','questions':{'x':{'type':'noul'}}}
        attempt=b.reserve(p,'case')
        assert b.summary()['budget_accounted_usd']==pytest.approx(.042)
        b.settle(attempt,{'input_tokens':1000,'output_tokens':500})
        assert b.summary()['budget_accounted_usd']==pytest.approx(1000*PRICE_PER_TOKEN)
        unknown=b.reserve(p,'timeout')
        b.settle(b.reserve(p,'retry'),{'input_tokens':100,'output_tokens':10})
        assert b.summary()['unconfirmed_or_inflight_reserved_usd']==pytest.approx(.042)
        # Conservative reservation refuses the next request before it is sent.
        b.events.append({'event':'reserve','attempt':'exhausted','reserved_usd':CAP_USD})
        with pytest.raises(RuntimeError,match='exceed'):b.reserve(p,'never-sent')
    finally:b.close()
    b=Budget(tmp_path)
    try:assert b.summary()['unconfirmed_or_inflight_reserved_usd']==pytest.approx(.042)
    finally:b.close()

def test_decode_ordered_ties_and_rounding_band():
    q={'x':{'type':'choice','criteria':{'z':'last alphabetically','a':'first alphabetically'}}}
    a={'answers':{'x':{'type':'choice','probabilities':{'a':.5,'z':.5}}}}
    assert decode(q,a)['pred']['x']==['z']
    a['answers']['x']['probabilities']={'a':.49,'z':.5}
    r=decode(q,a)
    assert r['renormalized_questions']==['x']
    assert sum(r['probabilities']['x'].values())==pytest.approx(1)
    a['answers']['x']['probabilities']={'a':.4,'z':.4}
    with pytest.raises(AssertionError):decode(q,a)
    a['answers']['x']['probabilities']={'a':float('nan'),'z':1}
    with pytest.raises(AssertionError):decode(q,a)

def test_multi_label_expansion_matches_typed_local_adapters():
    q={'tags':{'type':'multilabel','instructions':'Select applicable tags','criteria':{'a':'A','b':'B'}}}
    native,mapping=typed_questions(q)
    response={'answers':{k:{'type':'noul','noul':v} for k,v in zip(native,[.5,.49])}}
    result=decode_typed(decode(native,response)['probabilities'],q,mapping)
    assert result['pred']=={'tags':['a']}

def test_fresh_fixture_is_frozen_and_disjoint_from_prior_evidence():
    protocol=read_json(ROOT/'results/alternatives/jev_fresh-protocol.json')
    path=ROOT/'data/prepared/jev_fresh.jsonl'
    assert digest(path)==protocol['fixture_sha256']
    fresh=read_rows(path)
    old_text=set()
    for filename,sha in protocol['exclusion_file_sha256'].items():
        p=ROOT/'data/prepared'/filename;assert digest(p)==sha
        prior=[c for s in read_json(p)['suites'].values() for c in s['cases']] if filename=='claims.json' else read_rows(p)
        for r in prior:old_text.update(normalize(t) for t in leaves(r.get('state',r.get('text',''))) if len(t.strip())>=20)
    for r in fresh:
        assert all(normalize(t) not in old_text for t in leaves(r['state']) if len(t.strip())>=20)
    dev={r['family'] for r in fresh if r['split']=='development'}
    test={r['family'] for r in fresh if r['split']=='test'}
    assert not dev&test
    assert len(fresh)==1264 and len({r['id'] for r in fresh})==1264

def test_api_payloads_never_contain_gold_or_source_metadata():
    from laya_bench.jev_live import evaluate
    class Client:
        def request(self,payload,case_id):
            assert set(payload)=={'model','state','questions'}
            assert payload['questions']=={'decision':{'type':'noul','instructions':'Is it faulty?'}}
            return {'response':{'model':'test','answers':{'decision':{'type':'noul','noul':.9}}},'seconds':.1}
    row={'id':'case','suite':'test','state':'Faulty item','questions':{'decision':{'type':'noul','instructions':'Is it faulty?'}},'gold':{'decision':['true']},'input_sha256':'hash','source_fixture':'test'}
    assert evaluate(row,Client(),'test')['correct']

def test_metrics_count_failures_and_threshold_never_uses_test_labels():
    from laya_bench.jev_live_report import aggregate,selective_policy
    row={'questions':{'q':{'type':'noul'}},'gold':{'q':['true']},'input_sha256':'x'}
    good={'input_sha256':'x','probabilities':{'q':{'false':.01,'true':.99}},'pred':{'q':['true']}}
    fail={'input_sha256':'x','error':'timeout','probabilities':{},'pred':{}}
    metrics,_=aggregate([row,row],[good,fail])
    assert metrics['accuracy']==.5 and metrics['coverage']==.5 and metrics['failures']==1
    wrong_row={**row,'gold':{'q':['false']}}
    passed=selective_policy([row]*100,[good]*100,[row]*50,[good]*50)
    failed=selective_policy([row]*100,[good]*100,[wrong_row]*50,[good]*50)
    assert passed['threshold']==failed['threshold']==.5
    assert passed['accepted_accuracy']==1 and failed['accepted_accuracy']==0
    assert failed['test_coverage']==1
    assert selective_policy([row]*4,[good]*4,[row]*50,[good]*50)['threshold'] is None

def test_native_abstention_is_not_replaced_by_probability_argmax():
    from laya_bench.jev_live_report import aggregate,selective_policy
    row={'questions':{'q':{'type':'noul'}},'gold':{'q':['true']},'input_sha256':'x'}
    abstain={'input_sha256':'x','probabilities':{'q':{'false':.01,'true':.99}},'pred':{'q':[]},'abstained':{'q':True}}
    metrics,_=aggregate([row],[abstain])
    assert metrics['accuracy']==0 and metrics['coverage']==0
    assert metrics['abstained_records']==1
    assert selective_policy([row]*100,[abstain]*100,[row],[abstain])['threshold'] is None
