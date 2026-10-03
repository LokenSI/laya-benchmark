import json
import math
import pytest
from laya_bench.common import ROOT,read_json,digest,fingerprint
from laya_bench.alternatives_run import validate_answer,expected_failure
from laya_bench.alternatives_adapters import typed_questions,decode_typed,answer_probabilities

@pytest.mark.parametrize('fixture',['alternatives','alternatives_claims','alternatives_typed','intern_claims','jev_verified'])
def test_frozen_input_contract(fixture):
    path=ROOT/f'data/prepared/{fixture}.jsonl'
    protocol=read_json(ROOT/('results/alternatives/protocol.json' if fixture=='alternatives' else f'results/alternatives/{fixture}-protocol.json'))
    assert digest(path)==protocol['fixture_sha256']
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==protocol['n']==len({r['id'] for r in rows})
    for row in rows:
        assert row['input_sha256']==fingerprint({'state':row['state'],'questions':row['questions']})
        assert set(row['questions'])==set(row['gold'])
        for key,q in row['questions'].items():
            assert set(q)<= {'type','instructions','criteria'}
            labels=set(map(str,range(len(q['criteria'])))) if q['type']=='score' else set(q.get('criteria') or ['true','false'])
            assert set(row['gold'][key])<=labels

def test_output_missing_options_and_nonfinite_are_adapter_failures():
    row={'questions':{'x':{'type':'choice','criteria':{'a':'A','b':'B'}}}}
    good={'pred':{'x':['a']},'probabilities':{'x':{'a':.6,'b':.4}}}
    validate_answer(row,good)
    for p in [{'a':1},{'a':math.nan,'b':0},{'a':.6,'b':.6}]:
        with pytest.raises(AssertionError):validate_answer(row,{'pred':{'x':['a']},'probabilities':{'x':p}})

def test_multilabel_boolean_decode_has_no_gold_top_k():
    qs={'topics':{'type':'multilabel','instructions':'Pick all topics','criteria':{'a':'A','b':'B','c':'C'}}}
    expanded,mapping=typed_questions(qs)
    assert len(expanded)==3
    probs={sub:{'true':p,'false':1-p} for sub,p in zip(expanded,[.8,.55,.1])}
    out=decode_typed(probs,qs,mapping)
    assert out['pred']['topics']==['a','b']
    validate_answer({'questions':qs},out)

def test_unknown_native_shape_aborts_as_an_integration_problem():
    with pytest.raises(TypeError):answer_probabilities({'x':{'unexpected':['yes']}})

def test_wrapped_oom_is_a_hardware_failure_but_other_runtime_errors_are_not():
    class FakeTorch:
        class OutOfMemoryError(RuntimeError):pass
    assert expected_failure(RuntimeError('TorchScript: CUDA out of memory'),FakeTorch)
    assert not expected_failure(RuntimeError('Invalid tensor shape'),FakeTorch)
