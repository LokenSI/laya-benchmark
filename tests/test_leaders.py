import math
import pytest
from laya_bench.common import ROOT
from laya_bench.leaders_adapters import Winnow, CygnetNF4, native_module
from laya_bench.alternatives_run import validate_answer


def test_winnow_wire_contract_preserves_native_tie_choice_and_excludes_gold():
    class Reply:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {'answers': {'q': {'choice': 'b', 'probabilities': {'a': .5, 'b': .5}}}}
    class Session:
        def post(self, url, json, timeout):
            self.sent = json
            return Reply()
    adapter = Winnow.__new__(Winnow)
    adapter.url = 'http://127.0.0.1:12345'
    adapter.session = Session()
    row = {'state': 'private task input', 'questions': {'q': {'type': 'choice', 'instructions': 'Choose',
           'criteria': {'a': 'first', 'b': 'second'}}}, 'gold': {'q': ['a']}}
    result = adapter.batch([row])[0]
    validate_answer(row, result)
    assert result['pred']['q'] == ['b']
    assert set(adapter.session.sent) == {'state', 'questions', 'winnow'}
    assert adapter.session.sent['winnow']['reuse_prefix'] is False


def test_winnow_input_rejection_is_not_an_invented_prediction():
    class Reply:
        status_code = 422
        text = 'too many options'
    class Session:
        def post(self, *a, **kw):
            return Reply()
    adapter = Winnow.__new__(Winnow)
    adapter.url = 'http://127.0.0.1:12345'
    adapter.session = Session()
    with pytest.raises(ValueError, match='too many options'):
        adapter.batch([{'state': 'x', 'questions': {'q': {'type': 'noul', 'instructions': 'x'}}}])


def test_cygnet_publisher_readout_sums_duplicate_tokens_then_calibrates():
    source = ROOT / '.cache/leader_research/blockbrain-ai--cygnet-recipe/shim/cygnet_shim.py'
    if not source.exists():
        pytest.skip('Pinned Cygnet source not prepared')
    module = native_module('cygnet_contract_test', source)
    module.TEMPERATURE = 3.4
    module.call_vllm = lambda *_: {'choices': [{'logprobs': {'content': [{'top_logprobs': [
        {'token': 'A', 'logprob': math.log(.3)}, {'token': 'A', 'logprob': math.log(.3)},
        {'token': 'B', 'logprob': math.log(.4)}]}]}}]}
    answer, _, error = module.answer_for('x', {'type': 'choice', 'instructions': 'x', 'criteria': {'a': 'A', 'b': 'B'}})
    expected = .6 ** (1 / 3.4) / (.6 ** (1 / 3.4) + .4 ** (1 / 3.4))
    assert error is None
    assert answer['probabilities']['a'] == pytest.approx(expected)
    assert answer['choice'] == 'a'


def test_cygnet_native_option_limit_is_retained():
    source = ROOT / '.cache/leader_research/blockbrain-ai--cygnet-recipe/shim/cygnet_shim.py'
    if not source.exists():
        pytest.skip('Pinned Cygnet source not prepared')
    adapter = CygnetNF4.__new__(CygnetNF4)
    adapter.native = native_module('cygnet_limit_test', source)
    with pytest.raises(ValueError, match='26'):
        adapter.batch([{'state': 'x', 'questions': {'q': {'type': 'choice', 'instructions': 'x',
                       'criteria': {str(i): str(i) for i in range(27)}}}}])
