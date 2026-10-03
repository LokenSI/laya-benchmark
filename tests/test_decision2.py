import pytest
import hashlib
import json
from laya_bench.decision2_prepare import portable_identity
from laya_bench.decision2_adapter import Decision2
from laya_bench.alternatives_run import validate_answer, expected_failure

def test_windows_path_portability_still_rejects_changed_or_extra_files_and_wrong_identity():
    canonical = lambda value: json.dumps(value, sort_keys=True)
    files = {'backbone/model.safetensors': 'a' * 64}
    identity = {'fingerprint_files': files, 'model_sha256': hashlib.sha256(canonical(files).encode()).hexdigest()}
    result = {'files_sha256': {'backbone\\model.safetensors': 'a' * 64}}
    assert portable_identity(result, identity, canonical)['files_sha256'] == files
    for invalid in [{'backbone\\model.safetensors': 'b' * 64},
                    {**result['files_sha256'], 'extra': 'a' * 64},
                    {**result['files_sha256'], **files}]:
        with pytest.raises(ValueError, match='file roster'):
            portable_identity({'files_sha256': invalid}, identity, canonical)
    with pytest.raises(ValueError, match='Portable model identity'):
        portable_identity(result, {**identity, 'model_sha256': '0' * 64}, canonical)

def test_native_allocation_failure_is_retained_but_programming_errors_still_stop():
    class Torch:
        OutOfMemoryError = MemoryError
    assert expected_failure(RuntimeError('bad allocation'), Torch)
    assert not expected_failure(RuntimeError('wrong tensor dimensions'), Torch)

def test_native_choices_binary_ties_score_and_gold_blind_payload():
    class Model:
        def system_one(self, **payload):
            self.payload = payload
            return {'answers': {'route': {'choice': 'b', 'probabilities': {'a': .5, 'b': .5}},
                                'yes': {'noul': .5},
                                'rating': {'score': .6, 'probabilities': {'0': .6, '1': .2, '2': .2}}},
                    'usage': {'input_tokens': 10}}
    adapter = Decision2.__new__(Decision2)
    adapter.model = Model()
    row = {'state': 'input', 'questions': {
        'route': {'type': 'choice', 'instructions': 'Choose', 'criteria': {'a': None, 'b': 'B'}},
        'yes': {'type': 'noul', 'instructions': 'Yes?'},
        'rating': {'type': 'score', 'instructions': 'Rate', 'criteria': ['low', 'medium', 'high']}},
        'gold': {'route': ['a'], 'yes': ['false'], 'rating': ['1']}}
    result = adapter.batch([row])[0]
    validate_answer(row, result)
    assert result['pred'] == {'route': ['b'], 'yes': ['true'], 'rating': ['0']}
    assert adapter.model.payload == {'state': row['state'], 'questions': row['questions']}
    assert result['native_answers']['rating']['score'] == .6

def test_multilabel_uses_independent_native_binary_probabilities():
    class Model:
        def system_one(self, **payload):
            return {'answers': {'tags__label_0': {'noul': .5}, 'tags__label_1': {'noul': .2}}}
    adapter = Decision2.__new__(Decision2)
    adapter.model = Model()
    row = {'state': 'x', 'questions': {'tags': {'type': 'multilabel', 'instructions': 'Select',
                                               'criteria': {'a': 'A', 'b': 'B'}}}}
    result = adapter.batch([row])[0]
    validate_answer(row, result)
    assert result['pred']['tags'] == ['a']

@pytest.mark.parametrize('reason,error', [('max_length_exceeded', ValueError),
                                         ('invalid_question', ValueError), ('invalid_model_output', RuntimeError)])
def test_native_errors_are_not_guessed_answers(reason, error):
    class Model:
        def system_one(self, **payload):
            return {'answers': {'q': {'error': reason}}}
    adapter = Decision2.__new__(Decision2)
    adapter.model = Model()
    with pytest.raises(error, match=reason):
        adapter.batch([{'state': 'x', 'questions': {'q': {'type': 'noul', 'instructions': 'x'}}}])
