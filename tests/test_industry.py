import pytest
from laya_bench.industry import validate_input,asset_tags,question,keyword_route
from laya_bench.industry_cases import build_cases
from laya_bench.jev_compare import balanced_positions


def test_translation_families_never_cross_splits():
    cases=build_cases()
    assert len(cases)==200
    dev={c['family'] for c in cases if c['split']=='dev'}
    test={c['family'] for c in cases if c['split']=='test'}
    assert len(dev)==32 and len(test)==68 and not dev & test
    for family in dev|test:
        paired=[c for c in cases if c['family']==family]
        assert {c['language'] for c in paired}=={'en','nb'}
        assert len({c['gold'] for c in paired})==1


@pytest.mark.parametrize('payload',[{},[],{'text':''},{'text':'a'*4001,'task':'routing','language':'en'},
    {'text':'x','task':'execute','language':'en'},{'text':'x','task':'routing','language':'unknown'}])
def test_reject_invalid_requests(payload):
    with pytest.raises(ValueError):
        validate_input(payload)


def test_asset_copying_does_not_invent_identifiers():
    assert asset_tags('P-101 and XV-12, then P-101 again. A pump has no tag.')==['P-101','XV-12']
    assert asset_tags('pump one hundred one')==[]


def test_localized_prompts_preserve_answer_contract():
    for task in ['routing','documents']:
        assert list(question(task,'en','plain')['decision']['criteria'])==list(question(task,'nb','contextual')['decision']['criteria'])


def test_rules_abstain_when_there_is_no_evidence():
    assert keyword_route('Hello, how are you?','routing')=='review'


def test_balancing_exhausts_minorities_then_continues():
    targets=[0]*7+[1]*2+[2]
    chosen=balanced_positions(targets,8,42)
    assert len(chosen)==8 and len(set(chosen))==8
    assert {targets[i] for i in chosen}=={0,1,2}
    assert chosen==balanced_positions(targets,8,42)
