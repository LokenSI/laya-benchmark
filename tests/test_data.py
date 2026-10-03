import pytest
from laya_bench.data import clean_splits, normalized, sample
from laya_bench.challenges import cases, minimal_pairs
from laya_bench.questions import support_questions

def row(id,text,group=None):
    return {'id':id,'text':text,'group':group or id,'label':'positive'}

def test_test_set_has_priority_when_removing_cross_split_duplicates():
    source={'train':[row('a','SAME text'),row('b','unique train')],
            'validation':[row('c',' same  text '),row('d','unique validation')],
            'test':[row('e','Same text'),row('f','same TEXT')]}
    result,audit=clean_splits(source)
    assert [r['id'] for r in result['test']] == ['e']
    assert [r['id'] for r in result['validation']] == ['d']
    assert [r['id'] for r in result['train']] == ['b']
    assert audit['test']['removed_exact_text_duplicates'] == 1

def test_document_leakage_rejected_even_if_texts_differ():
    source={'train':[row('1','one','same-review')], 'validation':[row('2','two')],
            'test':[row('3','three','same-review')]}
    with pytest.raises(ValueError,match='Source document leakage'):
        clean_splits(source,grouped=True)

def test_sample_is_reproducible_input_order_and_label_independent():
    rows=[row(str(i),str(i)) for i in range(100)]
    ids=lambda rs:[r['id'] for r in rs]
    assert ids(sample(rows,20,42)) == ids(sample(list(reversed(rows)),20,42))
    assert ids(sample(rows,20,42)) == ids(sample([{**r,'label':'different'} for r in rows],20,42))
    assert len(sample(rows,0,42)) == 100
    assert ids(sample(rows,20,42)) != ids(sample(rows,20,43))

def test_diagnostics_have_complete_translations_and_valid_labels():
    rows=list(cases())
    assert len(rows) == 90
    assert len({r['id'] for r in rows}) == 90
    groups={r['group'] for r in rows}
    for group in groups:
        variants=[r for r in rows if r['group']==group]
        assert {r['language'] for r in variants} == {'en','nb','nn'}
        assert all(r['expected']==variants[0]['expected'] for r in variants)
    for r in rows:
        assert r['expected']['route'] in support_questions()['route']['criteria']
        assert r['expected']['refund_choice'] == ('B' if r['expected']['refund']=='1' else 'A')

def test_minimal_pairs_are_balanced_in_each_language():
    rows=list(minimal_pairs())
    for language in ['en','nb','nn']:
        labels=[r['expected']['refund'] for r in rows if r['language']==language]
        assert labels.count('0') == labels.count('1') == 6

