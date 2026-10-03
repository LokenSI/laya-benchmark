import numpy as np
import pytest
from laya_bench.adapter import decode
from laya_bench.metrics import (calibration, classification, cluster_interval, fit_temperature,
                               paired_comparison, probabilities, rescale, select_threshold, selective, wilson)

def test_confusion_and_macro_f1_do_not_hide_minority_failure():
    r = classification(['a']*9+['b'], ['a']*10, ['a','b'])
    assert r['accuracy'] == .9
    assert r['balanced_accuracy'] == .5
    assert r['macro_f1'] == pytest.approx((18/19)/2)
    assert r['confusion_matrix'] == [[9,0],[1,0]]
    assert r['per_class']['b']['recall'] == 0

def test_probability_metrics_against_hand_calculation():
    r = calibration([0,1], [[.8,.2],[.6,.4]])
    assert r['brier'] == pytest.approx(.4)
    assert r['nll'] == pytest.approx((-np.log(.8)-np.log(.4))/2)
    assert r['ece'] == pytest.approx(.4)

def test_confidence_one_is_included_in_last_bin():
    r = calibration([0,1], [[1.,0.],[1.,0.]])
    assert r['reliability'][-1]['n'] == 2
    assert r['ece'] == .5
    assert r['confident_errors'] == 1
    assert np.isfinite(r['nll'])

def test_sdk_rounding_is_renormalized_but_invalid_output_is_rejected():
    assert probabilities([[.3333,.3333,.3333]]).sum() == pytest.approx(1.)
    for bad in [[[0,0]],[[.5,np.nan]],[[-.1,1.1]],[[2,3]],[[np.inf,0]]]:
        with pytest.raises(ValueError):
            probabilities(bad)

def test_choice_preserves_sdk_prediction_when_rounding_creates_tie():
    pred, p = decode({'type':'choice','choice':'B','probabilities':{'A':.5,'B':.5}},['A','B'])
    assert pred == 'B'
    assert calibration([1],[p],[1])['nll'] == pytest.approx(np.log(2))

def test_binary_probability_means_true_not_entropy():
    pred,p = decode({'type':'noul','noul':.9,'confidence':.12},['0','1'])
    assert pred == '1'
    assert p == pytest.approx([.1,.9])

def test_decode_uses_explicit_label_mapping_not_dictionary_order():
    pred,p=decode({'type':'choice','choice':'negative','probabilities':{'positive':.1,'negative':.9}},['negative','positive'])
    assert pred == 'negative'
    assert p == [.9,.1]

def test_temperature_can_reduce_overconfidence_without_changing_argmax():
    p = np.array([[.99,.01]]*100)
    y = np.array([0]*60+[1]*40)
    t = fit_temperature(p,y)
    q = rescale(p,t)
    assert t > 1
    np.testing.assert_array_equal(p.argmax(1),q.argmax(1))
    assert calibration(y,q)['nll'] < calibration(y,p)['nll']

def test_policy_does_not_qualify_on_tiny_perfect_samples():
    assert select_threshold([[.99,.01]]*10,[0]*10) is None
    assert select_threshold([[.99,.01]]*100,[0]*90+[1]*10) is None
    chosen = select_threshold([[.99,.01]]*100,[0]*100)
    assert chosen['validation_wilson95'][0] > .95

def test_absent_policy_routes_everything_to_review():
    r = selective([[.99,.01]]*100,[0]*100,[0]*100,None)
    assert r['accepted'] == 0
    assert r['coverage'] == 0
    assert r['accuracy'] is None
    assert r['review_per_1000'] == 1000

def test_test_failures_are_not_hidden_by_validation_policy():
    policy = select_threshold([[.99,.01]]*100,[0]*100)
    r = selective([[.99,.01]]*100,[1]*100,[0]*100,policy['threshold'])
    assert r['accuracy'] == 0
    assert r['errors_per_1000_total'] == 1000

def test_wilson_is_not_100_percent_certainty_for_perfect_sample():
    assert wilson(0,0) == [None,None]
    low,high=wilson(100,100)
    assert .96 < low < 1
    assert high == pytest.approx(1)

def test_group_bootstrap_does_not_treat_repeated_sentences_as_independent():
    values=[0]*100+[1]*100
    interval=cluster_interval(values,['review-a']*100+['review-b']*100,repeats=1000)
    assert interval == [0.,1.]

def test_paired_comparison_aligns_ids_and_direction():
    a=[{'id':'1','group':'1','label':'x','prediction':'x'},{'id':'2','group':'2','label':'x','prediction':'y'}]
    b=[{**a[1],'prediction':'x'},a[0]]
    r=paired_comparison(a,b)
    assert r['delta_accuracy_b_minus_a'] == .5
    assert r['b_only_correct'] == 1
    with pytest.raises(ValueError):
        paired_comparison(a,b[:1])
    with pytest.raises(ValueError):
        paired_comparison(a,[{**b[0],'label':'y'},b[1]])

