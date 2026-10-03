import hashlib
import json
import pytest
from laya_bench.common import ROOT,read_json,digest
from laya_bench.alternatives_jev import read_rows,paired_summary,pilot_metrics,prepare

def test_publisher_artifacts_and_case_membership_verify():
    result=prepare()
    assert result['public_case_count']==231
    assert result['pilot_case_count']==300
    assert result['public_cases_match_existing_fixture']
    assert result['pilot_manifest_matches_published_hash']

def test_pilot_retains_exact_request_and_option_order():
    native=read_rows(ROOT/'data/prepared/jev_manifest.jsonl')
    rows=read_rows(ROOT/'data/prepared/jev_verified.jsonl')
    assert len(rows)==len(native)==300
    for source,row in zip(native,rows):
        assert row['state']=={'text':source['text']}
        assert list(row['questions'])==['label']
        q=row['questions']['label']
        assert q['instructions']=='Which single label best describes the input text?'
        assert list(q['criteria'])==[f'label_{i:03d}' for i in range(len(source['labels']))]
        assert list(q['criteria'].values())==source['labels']
        assert row['gold']['label']==[f'label_{source["target_index"]:03d}']
        ordered={'state':row['state'],'questions':row['questions']}
        assert row['ordered_input_sha256']==hashlib.sha256(json.dumps(ordered,ensure_ascii=False).encode()).hexdigest()
    assert {len(r['questions']['label']['criteria']) for r in rows if r['suite'].endswith('banking77')}=={72}

def test_paired_difference_preserves_case_alignment_and_groups():
    cases=[{'suite':'original','id':str(i),'native_task':{'group':str(i//2)}} for i in range(4)]
    result=paired_summary(cases,[1,0,1,0],[1,0,1,0])
    assert result['paired_cluster_bootstrap95_pp']==[0.,0.]
    assert result['independent_groups']==2
    assert result['local_only_correct']==result['jev_only_correct']==0
    result=paired_summary(cases,[1,0,1,0],[0,1,0,1])
    assert result['local_only_correct']==result['jev_only_correct']==2
    assert result['both_correct']==result['both_wrong']==0

def test_pilot_failure_counts_and_tie_use_publisher_rules():
    case={'suite':'jev_verified/example','source_id':'example','questions':{'label':{'criteria':{'label_000':'a','label_001':'b'}}},'gold':{'label':['label_000']}}
    tie={'model':'test','probabilities':{'label':{'label_001':.5,'label_000':.5}},'pred':{'label':['label_001']}}
    failed={'model':'test','probabilities':{},'pred':{},'error':'ValueError: unsupported'}
    result=pilot_metrics([case,case],[tie,failed])
    assert result['accuracy']==.5
    assert result['valid']==1 and result['failures']==1
    assert result['macro_f1']==pytest.approx(1/3)
