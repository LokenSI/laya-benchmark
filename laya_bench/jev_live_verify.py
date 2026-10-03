"""Read-only reconciliation of frozen requests, raw responses and paid usage."""
from collections import Counter
import hashlib
import json
from .common import ROOT,read_json,write_json,digest,fingerprint
from .alternatives_jev import read_rows
from .alternatives_adapters import typed_questions
from .jev_api import OUT,PRICE_PER_TOKEN,CAP_USD,now

def verify():
    protocol=read_json(OUT/'protocol.json')
    path=ROOT/'data/prepared/jev_live.jsonl'
    assert digest(path)==protocol['fixture_sha256']
    rows=read_rows(path);byid={r['id']:r for r in rows}
    assert len(rows)==len(byid)==protocol['n']
    for name,meta in protocol['sources'].items():
        source=ROOT/f'data/prepared/{name}.jsonl'
        assert digest(source)==meta['sha256']
        for r in read_rows(source):
            assert {k:v for k,v in byid[r['id']].items() if k!='source_fixture'}==r
    for r in rows:
        assert r['input_sha256']==fingerprint({'state':r['state'],'questions':r['questions']})
    main=read_rows(OUT/'predictions.jsonl');assert len(main)==len({r['id'] for r in main})
    assert {p['id'] for p in main}==set(byid),'Main run incomplete'
    diagnostics=read_rows(OUT/'diagnostic-predictions.jsonl')
    diagnostic_rows={r['id']:r for r in read_rows(ROOT/'data/prepared/jev_diagnostics.jsonl')}
    assert len(diagnostics)==len({p['id'] for p in diagnostics})
    attempts={};errors=Counter();model=protocol['model']['model']
    for p in main+diagnostics:
        r=(diagnostic_rows if p['id'].startswith('diagnostic/') else byid)[p['id']]
        expected={'model':model,'state':r['state'],'questions':typed_questions(r['questions'])[0]}
        # Serialized equality includes criterion order. No expected labels or
        # provenance fields are permitted in the request envelope.
        assert json.dumps(p['request'],ensure_ascii=False)==json.dumps(expected,ensure_ascii=False)
        assert p['request_ordered_sha256']==hashlib.sha256(json.dumps(expected,ensure_ascii=False).encode()).hexdigest()
        assert p['input_sha256']==r['input_sha256'] and p['gold']==r['gold']
        assert set(p['request'])=={'model','state','questions'}
        assert all(set(q)<={'type','instructions','criteria'} for q in p['request']['questions'].values())
        correct=set(p['pred'])==set(r['gold']) and all(set(p['pred'][k])==set(g) for k,g in r['gold'].items())
        assert p['correct']==correct
        if p.get('error'):errors[p['error']]+=1
        for a in p['attempts']:
            assert a['attempt'] not in attempts;attempts[a['attempt']]=(p['id'],a)
            if 'response' in a:assert a['response']['model']==model
    latency=read_rows(ROOT/'results/latency/api-warmup.jsonl')+read_rows(ROOT/'results/latency/jev-1.13.0.jsonl')
    if latency:
        latency_rows={r['id']:r for r in read_rows(ROOT/'data/prepared/latency.jsonl')}
        for p in latency:
            r=latency_rows[p['id']];payload={'model':model,'state':r['state'],'questions':r['questions']}
            assert json.dumps(p['request'],ensure_ascii=False)==json.dumps(payload,ensure_ascii=False)
            assert p['input_sha256']==r['input_sha256']
            a=p['raw'];case=f'latency/{p["pass"]}/{p["id"]}'
            assert a['attempt'] not in attempts;attempts[a['attempt']]=(case,a)
            if 'response' in a:assert a['response']['model']==model
    ledger=read_rows(OUT/'usage-ledger.jsonl');reserved={};settled={};balance=0.;peak=0.
    for e in ledger:
        if e['event']=='reserve':
            assert e['attempt'] not in reserved
            reserved[e['attempt']]=e;balance+=e['reserved_usd']
        else:
            assert e['event']=='settle' and e['attempt'] in reserved and e['attempt'] not in settled
            settled[e['attempt']]=e
            balance+=e['input_tokens']*PRICE_PER_TOKEN-reserved[e['attempt']]['reserved_usd']
        peak=max(peak,balance);assert balance<=CAP_USD+1e-9
    for aid,(case,a) in attempts.items():
        assert reserved[aid]['case_id']==case
        if 'response' in a:
            assert settled[aid]['input_tokens']==a['response']['usage']['input_tokens']
            assert settled[aid]['output_tokens']==a['response']['usage']['output_tokens']
    tokens=sum(e['input_tokens'] for e in settled.values())
    unconfirmed={k:v for k,v in reserved.items() if k not in settled}
    spend=read_json(OUT/'spend.json')
    assert spend['billable_input_tokens']==tokens
    assert abs(spend['estimated_cost_usd']-tokens*PRICE_PER_TOKEN)<1e-9
    extras=[v['case_id'] for k,v in reserved.items() if k not in attempts]
    local_counts={}
    for meta in (ROOT/'results/alternatives/models').glob('*.json'):
        count=0
        for fixture in protocol['sources']:
            folder='runs' if fixture=='alternatives' else fixture
            values=read_rows(ROOT/f'results/alternatives/{folder}/{meta.stem}/predictions.jsonl')
            assert len(values)==len({p['id'] for p in values})
            for p in values:
                assert p['id'] in byid and p['input_sha256']==byid[p['id']]['input_sha256']
            count+=len(values)
        local_counts[meta.stem]=count
    result={'updated':now(),'status':'verified','main_cases':len(main),'diagnostic_cases':len(diagnostics),'latency_calls_including_warmup':len(latency),
            'diagnostics_expected':len(diagnostic_rows),'model':model,'fixture_sha256':digest(path),
            'checks':['All frozen source hashes match','No duplicate case IDs','Exact ordered API payloads match frozen inputs',
                      'No labels or provenance added to request envelope','Saved correctness reconciles with frozen gold',
                      'Every raw successful response uses the pinned model','Each API attempt reconciles with the budget ledger',
                      'No observed ledger balance exceeded the cap','Available local predictions match shared input hashes'],
            'errors_including_diagnostics':dict(errors),'extra_probe_case_ids':extras,'local_prediction_counts':local_counts,
            'billable_input_tokens':tokens,'estimated_cost_usd':tokens*PRICE_PER_TOKEN,
            'unconfirmed_attempts':len(unconfirmed),'reserved_unconfirmed_usd':sum(e['reserved_usd'] for e in unconfirmed.values()),
            'peak_budget_accounted_usd':peak,'cap_usd':CAP_USD,
            'limits':'This audit verifies execution and accounting, not human label correctness, training-data independence or production readiness.'}
    write_json(OUT/'verification.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':verify()
