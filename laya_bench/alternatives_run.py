"""Resumable full-fixture evaluation; failed cases remain in the denominator."""
import argparse
import json
import platform
import time
import traceback
import faulthandler
import math
import atexit
from datetime import datetime,timezone
from .common import ROOT,read_json,write_json,digest

def validate_answer(row,answer):
    """Catch broken adapters before their output can be mistaken for model errors."""
    assert set(answer['pred'])==set(row['questions'])
    assert set(answer['probabilities'])==set(row['questions'])
    for key,q in row['questions'].items():
        allowed=set(map(str,range(len(q['criteria'])))) if q['type']=='score' else set(q.get('criteria') or ['false','true'])
        p=answer['probabilities'][key]
        assert set(p)==allowed,(key,set(p),allowed)
        assert set(answer['pred'][key])<=allowed
        assert all(math.isfinite(v) and 0<=v<=1 for v in p.values())
        # SDKs may round probabilities (Laya to four decimals).
        if q['type']!='multilabel':assert abs(sum(p.values())-1)<=max(.001,len(p)*.000051)

def expected_failure(error,torch):
    return isinstance(error,(ValueError,torch.OutOfMemoryError)) or isinstance(error,RuntimeError) and (
        'CUDA out of memory' in str(error) or str(error) == 'bad allocation')

def run(name,batch_size=8,limit=None,suite=None,fixture='alternatives'):
    faulthandler.enable()
    faulthandler.dump_traceback_later(300,repeat=True)
    import torch
    from .alternatives_adapters import create
    torch.set_num_threads(8);torch.manual_seed(20261001)
    # WDDM otherwise spills large attention matrices into shared system memory.
    # Fail promptly within this local GPU budget and retry smaller batches.
    torch.cuda.set_per_process_memory_fraction(.78)
    torch.backends.cuda.matmul.allow_tf32=False
    path=ROOT/f'data/prepared/{fixture}.jsonl'
    protocol=read_json(ROOT/('results/alternatives/protocol.json' if fixture=='alternatives' else f'results/alternatives/{fixture}-protocol.json'))
    assert digest(path)==protocol['fixture_sha256']
    rows=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
    if suite:rows=[r for r in rows if r['suite'].startswith(suite)]
    if limit:rows=rows[:limit]
    runroot='runs' if fixture=='alternatives' else fixture
    out=ROOT/f'results/alternatives/{runroot}/{name}';out.mkdir(parents=True,exist_ok=True)
    dest=out/'predictions.jsonl';done={}
    if dest.exists():
        for line in dest.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);assert r['id'] not in done;done[r['id']]=r
    for r in rows:
        if r['id'] in done:assert done[r['id']]['input_sha256']==r['input_sha256']
    if name=='nimble-9b' and dest.exists() and (out/'metadata.json').exists():
        previous=read_json(out/'metadata.json')
        failed=[k for k,p in done.items() if p.get('error','').endswith('choice_descriptions must map valid choice names to text.')]
        if failed and not previous.get('adapter',{}).get('choice_description_none_fallback'):
            tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            archive=out/f'predictions-before-null-description-repair-{tag}.jsonl'
            archive.write_bytes(dest.read_bytes())
            write_json(out/f'metadata-before-null-description-repair-{tag}.json',previous)
            write_json(out/f'null-description-repair-{tag}.json',{'reason':'Translate null option descriptions to their existing label names, as required by the native schema. Retry only integration rejections; preserve successful predictions.','retry_ids':failed,'original_predictions_sha256':digest(archive)})
            done={k:p for k,p in done.items() if k not in set(failed)}
            temp=out/'predictions.repair.tmp';temp.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in done.values()),encoding='utf-8');temp.replace(dest)
            print('Archived and queued',len(failed),'schema integration rejections',flush=True)
    if name=='kev-4b' and dest.exists() and (out/'metadata.json').exists():
        previous=read_json(out/'metadata.json')
        failed=[k for k,p in done.items() if p.get('error','').startswith('OutOfMemoryError')]
        if failed and not previous.get('adapter',{}).get('explicit_row_budget'):
            # One documented integration repair. Preserve the exact old run and
            # retry only hardware failures; successful predictions are untouched.
            tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            archive=out/f'predictions-before-row-batching-{tag}.jsonl'
            archive.write_bytes(dest.read_bytes())
            write_json(out/f'metadata-before-row-batching-{tag}.json',previous)
            write_json(out/f'row-batching-repair-{tag}.json',{'reason':'Native 16384-token question-row batches exceed local GPU capacity; explicitly pass 2048 tokens/pass, including the Python default argument captured at import. Preserve complete context and weights.','retry_ids':failed,'original_predictions_sha256':digest(archive)})
            done={k:p for k,p in done.items() if k not in set(failed)}
            temp=out/'predictions.repair.tmp';temp.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in done.values()),encoding='utf-8');temp.replace(dest)
            print('Archived and queued',len(failed),'hardware failures after row-batching repair',flush=True)
    todo=[r for r in rows if r['id'] not in done]
    if not todo:print(name,'already complete for this selection',flush=True);return
    t=time.perf_counter();adapter=create(name)
    if hasattr(adapter,'close'):atexit.register(adapter.close)
    meta={'model':name,'fixture_sha256':digest(path),'adapter':adapter.metadata,'python':platform.python_version(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'loaded_seconds':time.perf_counter()-t,'started':datetime.now(timezone.utc).isoformat()}
    write_json(out/'metadata.json',meta)
    # Warm-up is deliberately outside measured task throughput.
    print(name,'warmup',flush=True)
    for probe in rows[:16]:
        try:
            adapter.batch([probe]);break
        except Exception as e:
            if not expected_failure(e,torch):raise
            torch.cuda.empty_cache()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();failures=0
    with dest.open('a',encoding='utf-8') as f:
        def evaluate(batch):
            nonlocal failures
            torch.cuda.synchronize();t=time.perf_counter()
            try:
                ans=adapter.batch(batch);torch.cuda.synchronize()
                assert len(ans)==len(batch)
                for r,a in zip(batch,ans):
                    validate_answer(r,a)
                elapsed=time.perf_counter()-t
            except Exception as e:
                if not expected_failure(e,torch):
                    # Interface/programming errors are not evidence of model accuracy.
                    raise
                if len(batch)>1:
                    torch.cuda.empty_cache()
                    for r in batch:evaluate([r])
                    return
                failures+=1;elapsed=time.perf_counter()-t
                ans=[{'pred':{},'probabilities':{},'error':f'{type(e).__name__}: {e}'}]
                if failures<=3:traceback.print_exc()
                torch.cuda.empty_cache()
            for r,a in zip(batch,ans):
                result={**{k:r[k] for k in ['id','suite','input_sha256','gold']},**a,'model':name,'batch_size':len(batch),'seconds_per_item_in_batch':elapsed/len(batch)}
                result['correct']=set(a['pred'])==set(r['gold']) and all(set(a['pred'][k])==set(g) for k,g in r['gold'].items())
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
        for start in range(0,len(todo),batch_size):
            evaluate(todo[start:start+batch_size]);f.flush()
            if start//batch_size%10==0 or start+batch_size>=len(todo):
                print(name,f'{min(start+batch_size,len(todo))}/{len(todo)}',f'{time.perf_counter()-started:.1f}s',f'errors={failures}',todo[start]['suite'],flush=True)
                write_json(out/'progress.json',{'completed_total':len(done)+min(start+batch_size,len(todo)),'requested_selection':len(rows),'new_errors':failures,'elapsed_seconds':time.perf_counter()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
    meta['finished']=datetime.now(timezone.utc).isoformat();meta['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    meta['predictions_sha256']=digest(dest);write_json(out/'metadata.json',meta)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--limit',type=int);p.add_argument('--suite');p.add_argument('--fixture',default='alternatives');a=p.parse_args()
    run(a.model,a.batch_size,a.limit,a.suite,a.fixture)
