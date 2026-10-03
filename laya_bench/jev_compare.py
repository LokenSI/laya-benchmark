"""Reconstruct the independent Jev pilot and verify its published manifest hash.

Jev scores are published observations, never represented as a fresh API run.
"""
import argparse
import gc
import hashlib
import json
import random
from collections import defaultdict

from .common import ROOT, digest, read_json, write_json

REVISION="fef2a2ac62b69c58670047dddf045c53d7c3cb5e"


def balanced_positions(targets, limit, seed):
    groups=defaultdict(list)
    for i,target in enumerate(targets):
        groups[target].append(i)
    rng=random.Random(seed)
    for positions in groups.values():
        rng.shuffle(positions)
    chosen=[]
    while len(chosen)<min(limit,len(targets)):
        for label in sorted(groups):
            if groups[label] and len(chosen)<limit:
                chosen.append(groups[label].pop())
    return sorted(chosen)


def prepare():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    manifest=[]
    sources={}
    for offset,(name,task) in enumerate([("agnews","topic"),("emotiondair","emotion"),("banking77","intent")]):
        filename=f"{name}/test-00000-of-00001.parquet"
        path=hf_hub_download("btzsc/btzsc",filename,repo_type="dataset",revision=REVISION)
        rows=pq.read_table(path).to_pylist()
        k=next(i for i in range(1,len(rows)) if rows[i]["text"]!=rows[0]["text"])
        assert len(rows)%k==0
        labels=[str(r["hypothesis"]) for r in rows[:k]]
        valid=[]
        targets=[]
        for start in range(0,len(rows),k):
            values=[int(r["labels"]) for r in rows[start:start+k]]
            assert [str(r["hypothesis"]) for r in rows[start:start+k]]==labels
            if sum(values)==1:
                valid.append(start//k)
                targets.append(values.index(1))
        for pos in balanced_positions(targets,100,20260917+offset):
            i=valid[pos]
            text=str(rows[i*k]["text"])
            manifest.append({"dataset":name,"task":task,"example_id":f"{name}:{i}","text":text,
                "text_sha256":hashlib.sha256(text.encode()).hexdigest(),"labels":labels,"target_index":targets[pos]})
        sources[name]={"file":filename,"sha256":digest(path),"labels":k,"excluded_without_single_positive":len(rows)//k-len(valid)}
    path=ROOT/"data/prepared/jev_manifest.jsonl"
    path.write_bytes(("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in manifest)).encode())
    published=read_json(ROOT/".cache/jev/results/reports/btzsc-pilot-v1.json")
    observed=digest(path)
    expected=published["artifacts"]["manifest_sha256"]
    if observed!=expected:
        raise ValueError(f"Independent Jev fixture hash mismatch: {observed} != {expected}")
    write_json(ROOT/"results/jev/provenance.json",{"repo":"btzsc/btzsc","revision":REVISION,"sources":sources,
        "manifest_sha256":observed,"published_manifest_sha256":expected,"exact_manifest_match":True,
        "reference_repo_commit":(ROOT/".cache/jev/commit.txt").read_text().strip(),"published":published})
    print("Verified exact published Jev manifest SHA-256; 300 identical examples and label orders.",flush=True)


def run():
    import numpy as np
    import torch
    from laya.common import serialize_state
    from .adapter import LayaAdapter,decode
    from .metrics import classification,calibration
    torch.set_num_threads(8)
    torch.manual_seed(20260926)
    torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/"results/jev"
    provenance=read_json(out/"provenance.json")
    path=ROOT/"data/prepared/jev_manifest.jsonl"
    assert digest(path)==provenance["published_manifest_sha256"]
    cases=[json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    summary={"provenance":provenance,"results":{},"models":{},"jev_was_run_locally":False,
        "scope":"Identical 300-example manifest, instructions, states, labels and order. Jev 1.13.0 scores are the independent author's earlier hosted run. No paired significance test possible without their per-case predictions; no normalized speed ranking."}
    with (out/"predictions.jsonl").open("w",encoding="utf-8") as stream:
        for model in ["laya","laya-multilingual"]:
            adapter=LayaAdapter(model,offline=True)
            summary["models"][model]=adapter.metadata
            for dataset in ["agnews","emotiondair","banking77"]:
                subset=[c for c in cases if c["dataset"]==dataset]
                labels=[f"label_{i:03d}" for i in range(len(subset[0]["labels"]))]
                q={"label":{"type":"choice","instructions":"Which single label best describes the input text?",
                            "criteria":dict(zip(labels,subset[0]["labels"]))}}
                settings=[("default",{})]
                if dataset=="banking77":
                    settings.append(("expanded_options",{"max_len":4096,"head_max_len":3072}))
                for variant,kwargs in settings:
                    records=[]
                    # Batch one for comparable per-request local service latency, with one warmup.
                    adapter.predict([{"text":subset[0]["text"]}],q,**kwargs)
                    for case in subset:
                        state={"text":case["text"]}
                        answers,elapsed=adapter.predict([state],q,**kwargs)
                        pred,p=decode(answers[0]["answers"]["label"],labels)
                        row={"model":model,"dataset":dataset,"variant":variant,"id":case["example_id"],
                            "gold":labels[case["target_index"]],"pred":pred,"p":p,"seconds":elapsed,
                            "audit":adapter.audit(serialize_state(state),q,**kwargs)["label"]}
                        records.append(row)
                        stream.write(json.dumps(row)+"\n")
                    y=[r["gold"] for r in records]
                    pred=[r["pred"] for r in records]
                    measured=classification(y,pred,labels)
                    measured["calibration"]=calibration([labels.index(v) for v in y],[r["p"] for r in records],[labels.index(v) for v in pred])
                    measured["latency_p50_ms"]=1000*float(np.median([r["seconds"] for r in records]))
                    measured["latency_p95_ms"]=1000*float(np.quantile([r["seconds"] for r in records],.95))
                    measured["state_truncated"]=sum(r["audit"]["state_truncated"] for r in records)
                    measured["options_truncated_cases"]=sum(bool(r["audit"]["options_truncated"]) for r in records)
                    measured["settings"]=kwargs
                    measured["published_jev_accuracy"]=provenance["published"]["results"]["jev"][dataset]["accuracy"]
                    summary["results"][f"{model}/{dataset}/{variant}"]=measured
                    stream.flush()
                    write_json(out/"summary.json",summary)
                    print(model,dataset,variant,f'{measured["accuracy"]:.1%}',flush=True)
            adapter.close()
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
    summary["predictions_sha256"]=digest(out/"predictions.jsonl")
    write_json(out/"summary.json",summary)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["prepare","run"])
    args=p.parse_args()
    prepare() if args.action=="prepare" else run()
