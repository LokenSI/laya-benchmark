"""Repeat public claims with the shipped SDK; retain every decision and error."""
import gc
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from .common import ROOT, digest, fingerprint, read_json, write_json
from .adapter import LayaAdapter, decode
from .metrics import calibration, classification


def score(records, labels):
    result = classification([r["gold"] for r in records], [r["pred"] for r in records], labels)
    # Options can differ across cases in MASSIVE intent. Use per-case binary
    # confidence/correctness ECE separately from fixed-label multiclass Brier.
    fixed = all(r["options"] == records[0]["options"] for r in records)
    if fixed:
        options=records[0]["options"]
        result["calibration"]=calibration([options.index(r["gold"]) for r in records], [r["p"] for r in records], [options.index(r["pred"]) for r in records])
    result["majority_accuracy"]=max(Counter(r["gold"] for r in records).values())/len(records)
    result["state_truncated"]=sum(r["audit"]["state_truncated"] for r in records)
    result["options_truncated"]=sum(bool(r["audit"]["options_truncated"]) for r in records)
    result["inference_seconds"]=sum(r["seconds_per_item_in_batch"] for r in records)
    result["ms_per_item_in_batch"]=1000*result["inference_seconds"]/len(records)
    return result


def run():
    import torch
    import laya
    from laya.common import serialize_state
    torch.manual_seed(20260926)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    fixture=ROOT/"data/prepared/claims.json"
    data=read_json(fixture)
    out=ROOT/"results/claims"
    out.mkdir(parents=True,exist_ok=True)
    summary={"started":datetime.now(timezone.utc).isoformat(),"fixture_sha256":digest(fixture),"manifest":data["manifest"],
        "environment":{"python":platform.python_version(),"torch":torch.__version__,"laya":getattr(laya,"__version__","0.3.20"),"gpu":torch.cuda.get_device_name(0)},"models":{},"results":{}}
    with (out/"predictions.jsonl").open("w",encoding="utf-8") as stream:
        for model,pub_model in [("laya","english"),("laya-multilingual","multilingual")]:
            adapter=LayaAdapter(model,offline=True)
            summary["models"][model]=adapter.metadata
            for name,suite in data["suites"].items():
                cases=suite["cases"]
                records=[]
                # Keep question insertion order: it is part of the input under test.
                same=all(c["question"]==cases[0]["question"] for c in cases)
                batch_size=8 if same else 1
                for start in range(0,len(cases),batch_size):
                    batch=cases[start:start+batch_size]
                    q={"decision":batch[0]["question"]}
                    answers,seconds=adapter.predict([c["state"] for c in batch],q)
                    for c,a in zip(batch,answers):
                        labels=list(c["question"]["criteria"]) if c["question"]["type"]=="choice" else ["false","true"]
                        pred,p=decode(a["answers"]["decision"],labels)
                        row={"suite":name,"model":model,"id":c["id"],"gold":c["gold"],"pred":pred,"p":p,"options":labels,
                             "input_hash":fingerprint(c),"seconds_per_item_in_batch":seconds/len(batch),"audit":adapter.audit(serialize_state(c["state"]),q)["decision"]}
                        stream.write(json.dumps(row,ensure_ascii=False)+"\n")
                        records.append(row)
                measured=score(records,suite["labels"])
                measured["phase"]=suite["phase"]
                measured["publisher_task_in_training"]=suite["publisher_task_in_training"]
                if suite["published"]:
                    expected=suite["published"][pub_model]
                    measured.update(published_accuracy=expected,difference_pp=100*(measured["accuracy"]-expected),
                        verdict="reproduced within 2 percentage points" if abs(measured["accuracy"]-expected)<=.02000001 else "not reproduced within 2 percentage points")
                summary["results"][f"{model}/{name}"]=measured
                stream.flush()
                write_json(out/"summary.json",summary)
                print(model,name,f'{measured["accuracy"]:.1%}',measured.get("verdict","extension"),flush=True)
            adapter.close()
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
    summary["finished"]=datetime.now(timezone.utc).isoformat()
    summary["predictions_sha256"]=digest(out/"predictions.jsonl")
    write_json(out/"summary.json",summary)

if __name__=="__main__":
    run()
