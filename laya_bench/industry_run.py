import gc
import json
from datetime import datetime,timezone

import numpy as np

from .common import ROOT,digest,fingerprint,read_json,write_json
from .adapter import LayaAdapter,decode
from .industry import question,keyword_route,ROUTES,DOCS
from .metrics import classification,calibration,wilson


def evaluate(rows,labels):
    pred=[r["pred"] for r in rows]
    extra=[] if set(pred).issubset(labels) else ["review"]
    result=classification([r["gold"] for r in rows],pred,labels+extra)
    result["suggestion_coverage"]=sum(v!="review" for v in pred)/len(pred)
    suggested=[r for r in rows if r["pred"]!="review"]
    result["suggested_accuracy"]=sum(r["pred"]==r["gold"] for r in suggested)/len(suggested) if suggested else None
    result["correct_suggestions"]=sum(r["pred"]==r["gold"] for r in suggested)
    result["wrong_suggestions"]=sum(r["pred"]!=r["gold"] for r in suggested)
    return result


def run():
    import torch
    torch.set_num_threads(8)
    torch.manual_seed(20260926)
    torch.backends.cuda.matmul.allow_tf32=False
    path=ROOT/"data/prepared/industry.json"
    data=read_json(path)
    assert fingerprint(data["cases"])==data["cases_sha256"]
    cases=data["cases"]
    devfamilies={c["family"] for c in cases if c["split"]=="dev"}
    testfamilies={c["family"] for c in cases if c["split"]=="test"}
    assert not devfamilies & testfamilies
    out=ROOT/"results/industry"
    out.mkdir(parents=True,exist_ok=True)
    summary={"started":datetime.now(timezone.utc).isoformat(),"fixture_sha256":digest(path),"case_fingerprint":data["cases_sha256"],
        "description":data["description"],"dev_families":len(devfamilies),"test_families":len(testfamilies),
        "policy":"All suggestions require human review. No automatic work orders, purchases, operational instructions or safety decisions.",
        "selection":"Two prompt variants per model/task/language, chosen by dev accuracy only; ties prefer plain. Final workflow chosen using exploratory test results; requires new external validation.",
        "models":{},"results":{},"dev":{},"selected":{}}
    with (out/"predictions.jsonl").open("w",encoding="utf-8") as stream:
        for model in ["laya","laya-multilingual"]:
            adapter=LayaAdapter(model,offline=True)
            summary["models"][model]=adapter.metadata
            def infer(subset,q,variant):
                labels=list(q["decision"]["criteria"])
                records=[]
                for start in range(0,len(subset),8):
                    batch=subset[start:start+8]
                    answers,seconds=adapter.predict([c["text"] for c in batch],q)
                    for case,a in zip(batch,answers):
                        pred,p=decode(a["answers"]["decision"],labels)
                        row={**case,"model":model,"variant":variant,"pred":pred,"p":p,"options":labels,
                             "seconds_per_item_in_batch":seconds/len(batch),"audit":adapter.audit(case["text"],q)["decision"]}
                        stream.write(json.dumps(row,ensure_ascii=False)+"\n")
                        records.append(row)
                return records
            for task in ["routing","documents"]:
                for lang in ["en","nb"]:
                    key=f"{model}/{task}/{lang}"
                    labels=list(ROUTES if task=="routing" else DOCS)
                    dev=[c for c in cases if c["task"]==task and c["language"]==lang and c["split"]=="dev"]
                    scores={}
                    for variant in ["plain","contextual"]:
                        rows=infer(dev,question(task,lang,variant),variant)
                        scores[variant]=evaluate(rows,labels)
                    selected=max(scores,key=lambda variant:scores[variant]["accuracy"])
                    summary["dev"][key]=scores
                    summary["selected"][key]=selected
                    test=[c for c in cases if c["task"]==task and c["language"]==lang and c["split"]=="test"]
                    q=question(task,lang,selected)
                    rows=infer(test,q,selected)
                    measured=evaluate(rows,labels)
                    measured["calibration"]=calibration([labels.index(r["gold"]) for r in rows],[r["p"] for r in rows],[labels.index(r["pred"]) for r in rows])
                    measured["state_truncated"]=sum(r["audit"]["state_truncated"] for r in rows)
                    measured["options_truncated"]=sum(bool(r["audit"]["options_truncated"]) for r in rows)
                    measured["errors"]=[{"id":r["id"],"text":r["text"],"gold":r["gold"],"pred":r["pred"]} for r in rows if r["gold"]!=r["pred"]]
                    # Identical holdout, reversed options: stability diagnostic, not prompt selection.
                    q["decision"]["criteria"]=dict(reversed(list(q["decision"]["criteria"].items())))
                    reversed_rows=infer(test,q,"reversed")
                    measured["reversed_accuracy"]=sum(r["gold"]==r["pred"] for r in reversed_rows)/len(rows)
                    measured["option_order_flip_rate"]=sum(a["pred"]!=b["pred"] for a,b in zip(rows,reversed_rows))/len(rows)
                    # Warm single-request timings; excluded from task accuracy.
                    q=question(task,lang,selected)
                    times=[]
                    for c in test[:10]:
                        _,seconds=adapter.predict([c["text"]],q)
                        times.append(seconds*1000)
                    measured["latency_p50_ms"]=float(np.median(times))
                    measured["latency_p95_ms"]=float(np.quantile(times,.95))
                    measured["latency_n"]=len(times)
                    summary["results"][key]=measured
                    baseline=[{**c,"pred":keyword_route(c["text"],task)} for c in test]
                    summary["results"][f"rules/{task}/{lang}"]=evaluate(baseline,labels)
                    print(key,selected,f'{measured["accuracy"]:.1%}',"rules",f'{summary["results"][f"rules/{task}/{lang}"]["accuracy"]:.1%}',flush=True)
                    stream.flush()
                    write_json(out/"summary.json",summary)
            adapter.close()
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
    summary["finished"]=datetime.now(timezone.utc).isoformat()
    summary["predictions_sha256"]=digest(out/"predictions.jsonl")
    write_json(out/"summary.json",summary)

if __name__=="__main__":
    run()
