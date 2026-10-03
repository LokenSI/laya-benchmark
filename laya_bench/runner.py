import csv
import importlib.metadata
import json
import platform
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import ROOT, digest, fingerprint, read_json, write_json
from .adapter import LayaAdapter, decode
from .metrics import (calibration, classification, fit_temperature, paired_comparison,
                      rescale, select_threshold, selective)
from .questions import question, support_questions

def chunks(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start+size]

def evaluate_public(adapter, name, suite, batch_size, sink):
    q = question(suite)
    labels = suite["labels"]
    all_rows, times = {}, []
    adapter.predict([suite["splits"]["validation"][0]["text"]], q)
    for split in ["validation", "test"]:
        predictions = []
        rows = suite["splits"][split]
        for chunk in chunks(rows, batch_size):
            answers, seconds = adapter.predict([r["text"] for r in chunk], q)
            if split == "test":
                times.append({"n": len(chunk), "seconds": seconds})
            for row, answer in zip(chunk, answers):
                predicted, p = decode(answer["answers"]["decision"], labels)
                result = {**row, "model": adapter.name, "suite": name, "split": split,
                          "prediction": predicted, "probabilities": p, "labels": labels,
                          "audit": adapter.audit(row["text"], q)["decision"],
                          "sdk_answer": answer["answers"]["decision"]}
                sink(result)
                predictions.append(result)
            if len(predictions) % 200 < batch_size:
                print(f"{adapter.name} {name} {split}: {len(predictions)}/{len(rows)}", flush=True)
        all_rows[split] = predictions
    test, val = all_rows["test"], all_rows["validation"]
    # Calibration and threshold selection use separate source-document groups.
    groups = sorted({r["group"] for r in val}, key=lambda g: fingerprint(g))
    calibration_groups = set(groups[:len(groups)//2])
    cal = [r for r in val if r["group"] in calibration_groups]
    policy = [r for r in val if r["group"] not in calibration_groups]
    if min(len(cal), len(policy)) < 15:
        raise ValueError("Insufficient validation data for disjoint calibration/policy sets")
    y = lambda rows: np.array([labels.index(r["label"]) for r in rows])
    pred = lambda rows: np.array([labels.index(r["prediction"]) for r in rows])
    p = lambda rows: np.array([r["probabilities"] for r in rows])
    temperature = fit_temperature(p(cal), y(cal))
    fitted_test, fitted_policy = rescale(p(test), temperature), rescale(p(policy), temperature)
    selection = select_threshold(fitted_policy, y(policy), pred(policy))
    result = classification([r["label"] for r in test], [r["prediction"] for r in test], labels,
                            [r["group"] for r in test])
    result.update({"raw_calibration": calibration(y(test), p(test), pred(test)),
        "fitted_calibration": calibration(y(test), fitted_test, pred(test)),
        "temperature": temperature, "calibration_n": len(cal), "policy_validation_n": len(policy),
        "policy_selection": selection,
        "policy_test": selective(fitted_test, y(test), pred(test), selection["threshold"] if selection else None),
        "fixed_thresholds_raw": [selective(p(test), y(test), pred(test), t) for t in [.5,.7,.8,.9,.95,.99]],
        "fixed_thresholds_fitted": [selective(fitted_test, y(test), pred(test), t) for t in [.5,.7,.8,.9,.95,.99]],
        "audit": {"truncated_inputs": sum(r["audit"]["state_truncated"] for r in test),
                  "options_truncated": test[0]["audit"]["options_truncated"],
                  "instruction_truncated": test[0]["audit"]["instruction_truncated"]},
        "timing": {"batch_size": batch_size, "batch_p50_ms": float(np.median([x["seconds"] for x in times])*1000),
                   "batch_p95_ms": float(np.quantile([x["seconds"] for x in times], .95)*1000),
                   "samples_per_second": len(test)/sum(x["seconds"] for x in times),
                   "measured_batches": len(times)}})
    # Real batch-one latency: never label amortized batch timing as user latency.
    latencies = []
    for row in test[:50]:
        _, duration = adapter.predict([row["text"]], q)
        latencies.append(duration*1000)
    result["timing"].update({"single_p50_ms": float(np.median(latencies)),
                             "single_p95_ms": float(np.quantile(latencies, .95)), "single_n": len(latencies)})
    # Prespecified sensitivity audit; cannot replace the primary scores.
    reversed_q = {"decision": {**q["decision"], "criteria": dict(reversed(list(q["decision"]["criteria"].items())))}}
    changed, correct = 0, 0
    subset = test[:100]
    for chunk in chunks(subset, batch_size):
        answers, _ = adapter.predict([r["text"] for r in chunk], reversed_q)
        for row, answer in zip(chunk, answers):
            predicted, prob = decode(answer["answers"]["decision"], labels)
            changed += predicted != row["prediction"]
            correct += predicted == row["label"]
            sink({**row, "split": "diagnostic", "suite": name+"_reverse_options", "prediction": predicted, "probabilities": prob,
                  "sdk_answer": answer["answers"]["decision"], "audit": adapter.audit(row["text"], reversed_q)["decision"]})
    result["option_order_audit"] = {"n": len(subset), "changed_predictions": changed,
        "original_accuracy": sum(r["label"] == r["prediction"] for r in subset)/len(subset),
        "reversed_accuracy": correct/len(subset)}
    return result, test

def baselines(suite):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC
    train, test = suite["splits"]["train"], suite["splits"]["test"]
    target = [r["label"] for r in test]
    labels = suite["labels"]
    groups = [r["group"] for r in test]
    majority = Counter(r["label"] for r in train).most_common(1)[0][0]
    model = Pipeline([("features", FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=40000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, sublinear_tf=True, max_features=60000))])),
        ("classifier", LinearSVC(C=1., random_state=20260926, max_iter=5000))])
    started = time.perf_counter()
    model.fit([r["text"] for r in train], [r["label"] for r in train])
    training_seconds = time.perf_counter()-started
    pred = model.predict([r["text"] for r in test]).tolist()
    return {"majority": classification(target, [majority]*len(test), labels, groups),
            "tfidf_linear_svm": {**classification(target, pred, labels, groups),
                                  "train_n": len(train), "training_seconds": training_seconds,
                                  "note": "Supervised comparator trained only on deduplicated official training split. Laya is zero-shot."},
            "uniform_random_expected_accuracy": 1/len(labels)}, pred

def diagnostics(adapter, sink):
    from .challenges import cases, minimal_pairs
    all_results = defaultdict(list)
    for task, source in [("support", list(cases())), ("minimal_pairs", list(minimal_pairs()))]:
        for prompt in ["en", "nb"]:
            for row in source:
                if prompt == "nb" and row["language"] == "en":
                    continue
                all_q = support_questions(prompt == "nb")
                q = {key: all_q[key] for key in row["expected"]}
                outputs, duration = adapter.predict([row["text"]], q)
                audits = adapter.audit(row["text"], q)
                for key, expected in row["expected"].items():
                    ans = outputs[0]["answers"][key]
                    labels = ["0","1"] if key == "refund" else ["0","1","2"] if key == "urgency" else list(q[key]["criteria"])
                    prediction, p = decode(ans, labels)
                    result = {**row, "id": row["id"]+"-"+key, "label": expected, "prediction": prediction,
                              "model": adapter.name, "suite": f"{task}_{row['language']}_prompt_{prompt}_{key}",
                              "split": "diagnostic", "probabilities": p, "labels": labels,
                              "audit": audits[key], "sdk_answer": ans}
                    all_results[result["suite"]].append(result)
                    sink(result)
    # Long input: move the same request from the beginning to the end of neutral filler.
    q = {"route": support_questions()["route"]}
    for language, filler, request in [
        ("en", "This is archived background information. ", "Please refund the duplicate payment."),
        ("nb", "Dette er arkivert bakgrunnsinformasjon. ", "Vennligst refunder den doble betalingen."),
        ("nn", "Dette er arkivert bakgrunnsinformasjon. ", "Ver venleg og refunder den doble betalinga.")]:
        for repeats in [0, 40, 160]:
            for position in ["start", "end"]:
                text = request+" "+filler*repeats if position == "start" else filler*repeats+request
                variants = [("default", {})]
                if adapter.name == "laya-multilingual":
                    variants.append(("2048", {"max_len": 2048}))
                for budget, kwargs in variants:
                    output, seconds = adapter.predict([text], q, **kwargs)
                    pred, p = decode(output[0]["answers"]["route"], list(q["route"]["criteria"]))
                    name = f"long_input_{language}_{budget}"
                    result = {"id": f"{language}-{repeats}-{position}-{budget}", "group": language,
                              "text": text, "label": "billing", "prediction": pred, "probabilities": p,
                              "labels": list(q["route"]["criteria"]), "model": adapter.name, "suite": name,
                              "split": "diagnostic", "audit": adapter.audit(text,q,**kwargs)["route"],
                              "position": position, "filler_repeats": repeats, "latency_ms": seconds*1000}
                    sink(result)
                    all_results[name].append(result)
    summary = {}
    for name, rows in all_results.items():
        summary[name] = classification([r["label"] for r in rows], [r["prediction"] for r in rows], rows[0]["labels"])
        summary[name]["truncated_inputs"] = sum(r["audit"]["state_truncated"] for r in rows)
        if "urgency" in name:
            summary[name]["score_mae"] = float(np.mean([abs(r["sdk_answer"]["score"]-int(r["label"])) for r in rows]))
        if "long_input" in name:
            summary[name]["cases"] = [{k:r[k] for k in ["id","position","filler_repeats","prediction","label","audit","latency_ms"]} for r in rows]
    return summary

def run(args):
    import torch
    import psutil
    data = read_json(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"predictions.jsonl").exists() or (output/"summary.json").exists():
        raise FileExistsError("Choose a new --output directory: existing evidence will not be overwritten.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the GPU wheel or explicitly select --device cpu.")
    random.seed(data["seed"])
    np.random.seed(data["seed"])
    torch.manual_seed(data["seed"])
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    summary = {"schema": 1, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args), "data_sha256": digest(args.data), "seed": data["seed"],
        "sources": data["sources"], "data_audit": data["audit"], "data_files": data["files"],
        "dataset_counts": {k:v["counts"] for k,v in data["suites"].items()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": psutil.cpu_count(),
            "ram_gb": psutil.virtual_memory().total/1e9, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "threads": args.threads, "packages": {d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}},
        "source_hashes": {p.name:digest(p) for p in (ROOT/"laya_bench").glob("*.py")},
        "models": {}, "baselines": {}, "comparisons": {}}
    errors = []
    def save():
        write_json(output/"summary.json", summary)
    with (output/"predictions.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
        def sink(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            if row["prediction"] != row["label"]:
                errors.append({k:row.get(k) for k in ["model","suite","split","id","text","label","prediction"]})
        try:
            for name, suite in data["suites"].items():
                print(f"Training simple baseline: {name}", flush=True)
                summary["baselines"][name], predictions = baselines(suite)
                for row, pred in zip(suite["splits"]["test"], predictions):
                    sink({**row, "prediction": pred, "model": "tfidf_linear_svm", "suite": name, "split": "test"})
            save()
            test_records = {}
            for model in args.models:
                print(f"Loading {model}", flush=True)
                adapter = LayaAdapter(model, args.device, args.offline)
                summary["models"][model] = {"metadata": adapter.metadata, "suites": {}}
                test_records[model] = {}
                try:
                    for name, suite in data["suites"].items():
                        result, records = evaluate_public(adapter,name,suite,args.batch_size,sink)
                        summary["models"][model]["suites"][name] = result
                        test_records[model][name] = records
                        save()
                    print(f"Running Norwegian and business diagnostics: {model}", flush=True)
                    summary["models"][model]["diagnostics"] = diagnostics(adapter,sink)
                    if args.device == "cuda":
                        summary["models"][model]["metadata"]["peak_allocated_gpu_gb"] = torch.cuda.max_memory_allocated()/1e9
                    save()
                finally:
                    adapter.close()
            if len(args.models) == 2:
                for name in data["suites"]:
                    summary["comparisons"][name] = paired_comparison(test_records["laya"][name],test_records["laya-multilingual"][name])
            summary["language_gaps"] = {model: paired_comparison(suites["massive_en"],suites["massive_nb"])
                                         for model,suites in test_records.items()}
            summary["status"] = "complete"
        except Exception as exc:
            summary["status"] = "failed"
            summary["failure"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save()
            with (output/"errors.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle,fieldnames=["model","suite","split","id","text","label","prediction"])
                writer.writeheader()
                # Protect spreadsheet readers from formula injection in untrusted source text.
                for row in errors:
                    writer.writerow({k:("'"+v if isinstance(v,str) and v[:1] in "=+-@" else v) for k,v in row.items()})
    from .report import generate
    generate(output/"summary.json")
    print(f"Completed: {output.resolve() / 'report.html'}", flush=True)
