"""Independent evidence audit; no model calls and no fitted metrics are changed."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .common import digest, read_json, write_json

def verify(path):
    path = Path(path)
    summary = read_json(path)
    if summary["status"] != "complete":
        raise ValueError("Only a completed run can pass the evidence audit")
    data_path = Path(summary["arguments"]["data"])
    if digest(data_path) != summary["data_sha256"]:
        raise ValueError("Prepared evaluation data changed after inference")
    data = read_json(data_path)
    records = defaultdict(list)
    seen = set()
    record_path = path.parent/"predictions.jsonl"
    for line in record_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["model"],row["suite"],row["split"],row["id"])
        if key in seen:
            raise ValueError(f"Duplicate prediction: {key}")
        seen.add(key)
        records[key[:3]].append(row)
    groups_checked = 0
    primary_count = 0
    for model, detail in summary["models"].items():
        if set(detail["suites"]) != set(data["suites"]):
            raise ValueError(f"Missing suite for {model}")
        for name, result in detail["suites"].items():
            for split in ["validation", "test"]:
                actual = {r["id"]:r for r in records[(model,name,split)]}
                expected = {r["id"]:r for r in data["suites"][name]["splits"][split]}
                if actual.keys() != expected.keys():
                    raise ValueError(f"Missing or extra scored rows: {model}/{name}/{split}")
                for id, r in actual.items():
                    if any(r[k] != expected[id][k] for k in ["text","label","group"]):
                        raise ValueError("Prediction records differ from frozen test data")
                    if not np.isfinite(r["probabilities"]).all() or not np.isclose(sum(r["probabilities"]),1):
                        raise ValueError("Invalid saved probability distribution")
            rows = records[(model,name,"test")]
            gold = [r["label"] for r in rows]
            predicted = [r["prediction"] for r in rows]
            if result["n"] != len(rows) or result["correct"] != sum(a == b for a,b in zip(gold,predicted)):
                raise ValueError("Summary count mismatch")
            if not np.isclose(result["accuracy"],accuracy_score(gold,predicted)):
                raise ValueError("Accuracy mismatch")
            if not np.isclose(result["macro_f1"],f1_score(gold,predicted,labels=result["labels"],average="macro",zero_division=0)):
                raise ValueError("Macro F1 mismatch")
            if result["confusion_matrix"] != confusion_matrix(gold,predicted,labels=result["labels"]).tolist():
                raise ValueError("Confusion matrix mismatch")
            if result["audit"]["truncated_inputs"] != sum(r["audit"]["state_truncated"] for r in rows):
                raise ValueError("Truncation mismatch")
            primary_count += len(rows)
            groups_checked += 1
        for name, result in detail["diagnostics"].items():
            rows = records[(model,name,"diagnostic")]
            if len(rows) != result["n"] or not np.isclose(result["accuracy"],sum(r["label"] == r["prediction"] for r in rows)/len(rows)):
                raise ValueError(f"Diagnostic score mismatch: {model}/{name}")
    report = {"status":"passed", "primary_model_test_decisions":primary_count,
              "all_prediction_records":len(seen), "model_suite_groups_checked":groups_checked,
              "summary_sha256":digest(path),"predictions_sha256":digest(record_path),
              "prepared_data_sha256":digest(data_path),
              "checks":["complete status","frozen data hash","unique prediction IDs","all validation/test rows present",
                        "gold labels, groups and inputs unchanged","valid probabilities","independently recomputed accuracy",
                        "independently recomputed macro F1 and confusion matrices","truncation counts","diagnostic accuracy"]}
    write_json(path.parent/"verification.json",report)
    print(json.dumps(report,indent=2))
    return report

