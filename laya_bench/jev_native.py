"""Explain the Jev gap: rerun the identical 300 Jev cases with Laya-native question formats.

Jev format (jev_compare.py): opaque keys `label_000` + BTZSC hypothesis sentences, one generic
instruction. Native format: short natural label names, task instruction, as in the publisher's
own bench_apps.py. For banking77 we also test the SDK-recommended embedding shortlist (top-k
options by bi-encoder similarity, then one Laya call). No example is tuned on; all variants are
declared here before running.
"""
import gc
import json
import re
import time

from .common import ROOT, digest, read_json, write_json

EMBEDDER = "BAAI/bge-small-en-v1.5"
INSTRUCTIONS = {"agnews": "What is the topic of `text`?",
                "emotiondair": "Which emotion is most strongly expressed in `text`?",
                "banking77": "Which banking intent does `text` express?"}


def short_name(hypothesis):
    name = re.sub(r"^This (banking )?customer example message ", "", hypothesis)
    name = re.sub(r"^This example (news text|tweet) ", "", name)
    name = re.sub(r"^(is about|is related to|requests|questions|expresses the emotion:)\s*", "", name)
    name = re.sub(r"^(a|an|the) ", "", name)
    name = re.sub(r"( news)$", "", name.rstrip(". "))
    return name.strip()


class Embedder:
    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
        path = snapshot_download(EMBEDDER, allow_patterns=["*.json", "*.txt", "model.safetensors"])
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).cuda().eval()
        self.torch = torch
        self.revision = path.replace("\\", "/").rstrip("/").split("/")[-1]

    def __call__(self, texts):
        with self.torch.inference_mode():
            enc = self.tok(list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            v = self.model(**enc).last_hidden_state[:, 0]
            return self.torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy()


def run():
    import numpy as np
    import torch
    from .adapter import LayaAdapter
    from .metrics import classification, wilson
    torch.manual_seed(20260926)
    out = ROOT / "results/jev_native"
    out.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data/prepared/jev_manifest.jsonl"
    provenance = read_json(ROOT / "results/jev/provenance.json")
    assert digest(path) == provenance["published_manifest_sha256"]
    cases = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    embed = Embedder()
    summary = {"manifest_sha256": digest(path), "embedder": {"repo": EMBEDDER, "revision": embed.revision}, "results": {}, "label_names": {},
               "published": provenance["published"]["results"]}
    # Embedding-only baseline and shortlist recall are model-independent.
    ranks = {}
    for dataset in ["agnews", "emotiondair", "banking77"]:
        subset = [c for c in cases if c["dataset"] == dataset]
        names = [short_name(h) for h in subset[0]["labels"]]
        assert len(set(names)) == len(names), names
        summary["label_names"][dataset] = names
        lab = embed(names)
        q = embed([c["text"] for c in subset])
        ranks[dataset] = np.argsort(-(q @ lab.T), axis=1, kind="mergesort")
        top1 = [int(r[0]) == c["target_index"] for r, c in zip(ranks[dataset], subset)]
        summary["results"][f"embedding-only/{dataset}"] = {"n": len(subset), "accuracy": float(np.mean(top1))}
        for k in (10, 20):
            summary["results"][f"embedding-only/{dataset}"][f"recall_at_{k}"] = float(np.mean(
                [c["target_index"] in r[:k] for r, c in zip(ranks[dataset], subset)]))
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for model in ["laya", "laya-multilingual"]:
            adapter = LayaAdapter(model, offline=True)
            for dataset in ["agnews", "emotiondair", "banking77"]:
                subset = [c for c in cases if c["dataset"] == dataset]
                names = summary["label_names"][dataset]
                variants = [("native", None)]
                if dataset == "banking77":
                    variants += [("native_shortlist20", 20), ("native_shortlist10", 10)]
                for variant, k in variants:
                    gold, pred, truncated = [], [], 0
                    for i, case in enumerate(subset):
                        keep = list(range(len(names))) if k is None else sorted(int(x) for x in ranks[dataset][i][:k])
                        options = [names[j] for j in keep]
                        q = {"label": {"type": "choice", "instructions": INSTRUCTIONS[dataset],
                                       "criteria": {o: None for o in options}}}
                        state = {"text": case["text"]}
                        answers, elapsed = adapter.predict([state], q)
                        choice = answers[0]["answers"]["label"]["choice"]
                        from laya.common import serialize_state
                        truncated += bool(adapter.audit(serialize_state(state), q)["label"]["options_truncated"])
                        gold.append(names[case["target_index"]])
                        pred.append(choice)
                        stream.write(json.dumps({"model": model, "dataset": dataset, "variant": variant, "id": case["example_id"],
                                                 "gold": gold[-1], "pred": choice, "options": options, "seconds": elapsed}) + "\n")
                    measured = classification(gold, pred, names)
                    correct = sum(a == b for a, b in zip(gold, pred))
                    measured.update({"correct": correct, "wilson95": wilson(correct, len(gold)),
                                     "options_truncated_cases": truncated, "shortlist_k": k})
                    summary["results"][f"{model}/{dataset}/{variant}"] = measured
                    write_json(out / "summary.json", summary)
                    print(model, dataset, variant, f"{measured['accuracy']:.1%}", "trunc", truncated, flush=True)
            adapter.close()
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
    summary["predictions_sha256"] = digest(out / "predictions.jsonl")
    write_json(out / "summary.json", summary)


if __name__ == "__main__":
    run()
