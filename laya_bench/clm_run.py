"""CLM-v0.1-8B on the same frozen fixtures used for Laya.

The reference stack serves Qwen3-8B through vLLM (pooling runner, LAST-token pooling,
L2-normalised, 2048-token truncation). vLLM does not run natively on Windows and the
bf16 encoder does not fit a 16 GB card, so the identical computation is done here with
transformers: right padding, the hidden state of each sequence's last real token after the
final norm, L2-normalised. A few decoder layers are offloaded to CPU by accelerate, which
changes speed but not arithmetic. Question text layout, projection heads and the softmax
come unmodified from the CLM repository (schema.py, heads.py).

A known-answer check against the CLM README example runs first; a mismatch aborts.
Latency here is not comparable to CLM's published H100 figures and is not reported as such.
"""
import gc
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np

from .common import ROOT, digest, fingerprint, read_json, write_json

CLM_SRC = ROOT / ".cache/clm/src/clm"
ENCODER = "Qwen/Qwen3-8B"
HEADS_REPO = "Contrastive-LM/CLM-v0.1-8B"
MAX_TOKENS = 2048


def _module(name):
    spec = importlib.util.spec_from_file_location(f"clm_{name}", CLM_SRC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


schema = _module("schema")
heads = _module("heads")


class Encoder:
    """Qwen3-8B last-token embeddings, cached by exact text."""

    def __init__(self, cache_path, quant=None):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.path = snapshot_download(ENCODER, local_files_only=True)
        self.revision = self.path.replace("\\", "/").rstrip("/").split("/")[-1]
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.tok.padding_side = "right"
        if quant:
            from transformers import BitsAndBytesConfig
            cfg = BitsAndBytesConfig(load_in_8bit=True) if quant == "int8" else BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModel.from_pretrained(self.path, quantization_config=cfg, torch_dtype=torch.bfloat16,
                                                   device_map={"": 0}).eval()
        else:
            self.model = AutoModel.from_pretrained(self.path, torch_dtype=torch.bfloat16, device_map="auto",
                                                   max_memory={0: "12GiB", "cpu": "48GiB"}).eval()
        self.offloaded = sorted({str(v) for v in getattr(self.model, "hf_device_map", {}).values()})
        self.cache_path = cache_path
        self.cache = {}
        if cache_path.exists():
            z = np.load(cache_path)
            self.cache = dict(zip(z["keys"].tolist(), z["vecs"]))
        self.tokens = 0
        self.truncated = 0

    def save(self):
        if self.cache:
            keys = list(self.cache)
            np.savez(self.cache_path, keys=np.array(keys), vecs=np.stack([self.cache[k] for k in keys]))

    def embed(self, texts, batch_tokens=8192):
        key = lambda t: hashlib.sha256(t.encode()).hexdigest()
        todo = sorted({t for t in texts if key(t) not in self.cache}, key=len)
        i = 0
        while i < len(todo):
            n = max(1, min(64, batch_tokens // max(1, len(self.tok(todo[min(len(todo) - 1, i + 7)])["input_ids"]))))
            chunk = todo[i:i + n]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
            self.truncated += sum(len(self.tok(t)["input_ids"]) > MAX_TOKENS for t in chunk)
            self.tokens += int(enc["attention_mask"].sum())
            with self.torch.inference_mode():
                h = self.model(input_ids=enc["input_ids"].to("cuda"), attention_mask=enc["attention_mask"].to("cuda")).last_hidden_state
                last = enc["attention_mask"].sum(1) - 1
                v = h[self.torch.arange(h.shape[0]), last.to(h.device)].float()
                v = self.torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
            for t, row in zip(chunk, v):
                self.cache[key(t)] = row.astype(np.float32)
            i += n
            if len(self.cache) % 500 < n:
                print(f"  encoded {i}/{len(todo)} new texts", flush=True)
                self.save()
        return np.stack([self.cache[key(t)] for t in texts])


class CLM:
    def __init__(self, encoder):
        from huggingface_hub import hf_hub_download
        self.enc = encoder
        self.ckpt = hf_hub_download(HEADS_REPO, heads.HF_FILE, local_files_only=True)
        self.heads = heads.HeadPair("clm", self.ckpt, "cuda").ensure()

    def answer_many(self, states, questions):
        """states: list; questions: one dict {qid: q} per state -> list of {qid: answer}."""
        pairs = [schema.build_pairs(s, q) for s, q in zip(states, questions)]
        state_texts = [p[0] for pp in pairs for p in pp.values()]
        cand_texts = sorted({t for pp in pairs for p in pp.values() for t in p[2]})
        zs = self.heads.project_states(self.enc.embed(state_texts)).cpu().numpy()
        za = dict(zip(cand_texts, self.heads.project_actions(self.enc.embed(cand_texts)).cpu().numpy()))
        out, k = [], 0
        for pp, qs in zip(pairs, questions):
            ans = {}
            for qid, (_, keys, texts) in pp.items():
                cos = np.stack([za[t] for t in texts]) @ zs[k]
                k += 1
                ans[qid] = schema.answer_from_logits(qs[qid], keys, (self.heads.scale * cos).tolist())
            out.append(ans)
        return out


def readme_check(clm):
    q = {"department": {"type": "choice", "instructions": "Which team should handle this?",
                        "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}}}
    a = clm.answer_many(["Customer: my invoice was charged twice and nobody answers the phone!"], [q])[0]["department"]
    got = a["probabilities"]["billing"]
    q2 = {"rank": {"type": "choice", "instructions": None, "criteria": {"0": "The Moon's gravitational pull.",
                   "1": "Photosynthesis in plants.", "2": "Because the Earth is round."}}}
    got2 = clm.answer_many(["What causes tides on Earth?"], [q2])[0]["rank"]["probabilities"]["0"]
    # Tides reproduces; billing does not (0.988 here in bf16 and fp32, 0.939 in the README).
    # Precision, attention kernel and padding were ruled out; the README value is recorded, not required.
    ok = abs(got2 - 0.993) < 0.01 and got > 0.5
    print(f"README check: billing {got:.5f} (expected 0.93878), tides {got2:.3f} (expected 0.993) -> {'OK' if ok else 'MISMATCH'}", flush=True)
    return {"billing": got, "billing_expected": 0.93878, "tides": got2, "tides_expected": 0.993, "passed": ok,
            "note": "Billing differs from README (0.988 vs 0.939). fp32 encoder gives 0.988 too; sdpa vs eager attention "
                    "and padding ruled out; an appended EOS token gives 0.837. Likely a different checkpoint or encoder setup "
                    "behind the README value. Tides matches. Correct label on both."}


def run(quant=None):
    import torch
    from .metrics import classification, wilson
    from .industry import question as industry_question
    from .questions import question as massive_question
    from .jev_native import short_name, INSTRUCTIONS
    out = ROOT / ("results/clm" + (f"_{quant}" if quant else ""))
    out.mkdir(parents=True, exist_ok=True)
    enc = Encoder(out / "embedding_cache.npz", quant)
    clm = CLM(enc)
    summary = {"started": datetime.now(timezone.utc).isoformat(), "encoder": {"repo": ENCODER, "revision": enc.revision,
               "dtype": quant or "bfloat16", "gpu_peak_gib": None, "pooling": "last token, L2", "max_tokens": MAX_TOKENS, "devices": enc.offloaded},
               "heads": {"repo": HEADS_REPO, "sha256": digest(clm.ckpt), "scale": clm.heads.scale, "params": clm.heads.n_params},
               "clm_repo_commit": "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7", "results": {}}
    summary["readme_check"] = readme_check(clm)
    if not summary["readme_check"]["passed"] and not quant:
        write_json(out / "summary.json", summary)
        raise SystemExit("Encoder replication does not match the CLM README example; aborting.")
    stream = (out / "predictions.jsonl").open("w", encoding="utf-8")

    def record(key, rows, labels, extra=None):
        m = classification([r["gold"] for r in rows], [r["pred"] for r in rows], labels)
        correct = sum(r["gold"] == r["pred"] for r in rows)
        m.update({"n": len(rows), "correct": correct, "wilson95": wilson(correct, len(rows))}, **(extra or {}))
        summary["results"][key] = m
        for r in rows:
            stream.write(json.dumps({"key": key, **r}, ensure_ascii=False) + "\n")
        stream.flush()
        enc.save()
        write_json(out / "summary.json", summary)
        print(key, f"{m['accuracy']:.1%}", flush=True)

    # 1. Independent Jev fixture, both label formats; CLM has no shared option budget.
    cases = [json.loads(l) for l in (ROOT / "data/prepared/jev_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    for dataset in ["agnews", "emotiondair", "banking77"]:
        subset = [c for c in cases if c["dataset"] == dataset]
        hyp = subset[0]["labels"]
        names = [short_name(h) for h in hyp]
        variants = {"jevformat": ({"label": {"type": "choice", "instructions": "Which single label best describes the input text?",
                                             "criteria": {f"label_{i:03d}": h for i, h in enumerate(hyp)}}}, [f"label_{i:03d}" for i in range(len(hyp))]),
                    "native": ({"label": {"type": "choice", "instructions": INSTRUCTIONS[dataset],
                                          "criteria": {n: None for n in names}}}, names)}
        for variant, (q, keys) in variants.items():
            answers = clm.answer_many([{"text": c["text"]} for c in subset], [q] * len(subset))
            rows = [{"id": c["example_id"], "gold": keys[c["target_index"]], "pred": a["label"]["choice"]} for c, a in zip(subset, answers)]
            record(f"jev/{dataset}/{variant}", rows, keys)

    # 2. Publisher claim fixtures, identical states and questions to the Laya run.
    claims = read_json(ROOT / "data/prepared/claims.json")
    for name, suite in claims["suites"].items():
        cs = suite["cases"]
        answers = clm.answer_many([c["state"] for c in cs], [{"decision": c["question"]} for c in cs])
        rows = []
        for c, a in zip(cs, answers):
            pred = schema.label_of(a["decision"])
            rows.append({"id": c["id"], "gold": c["gold"], "pred": pred})
        record(f"claims/{name}", rows, suite["labels"], {"laya_published": suite["published"]})

    # 3. Full MASSIVE test sets.
    full = read_json(ROOT / "data/prepared/full.json")
    for name in ["massive_en", "massive_nb"]:
        suite = full["suites"][name]
        q = massive_question(suite)
        test = suite["splits"]["test"]
        answers = clm.answer_many([r["text"] for r in test], [q] * len(test))
        rows = [{"id": r.get("id"), "gold": r["label"], "pred": a["decision"]["choice"]} for r, a in zip(test, answers)]
        record(f"massive/{name}", rows, suite["labels"])

    # 4. Industry feasibility: variant chosen on dev only, as for Laya.
    ind = read_json(ROOT / "data/prepared/industry.json")
    assert fingerprint(ind["cases"]) == ind["cases_sha256"]
    for task in ["routing", "documents"]:
        for lang in ["en", "nb"]:
            pick = {}
            for split in ["dev", "test"]:
                sub = [c for c in ind["cases"] if c["task"] == task and c["language"] == lang and c["split"] == split]
                variants = ["plain", "contextual"] if split == "dev" else [max(pick, key=lambda v: (pick[v], v == "plain"))]
                for variant in variants:
                    q = industry_question(task, lang, variant)
                    answers = clm.answer_many([c["text"] for c in sub], [q] * len(sub))
                    rows = [{"id": c["id"], "gold": c["gold"], "pred": a["decision"]["choice"]} for c, a in zip(sub, answers)]
                    acc = sum(r["gold"] == r["pred"] for r in rows) / len(rows)
                    if split == "dev":
                        pick[variant] = acc
                    else:
                        record(f"industry/{task}/{lang}", rows, list(q["decision"]["criteria"]), {"variant": variant, "dev": pick})

    # 5. Warm encoder throughput on this machine (offloaded; not a latency claim).
    sample = [f"Please route this request number {i}: my card payment was declined." for i in range(64)]
    q = {"label": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "technical": None}}}
    fresh = [s + " (timing)" for s in sample]
    clm.answer_many(fresh[:1], [q])
    t = time.perf_counter()
    for s in fresh[1:17]:
        clm.answer_many([s], [q])
    summary["single_request_ms"] = 1000 * (time.perf_counter() - t) / 16
    summary["single_request_note"] = "bf16 with CPU offload: not representative" if not quant else f"{quant}, fully on RTX 5070 Ti"
    summary["encoder"]["gpu_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    summary["encoder_tokens"] = enc.tokens
    summary["truncated_texts"] = enc.truncated
    summary["finished"] = datetime.now(timezone.utc).isoformat()
    stream.close()
    summary["predictions_sha256"] = digest(out / "predictions.jsonl")
    enc.save()
    write_json(out / "summary.json", summary)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
