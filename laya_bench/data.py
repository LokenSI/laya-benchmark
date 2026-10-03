"""Pinned Parquet downloads; labels never enter model input."""
import hashlib
from collections import Counter
from pathlib import Path

from .common import ROOT, digest, read_json, write_json

SCENARIOS = ['social', 'transport', 'calendar', 'play', 'news', 'datetime',
             'recommendation', 'email', 'iot', 'general', 'qa', 'cooking',
             'takeaway', 'music', 'lists', 'alarm', 'weather', 'audio']

def normalized(text):
    return " ".join(text.casefold().split())

def sample(rows, n, seed):
    """Uniform, label-blind deterministic sample. 0 means the entire cleaned split."""
    ordered = sorted(rows, key=lambda r: hashlib.sha256(f'{seed}:{r["id"]}'.encode()).hexdigest())
    return ordered if n == 0 else ordered[:n]

def clean_splits(splits, grouped=False):
    stats = {}
    result = {}
    seen = set()
    # Test has priority: drop train/validation overlaps, not difficult test examples.
    for split in ["test", "validation", "train"]:
        kept = []
        for row in splits[split]:
            key = normalized(row["text"])
            if not key:
                raise ValueError("Empty public dataset input")
            if key not in seen:
                kept.append(row)
                seen.add(key)
        result[split] = kept
        stats[split] = {"original": len(splits[split]), "kept": len(kept),
                        "removed_exact_text_duplicates": len(splits[split]) - len(kept)}
    if grouped:
        groups = {s: {r["group"] for r in result[s]} for s in result}
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            if groups[a] & groups[b]:
                raise ValueError(f"Source document leakage: {a}/{b}")
    return result, stats

def load_parquet(source, filename):
    from huggingface_hub import hf_hub_download
    from datasets import Dataset
    path = hf_hub_download(source["repo"], filename, repo_type="dataset", revision=source["revision"])
    ds = Dataset.from_parquet(path)
    return ds, {"file": filename, "sha256": digest(path), "rows": len(ds)}

def prepare(output, test_size=1000, validation_size=400, seed=20260926):
    sources = read_json(ROOT / "sources.json")
    suites, files, audit = {}, {}, {}
    for locale, language in [("en-US", "en"), ("nb-NO", "nb")]:
        name = f"massive_{language}"
        splits = {}
        files[name] = []
        for split in ["train", "validation", "test"]:
            ds, info = load_parquet(sources["datasets"]["massive"], f"{locale}/{split}/0000.parquet")
            files[name].append(info)
            feature = ds.features["scenario"]
            labels = list(feature.names) if hasattr(feature, "names") else sorted(set(ds["scenario"]))
            if set(labels) != set(SCENARIOS):
                raise ValueError(f"Unexpected MASSIVE label set: {labels}")
            rows = []
            for r in ds:
                label = feature.int2str(r["scenario"]) if hasattr(feature, "int2str") else r["scenario"]
                rows.append({"id": str(r["id"]), "group": str(r["id"]), "text": r["utt"], "label": label})
            splits[split] = rows
        cleaned, stats = clean_splits(splits)
        suites[name] = {"language": language, "task": "scenario", "source": "massive",
                        "labels": sorted(SCENARIOS), "splits": cleaned}
        audit[name] = stats
    # Equal translation IDs: a language gap cannot be caused by a different sample.
    for split, cap in [("validation", validation_size), ("test", test_size)]:
        en = {r["id"]: r for r in suites["massive_en"]["splits"][split]}
        nb = {r["id"]: r for r in suites["massive_nb"]["splits"][split]}
        shared = set(en) & set(nb)
        if any(en[i]["label"] != nb[i]["label"] for i in shared):
            raise ValueError("Translated MASSIVE labels disagree")
        selected = sample([en[i] for i in shared], cap, seed)
        for name, lookup in [("massive_en", en), ("massive_nb", nb)]:
            suites[name]["splits"][split] = [lookup[r["id"]] for r in selected]
            audit[name][split]["paired_pool"] = len(shared)
    splits = {}
    files["norec_no"] = []
    for split in ["train", "validation", "test"]:
        ds, info = load_parquet(sources["datasets"]["norec"], f"ternary/{split}-00000-of-00001.parquet")
        files["norec_no"].append(info)
        labels = {0: "negative", 1: "positive", 2: "neutral"}
        splits[split] = [{"id": r["id"], "group": r["id"].split("-")[0],
                          "text": r["review"], "label": labels[r["sentiment"]]} for r in ds]
    cleaned, stats = clean_splits(splits, grouped=True)
    for split, cap in [("validation", validation_size), ("test", test_size)]:
        cleaned[split] = sample(cleaned[split], cap, seed)
    suites["norec_no"] = {"language": "no", "task": "sentiment", "source": "norec",
                          "labels": ["negative", "neutral", "positive"], "splits": cleaned}
    audit["norec_no"] = stats
    for suite in suites.values():
        for split, rows in suite["splits"].items():
            if len({r["id"] for r in rows}) != len(rows):
                raise ValueError("Duplicate row IDs")
        suite["counts"] = {s: dict(Counter(r["label"] for r in rows)) for s, rows in suite["splits"].items()}
    result = {"schema": 1, "seed": seed, "requested_test_size": test_size,
              "requested_validation_size": validation_size, "sources": sources,
              "files": files, "audit": audit, "suites": suites}
    write_json(output, result)
    print(f"Prepared {output}", flush=True)
    for name, suite in suites.items():
        print(name, {s: len(rows) for s, rows in suite["splits"].items()}, flush=True)
    return result

