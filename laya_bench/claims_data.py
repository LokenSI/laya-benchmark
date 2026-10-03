"""Pinned, local reconstruction of selected publisher benchmark fixtures.

No downloaded Python is executed. Dataset text is retained only in the ignored cache.
"""
import csv
import gzip
import json
import random
import urllib.request

from .common import ROOT, digest, write_json

SOURCES = {
    "news": ("fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4", "data/test-00000-of-00001.parquet"),
    "emotion": ("dair-ai/emotion", "cab853a1dbdf4c42c2b3ef2173804746df8825fe", "split/test-00000-of-00001.parquet"),
    "spam": ("SetFit/enron_spam", "1916f66c89d52221ae33eb57d44498b4f3a5df22", "test.jsonl"),
    "phishing": ("zefang-liu/phishing-email-dataset", "34085a032c123ca237f314a01a67909cdea35e34", "Phishing_Email.csv"),
    "toxic": ("lmsys/toxic-chat", "29df8e4dba60e1f4af4b4075c0705c5b313548a8", "data/0124/toxic-chat_annotation_test.csv"),
    "intent": ("mteb/amazon_massive_intent", "940fd47a81eaa7f2cc7b129674d945d618ac38c2", "test/en.json.gz"),
    "scenario": ("mteb/amazon_massive_scenario", "58871793b91addb7c5f7afff26ccf08737fb6697", "test/en.json.gz"),
}


def download():
    from huggingface_hub import hf_hub_download
    csv.field_size_limit(256 * 1024 * 1024)
    provenance, tables = {}, {}
    for key, (repo, revision, filename) in SOURCES.items():
        path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
        provenance[key] = {"repo": repo, "revision": revision, "file": filename, "sha256": digest(path)}
        if filename.endswith("parquet"):
            import pyarrow.parquet as pq
            rows = pq.read_table(path).to_pylist()
        elif filename.endswith(".csv"):
            with open(path, encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        else:
            opener = gzip.open if filename.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
        tables[key] = rows
        print(key, len(rows), list(rows[0]), flush=True)
    return tables, provenance


def build():
    import laya
    tables, provenance = download()
    # Freeze the exact publisher source revision alongside the downloaded data.
    headers = {"User-Agent": "laya-local-audit"}
    def fetch(url):
        return urllib.request.urlopen(urllib.request.Request(url, headers=headers)).read()
    commit = json.loads(fetch("https://api.github.com/repos/NandhaKishorM/laya/commits/main"))["sha"]
    root = ROOT / ".cache" / "claim_sources"
    root.mkdir(parents=True, exist_ok=True)
    upstream = {}
    for name in ["BENCHMARKS.md", "research/results/app_benchmark_results.json", "research/scripts/bench_apps.py"]:
        url = f"https://raw.githubusercontent.com/NandhaKishorM/laya/{commit}/{name}"
        path = root / name.split("/")[-1]
        path.write_bytes(fetch(url))
        upstream[name] = {"url": url, "sha256": digest(path)}
    published = json.loads((root / "app_benchmark_results.json").read_text())["suites"]
    suites = {}
    def register(name, rows, states, gold, question, *, published_key=None, training=None, phase="replication"):
        labels = list(question.get("criteria", {})) if question["type"] == "choice" else ["false", "true"]
        suites[name] = {"phase": phase, "publisher_task_in_training": training,
            "published": {m: published[published_key][m]["accuracy"] for m in ["english", "multilingual"]} if published_key else None,
            "cases": [{"id": str(i), "state": st, "gold": labels[int(g)], "question": question} for i, st, g in zip(rows, states, gold)],
            "labels": labels}
    qnews = {"type": "choice", "instructions": "What is the topic of `article`?", "criteria": {
        "world": "world news and international politics", "sports": "sports", "business": "business and economy", "sci_tech": "science and technology"}}
    qemo = {"type": "choice", "instructions": "Which emotion is most strongly expressed in `text`?", "criteria": dict.fromkeys(["sadness", "joy", "love", "anger", "fear", "surprise"])}
    qspam = {"type": "noul", "instructions": "Is this email unsolicited spam or bulk marketing?"}
    for key, q, pub, training, make in [
        ("news", qnews, "jev.ag_news", True, lambda r: {"article": r["text"]}),
        ("emotion", qemo, "jev.emotion", False, lambda r: {"text": r["text"]}),
        ("spam", qspam, "app.email_spam", True, lambda r: laya.email_state(r.get("subject") or "", (r.get("message") or "")[:3000]))]:
        for phase, lo, hi in [("replication", 0, 400), ("extension", 400, 1400)]:
            rows = tables[key][lo:hi]
            register(f"{key}.{phase}", range(lo, lo+len(rows)), [make(r) for r in rows], [r["label"] for r in rows], q,
                     published_key=pub if phase == "replication" else None, training=training, phase=phase)
    rng = random.Random(13)
    rows = [(i, r) for i, r in enumerate(tables["phishing"][:6000]) if (r.get("Email Text") or "").strip() and r.get("Email Type") in ("Safe Email", "Phishing Email")]
    rng.shuffle(rows)
    qphish = {"type": "noul", "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?", "criteria": {"true": "phishing, scam, or fraud", "false": "a legitimate email (even if promotional)"}}
    for phase, lo, hi in [("replication", 0, 400), ("extension", 400, 1400)]:
        selected = rows[lo:hi]
        register(f"phishing.{phase}", [i for i,r in selected], [{"email": r["Email Text"][:3000]} for i,r in selected],
                 [int(r["Email Type"] == "Phishing Email") for i,r in selected], qphish,
                 published_key="app.phishing" if phase == "replication" else None, training=True, phase=phase)
    toxic = [(i,r) for i,r in enumerate(tables["toxic"]) if (r.get("user_input") or "").strip()]
    for key, field, instruction, pub in [
        ("jailbreaking", "prompt", "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?", "app.guardrails_jailbreak"),
        ("toxicity", "post", "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?", "app.moderation_toxicity")]:
        pos = [(i,r) for i,r in toxic if int(r[key]) == 1][:200]
        neg = [(i,r) for i,r in toxic if int(r[key]) == 0][:400-len(pos)]
        selected = pos + neg
        rng.shuffle(selected)
        register(key, [i for i,r in selected], [{field:r["user_input"][:3000]} for i,r in selected], [int(r[key]) for i,r in selected],
                 {"type":"noul", "instructions":instruction}, published_key=pub, training=False)
    for key, instruction, expected in [
        ("intent", "What is the user asking for in `utterance`?", {"english":235/300,"multilingual":197/300}),
        ("scenario", "Which domain does `utterance` belong to?", {"english":181/300,"multilingual":168/300})]:
        all_labels = sorted({r["label_text"] for r in tables[key]})
        rng = random.Random(13)
        cases=[]
        for i,r in enumerate(tables[key][:300]):
            gold=r["label_text"]
            keys=[gold]+rng.sample([v for v in all_labels if v!=gold],min(19,len(all_labels)-1))
            rng.shuffle(keys)
            cases.append({"id":str(i),"state":{"utterance":r["text"]},"gold":gold,"question":{
                "type":"choice","instructions":instruction,"criteria":{v:v.replace("_"," ").replace(".",": ") for v in keys}}})
        suites[key]={"phase":"replication","published":expected,"labels":all_labels,"cases":cases,"publisher_task_in_training":True,
                     "source":"research commit 28d43add7e47ce502489c9433310d55276c64e0f; build_benchmark_nb.py; T4 result"}
    manifest={"datasets":provenance,"publisher_commit":commit,"publisher_files":upstream,"publisher_application_environment":{"sdk":"0.2.1","device":"cpu"},
              "comparison_tolerance_percentage_points":2,"note":"Tolerance is an engineering repeatability check, not a significance test. Same sampling/questions, newer SDK and different device. Published dataset revisions were not pinned; our pinned files are current snapshots. No independent training-overlap audit."}
    write_json(ROOT/"data/prepared/claims.json", {"manifest":manifest,"suites":suites})
    write_json(ROOT/"results/claims/provenance.json",manifest)
    print({k:len(v["cases"]) for k,v in suites.items()},flush=True)

if __name__ == "__main__":
    build()
