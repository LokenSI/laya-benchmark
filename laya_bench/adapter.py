import time
from collections import Counter

from .common import ROOT, digest, read_json
from .metrics import probabilities

class LayaAdapter:
    def __init__(self, name, device="cuda", offline=False):
        import torch
        import laya
        from huggingface_hub import snapshot_download
        source = read_json(ROOT / "sources.json")["models"][name]
        path = snapshot_download(source["repo"], revision=source["revision"], local_files_only=offline,
                allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
        start = time.perf_counter()
        self.agent = laya.load(path, device=device, fast=False)
        self.load_seconds = time.perf_counter()-start
        self.name = name
        self.requested_device = device
        self.check_device()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        self.metadata = {**source, "load_seconds_excluding_download": self.load_seconds,
            "device": str(self.agent.device), "amp_dtype": str(self.agent.dtype),
            "weight_dtype": str(next(self.agent.model.parameters()).dtype), "config": self.agent.cfg,
            "runtime_temperature": self.agent.temperature,
            "runtime_temperature_by_options": self.agent.temperature_by_options,
            "weights_sha256": digest(f"{path}/model.safetensors")}

    def check_device(self):
        if str(self.agent.device).split(":")[0] != self.requested_device.split(":")[0]:
            raise RuntimeError(f"Laya changed device to {self.agent.device}. Rerun explicitly on CPU or with a smaller batch; mixed-device timing is invalid.")

    def synchronize(self):
        import torch
        if self.agent.device.type == "cuda":
            torch.cuda.synchronize()

    def audit(self, text, questions, max_len=None, head_max_len=None):
        from laya.common import build_sequence, render_options
        agent = self.agent
        maximum = max_len or agent.cfg["max_len"]
        head = head_max_len or agent.cfg["head_max_len"]
        state_tokens = len(agent.tok(text.replace(agent.tok.mask_token, " "), add_special_tokens=False)["input_ids"])
        result = {}
        for name, question in questions.items():
            internal = agent._to_internal(question)
            empty, markers = build_sequence(agent.tok, "", internal, maximum, head)
            room = maximum - len(empty)
            option_lengths = [len(agent.tok(" " + o, add_special_tokens=False)["input_ids"]) for o in render_options(internal)]
            actual_lengths = [((markers[i+1] if i+1 < len(markers) else len(empty)-2)-m-1) for i,m in enumerate(markers)]
            ins_length = len(agent.tok(f'{internal["t"]} question: {internal["ins"]}', add_special_tokens=False)["input_ids"])
            result[name] = {"state_tokens": state_tokens, "state_budget": room,
                "state_truncated": state_tokens > room, "max_len": maximum, "head_max_len": head,
                "options_truncated": sum(a < b for a,b in zip(actual_lengths, option_lengths)),
                "instruction_truncated": bool(markers and markers[0]-2 < ins_length)}
        return result

    def predict(self, texts, questions, **kwargs):
        self.synchronize()
        started = time.perf_counter()
        results = self.agent.predict_batch(texts, questions, batch_size=len(texts), **kwargs)
        self.synchronize()
        elapsed = time.perf_counter()-started
        self.check_device()
        if len(results) != len(texts):
            raise ValueError("SDK returned the wrong number of results")
        return results, elapsed

    def close(self):
        self.agent.__exit__(None, None, None)

def decode(answer, labels):
    kind = answer["type"]
    if kind == "noul":
        p = [1-answer["noul"], answer["noul"]]
        pred = labels[int(answer["noul"] >= .5)]
    else:
        p = [answer["probabilities"][label] for label in labels]
        # Preserve SDK choice when four-decimal rounding creates ties.
        pred = answer["choice"] if kind == "choice" else labels[max(range(len(p)), key=p.__getitem__)]
    if pred not in labels:
        raise ValueError(f"Unknown model output: {pred}")
    return pred, probabilities([p])[0].tolist()
