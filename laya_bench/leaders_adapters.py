"""Local leader integrations. Quantized variants are named and disclosed separately."""
import atexit
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from .common import ROOT, read_json, digest
from .alternatives_adapters import typed_questions, decode_typed, answer_probabilities, text_state


def native_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_nf4(meta):
    import torch
    import transformers
    config = transformers.AutoConfig.from_pretrained(meta['path'], local_files_only=True)
    quant = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        meta['path'], local_files_only=True, dtype=torch.bfloat16, device_map='cuda',
        quantization_config=quant, attn_implementation='sdpa').eval()
    return model


class Decision4B:
    def __init__(self, meta):
        self.native = native_module('flymy_decision_pinned', Path(meta['path']) / 'model.py')
        self.model = self.native.load(assets=meta['bases'][0]['path'], device='cuda', graphs=False)
        self.metadata = {'precision': 'bf16 merged LoRA; fp32 letter projection',
                         'interface': 'publisher verified model.load and decide, eager CUDA',
                         'temperature': self.model.temperature, 'max_tokens': self.model.max_tokens,
                         'option_limit': 26, 'context_policy': 'reject overflow, no truncation',
                         'runtime_note': 'CUDA graph optimization disabled; native inference math retained'}

    def batch(self, rows):
        out = []
        for row in rows:
            questions, mapping = typed_questions(row['questions'])
            answers = {key: self.model.decide(row['state'], q)['probabilities'] for key, q in questions.items()}
            out.append(decode_typed(answers, row['questions'], mapping))
        return out


class Winnow:
    def __init__(self, meta):
        import requests
        runtime = read_json(ROOT / 'results/alternatives/winnow-runtime.json')
        executable = Path(runtime['executable'])
        assert digest(executable) == runtime['sha256']
        model = Path(meta['path']) / 'gguf/Winnow-12B-Q8_0.gguf'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        self.session = requests.Session()
        self.session.trust_env = False
        self.log = (ROOT / 'results/alternatives/logs/winnow-server.log').open('a', encoding='utf-8')
        env = dict(os.environ)
        env.update(WINNOW_CONTEXT='16384', WINNOW_HEAD='selected', WINNOW_CACHE='q8_0',
                   WINNOW_PIPELINE='optimized', WINNOW_MEMORY='exclusive', WINNOW_PARALLEL='1',
                   WINNOW_BATCH='1024', WINNOW_UBATCH='512', CUDA_VISIBLE_DEVICES='0')
        env['PATH'] = ('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/x64;'
                       'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin;' + env['PATH'])
        command = [str(executable), '--model', str(model), '--alias', 'Winnow-12B', '--ctx-size', '16384',
                   '--parallel', '1', '--n-gpu-layers', '999', '--fit', 'off', '--flash-attn', 'on',
                   '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0', '--no-context-shift',
                   '--lazy-mode', 'off', '--split-mode', 'none', '--override-tensor',
                   r'^(token_embd|per_layer_token_embd)\.weight$=CUDA0', '--batch-size', '1024',
                   '--ubatch-size', '512', '--threads', '8', '--host', '127.0.0.1', '--port', str(port),
                   '--jinja', '--reasoning', 'off', '--no-warmup', '--cache-ram', '0', '--cors-origins', '']
        self.proc = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        atexit.register(self.close)
        for _ in range(180):
            if self.proc.poll() is not None:
                raise RuntimeError('Winnow server exited; inspect winnow-server.log')
            try:
                if self.session.get(self.url + '/health', timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            self.close()
            raise RuntimeError('Winnow did not become ready')
        self.metadata = {'precision': 'publisher Q8_0 GGUF', 'interface': 'native /v1/systemone, selected head',
                         'max_tokens': 16384, 'option_limit': 64, 'temperature': 1.0,
                         'context_policy': 'native overflow rejection; no context shift',
                         'runtime': runtime, 'server_command': command,
                         'timing_warning': 'External native CUDA process; Torch peak memory excludes server allocations. Prefix reuse disabled for benchmark requests.'}

    def batch(self, rows):
        out = []
        for row in rows:
            qs, mapping = typed_questions(row['questions'])
            response = self.session.post(self.url + '/v1/systemone', json={
                'state': row['state'], 'questions': qs,
                'winnow': {'temperature': 1.0, 'reuse_prefix': False, 'diagnostics': True}}, timeout=300)
            if response.status_code in (400, 413, 422):
                raise ValueError('Winnow native input rejection: ' + response.text[:600])
            response.raise_for_status()
            data = response.json()
            result = decode_typed(answer_probabilities(data['answers']), row['questions'], mapping)
            for key, q in row['questions'].items():
                if q['type'] == 'choice':
                    result['pred'][key] = [data['answers'][key]['choice']]
            result['native_usage'] = data.get('usage')
            result['native_diagnostics'] = data.get('winnow')
            out.append(result)
        return out

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.session.close()
        if not self.log.closed:
            self.log.close()


class JevOmniNF4:
    def __init__(self, meta):
        import torch
        from transformers import AutoProcessor
        root = Path(meta['path'])
        native = native_module('jev_omni_pinned', root / 'jev_omni.py')
        model = load_nf4(meta)
        cfg = read_json(root / 'decision_config.json')
        head = native._Head256(cfg['hidden_size']).to('cuda').eval()
        head.load_state_dict(torch.load(root / 'head.pt', map_location='cuda', weights_only=True))
        _, decoder = native._find_backbone(model)
        self.model = native.JevOmni(model, head, AutoProcessor.from_pretrained(root, local_files_only=True), decoder)
        self.metadata = {'precision': 'NF4 backbone; publisher fp32 classifier head',
                         'interface': 'publisher JevOmni.predict, text only', 'option_limit': 256,
                         'comparison_scope': 'Local quantized variant; published BF16 leaderboard accuracy not directly reproduced',
                         'context_policy': 'No truncation; GPU allocation failures retained'}

    def batch(self, rows):
        out = []
        for row in rows:
            qs, mapping = typed_questions(row['questions'])
            answers = {}
            for key, q in qs.items():
                criteria = q.get('criteria') or {'false': 'no', 'true': 'yes'}
                items = list(enumerate(criteria)) if isinstance(criteria, list) else list(criteria.items())
                keys = [str(k) for k, _ in items]
                labels = [str(v) if v is not None else str(k) for k, v in items]
                if len(set(labels)) != len(labels):
                    # The publisher returns a dict keyed by description; do not
                    # silently lose repeated options or change their semantics.
                    raise ValueError('Jev-Omni native API cannot preserve duplicate option descriptions')
                answer = self.model.predict(state=text_state(row['state']), question=q['instructions'], options=labels)
                answers[key] = {k: answer['probabilities'][label] for k, label in zip(keys, labels)}
            out.append(decode_typed(answers, row['questions'], mapping))
        return out


class CygnetNF4:
    def __init__(self, meta):
        from transformers import AutoTokenizer
        root = ROOT / '.cache/leader_research/blockbrain-ai--cygnet-recipe'
        self.native = native_module('cygnet_pinned', root / 'shim/cygnet_shim.py')
        self.native.TEMPERATURE = 3.4
        self.model = load_nf4(meta)
        self.tokenizer = AutoTokenizer.from_pretrained(meta['path'], local_files_only=True)
        self.letter_tokens = {}
        # Account for distinct vocabulary IDs which decode to the same letter.
        for token, idx in self.tokenizer.get_vocab().items():
            if token.strip('▁Ġ ') in self.native.LETTERS and len(token.strip('▁Ġ ')) == 1:
                decoded = self.tokenizer.decode([idx], skip_special_tokens=False)
                if decoded in self.native.LETTERS and len(decoded) == 1:
                    self.letter_tokens[idx] = decoded
        assert set(self.letter_tokens.values()) == set(self.native.LETTERS)
        self.native.call_vllm = self.forward
        self.metadata = {'precision': 'NF4 backbone, BF16 compute', 'temperature': 3.4,
                         'interface': 'publisher prompts, letter mass aggregation and calibration; local Transformers one-token readout',
                         'source_revision': read_json(root / 'metadata.json')['rev'], 'option_limit': 26,
                         'max_tokens': 16384, 'context_policy': 'reject overflow, no truncation',
                         'comparison_scope': 'Local NF4/Transformers variant. Published Cygnet uses BF16/vLLM; no runtime parity claim.',
                         'readout': 'Mask to exact decoded uppercase option-letter tokens, then publisher top-20 probability aggregation'}

    def forward(self, prompt_text, allowed_letters):
        import torch
        messages = [{'role': 'system', 'content': self.native.SYSTEM}, {'role': 'user', 'content': prompt_text}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                   **self.native.CHAT_TEMPLATE_KWARGS)
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').to('cuda')
        if inputs.input_ids.shape[-1] > 16384:
            raise ValueError('Cygnet prompt exceeds 16384 tokens')
        candidates = [i for i, letter in self.letter_tokens.items() if letter in allowed_letters]
        with torch.inference_mode():
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()[candidates]
            logprobs = logits.log_softmax(-1)
            order = logprobs.argsort(descending=True)[:20].cpu().tolist()
            values = logprobs.cpu().tolist()
        top = [{'token': self.letter_tokens[candidates[i]], 'logprob': values[i]} for i in order]
        return {'choices': [{'logprobs': {'content': [{'top_logprobs': top}]}}],
                'usage': {'prompt_tokens': inputs.input_ids.shape[-1], 'completion_tokens': 1}}

    def batch(self, rows):
        out = []
        for row in rows:
            qs, mapping = typed_questions(row['questions'])
            answers = {}
            for key, question in qs.items():
                q = dict(question)
                if q['type'] == 'noul' and not q.get('criteria'):
                    q['criteria'] = {'false': 'no', 'true': 'yes'}
                try:
                    answer, _, error = self.native.answer_for(row['state'], q)
                except self.native.Unprocessable as exc:
                    raise ValueError(str(exc)) from exc
                if error:
                    raise RuntimeError(error)
                answers[key] = answer
            out.append(decode_typed(answer_probabilities(answers), row['questions'], mapping))
        return out
