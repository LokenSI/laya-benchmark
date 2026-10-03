"""TypeSafe HTTP client. Credentials never enter persisted requests or logs."""
import argparse
import ctypes
import json
import math
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import requests
from .common import ROOT, write_json, read_json, fingerprint

OUT = ROOT / 'results/jev_live'
BASE = 'https://api.typesafe.ai'
PRICE_PER_TOKEN = 42 / 1_000_000_000
CAP_USD = 25.0

def now():
    return datetime.now(timezone.utc).isoformat()

def credential():
    if os.environ.get('TYPESAFE_API_KEY'):
        return os.environ['TYPESAFE_API_KEY']
    encrypted = bytes.fromhex((ROOT / '.cache/secrets/jev-key.dpapi').read_text(encoding='utf-8-sig').strip())
    from ctypes import wintypes
    class Blob(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    buf = (ctypes.c_ubyte * len(encrypted)).from_buffer_copy(encrypted)
    source, dest = Blob(len(encrypted), buf), Blob()
    crypt = ctypes.windll.crypt32.CryptUnprotectData
    crypt.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    crypt.restype = wintypes.BOOL
    if not crypt(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(dest)):
        raise RuntimeError('Windows credential decryption failed')
    free = ctypes.windll.kernel32.LocalFree
    free.argtypes, free.restype = [ctypes.c_void_p], ctypes.c_void_p
    try:
        return ctypes.string_at(dest.data, dest.length).decode('utf-16-le')
    finally:
        free(dest.data)

def append(path, data):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

class Budget:
    """Reserve before sending. Unconfirmed attempts retain their reservation.

    Single process owns this ledger; a Windows file lock prevents other runners.
    Conservative token reservation covers tokenization and repeated questions.
    Actual charges are estimates from reported billable tokens and dated price.
    """
    def __init__(self, directory=OUT):
        import msvcrt
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.path = directory / 'usage-ledger.jsonl'
        self.guard = (directory / 'runner.lock').open('a+b')
        self.guard.seek(0); self.guard.write(b'0'); self.guard.flush(); self.guard.seek(0)
        msvcrt.locking(self.guard.fileno(), msvcrt.LK_NBLCK, 1)
        self.lock = threading.Lock()
        self.events = [json.loads(s) for s in self.path.read_text(encoding='utf-8').splitlines()] if self.path.exists() else []
    def summary(self):
        reserved = {e['attempt']: e['reserved_usd'] for e in self.events if e['event'] == 'reserve'}
        tokens = 0; output = 0; completed = 0
        for e in self.events:
            if e['event'] == 'settle':
                reserved.pop(e['attempt'], None)
                tokens += e['input_tokens']; output += e['output_tokens']; completed += 1
        cost = tokens * PRICE_PER_TOKEN
        return {'billable_input_tokens': tokens, 'output_tokens': output, 'completed_attempts': completed,
                'estimated_cost_usd': cost, 'unconfirmed_or_inflight_reserved_usd': sum(reserved.values()),
                'budget_accounted_usd': cost + sum(reserved.values()), 'cap_usd': CAP_USD,
                'price_per_million_input_tokens_usd': PRICE_PER_TOKEN * 1e6, 'updated': now()}
    def reserve(self, payload, case_id):
        token_bound = max(1_000_000, 16 * (len(json.dumps(payload, ensure_ascii=False).encode('utf-8')) + 4096) * len(payload['questions']))
        amount = token_bound * PRICE_PER_TOKEN
        with self.lock:
            if self.summary()['budget_accounted_usd'] + amount > CAP_USD:
                raise RuntimeError('Budget reservation would exceed the authorized $25 cap')
            event = {'event': 'reserve', 'attempt': str(uuid.uuid4()), 'case_id': case_id,
                     'payload_sha256': fingerprint(payload), 'reserved_usd': amount, 'time': now()}
            append(self.path, event); self.events.append(event)
            return event['attempt']
    def settle(self, attempt, usage):
        ni, no = usage['input_tokens'], usage['output_tokens']
        if type(ni) is not int or type(no) is not int or min(ni, no) < 0:
            raise ValueError('Invalid billing usage')
        with self.lock:
            event = {'event': 'settle', 'attempt': attempt, 'input_tokens': ni, 'output_tokens': no, 'time': now()}
            append(self.path, event); self.events.append(event)
            write_json(self.directory / 'spend.json', self.summary())
            if self.summary()['budget_accounted_usd'] > CAP_USD:
                raise RuntimeError('Reported usage exceeded reserved budget; all further calls stopped')
    def close(self):
        self.guard.close()

class Client:
    def __init__(self, budget, session=None):
        self.budget = budget
        self.key = credential()
        self.session = session
    def request(self, payload, case_id):
        attempt = self.budget.reserve(payload, case_id)
        t = time.perf_counter()
        try:
            r = (self.session or requests).post(BASE + '/v1/systemone', json=payload,
                              headers={'Authorization': 'Bearer ' + self.key},
                              timeout=(20, 90), allow_redirects=False)
        except requests.RequestException as e:
            return {'error': type(e).__name__, 'attempt': attempt, 'seconds': time.perf_counter() - t}
        elapsed = time.perf_counter() - t
        # Persist only the JSON service response; never a requests object/header.
        try: body = r.json()
        except ValueError: body = None
        if isinstance(body, dict) and isinstance(body.get('usage'), dict):
            self.budget.settle(attempt, body['usage'])
        if r.status_code != 200:
            return {'error': 'HTTP ' + str(r.status_code), 'status_code': r.status_code,
                    'attempt': attempt, 'seconds': elapsed}
        if not isinstance(body, dict) or 'usage' not in body:
            return {'error': 'Invalid API response', 'attempt': attempt, 'seconds': elapsed}
        return {'response': body, 'attempt': attempt, 'seconds': elapsed}

def decode(questions, response):
    from .alternatives_run import validate_answer
    pred, probs, renormalized = {}, {}, []
    assert set(response['answers']) == set(questions), 'Answer question keys differ'
    for key, q in questions.items():
        a = response['answers'][key]
        assert a['type'] == q['type'], 'Answer type differs'
        if q['type'] == 'noul':
            p = float(a['noul']); values = {'false': 1-p, 'true': p}
        else:
            values = {str(k): float(v) for k, v in a['probabilities'].items()}
            assert all(math.isfinite(v) and 0<=v<=1 for v in values.values()), 'Invalid probability'
            total=sum(values.values())
            assert total>0 and abs(total-1)<=.020000001, 'Probability sum outside publisher tolerance'
            if abs(total-1)>1e-8:
                renormalized.append(key); values={k:v/total for k,v in values.items()}
        probs[key] = values
        # Uniform argmax rule; original ordered options break ties.
        keys = list(map(str, range(len(q['criteria'])))) if q['type'] == 'score' else list(q.get('criteria') or ['false', 'true'])
        pred[key] = [max(keys, key=values.__getitem__)]
    result = {'pred': pred, 'probabilities': probs, 'renormalized_questions': renormalized}
    validate_answer({'questions': questions}, result)
    return result

def probe():
    OUT.mkdir(parents=True, exist_ok=True)
    headers = {'Authorization': 'Bearer ' + credential()}
    r = requests.get(BASE + '/v1/models', headers=headers, timeout=30, allow_redirects=False)
    print('Model inventory HTTP', r.status_code, flush=True)
    if r.status_code != 200: raise RuntimeError('Authenticated model discovery failed')
    inventory = r.json(); write_json(OUT / 'model-inventory.json', {'retrieved': now(), **inventory})
    print(json.dumps(inventory, ensure_ascii=False, indent=2), flush=True)
    questions = {'category': {'type': 'choice', 'instructions': 'What is broken?', 'criteria': {'printer': 'A printer', 'network': 'A network'}},
                 'failure': {'type': 'noul', 'instructions': 'Does the message report a failure?'},
                 'count': {'type': 'score', 'instructions': 'How many printers are broken?', 'criteria': ['Zero', 'One', 'Two']}}
    payload = {'model': 'jev-latest', 'state': 'One printer is broken.', 'questions': questions}
    budget = Budget()
    try:
        result = Client(budget).request(payload, 'integration-probe')
        write_json(OUT / 'integration-probe.json', {'request': payload, **result})
        if 'response' not in result: raise RuntimeError(result['error'])
        decode(questions, result['response'])
        write_json(OUT / 'model-pin.json', {'model': result['response']['model'], 'resolved': now(), 'alias': 'jev-latest',
                   'pricing_source': 'https://typesafe.ai/', 'pricing_observed': '2026-10-01', 'usd_per_billion_input_tokens': 42,
                   'output_tokens_free_source': 'https://api.typesafe.ai/openapi.json'})
        print(json.dumps(result, indent=2), flush=True)
    finally: budget.close()

if __name__ == '__main__':
    probe()
