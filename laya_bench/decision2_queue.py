"""Sequential native Decision 2.0 evaluation, prioritizing complete public suites."""
import os
import subprocess
import time
from .common import ROOT, write_json
from .decision2_prepare import MODELS
from .jev_api import now

def run():
    out = ROOT / 'results/alternatives'
    state_path = out / 'queue-decision2.json'
    if state_path.exists():
        import psutil
        from .common import read_json
        prior = read_json(state_path)
        try:
            process = psutil.Process(prior['pid'])
            if 'laya_bench.decision2_queue' in process.cmdline():
                raise RuntimeError('A Decision 2.0 GPU supervisor is already running')
        except psutil.NoSuchProcess:
            pass
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    python = str(ROOT / '.venv-decision/Scripts/python.exe')
    temporary = ROOT / '.cache/decision2-tmp'
    temporary.mkdir(exist_ok=True)
    environment = {**os.environ, 'TMP': str(temporary), 'TEMP': str(temporary),
                   'TRITON_CACHE_DIR': str(ROOT / '.cache/decision2-triton'),
                   'TORCHINDUCTOR_CACHE_DIR': str(ROOT / '.cache/decision2-inductor'),
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    state = {'pid': os.getpid(), 'started': now(), 'status': 'running', 'jobs': [],
             'models': list(MODELS), 'scope': 'Pinned native weights, existing frozen fixtures; one GPU worker at a time',
             'deferred': {'Decision-2.0-Lux-9B': 'Native weights exceed local GPU budget',
                          'Decision-2.0-Vega-27B': 'Native weights exceed local GPU budget'}}
    def save():
        write_json(out / 'queue-decision2.json', state)
    def job(name, stage, args):
        entry = {'model': name, 'stage': stage, 'status': 'running', 'started': now(),
                 'log': str(logs / f'{name}-{stage}.log')}
        state['jobs'].append(entry)
        with open(entry['log'], 'a', encoding='utf-8') as stream:
            process = subprocess.Popen([python, '-X', 'utf8', *args], cwd=ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       env=environment,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            entry['worker_pid'] = process.pid
            save()
            code = process.wait()
        entry.update(status='finished' if code == 0 else 'needs_repair', exit_code=code, finished=now())
        save()
        with (logs / 'decision2-report.log').open('a', encoding='utf-8') as stream:
            subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8', '-m',
                            'laya_bench.decision2_report'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        return code == 0
    ready = []
    save()
    for name in MODELS:
        state['waiting_for_checkpoint'] = name
        save()
        deadline = time.monotonic() + 7200
        while not (out / f'models/{name}.json').exists():
            if time.monotonic() > deadline:
                state.update(status='preparation_timeout', finished=now())
                save()
                return
            time.sleep(5)
        state.pop('waiting_for_checkpoint', None)
        if job(name, 'integration', ['-m', 'laya_bench.leaders_smoke', name]) and job(
                name, 'public231', ['-m', 'laya_bench.alternatives_run', name, '--fixture',
                                    'alternatives_claims', '--suite', 'jevbench_public/', '--batch-size', '1']):
            ready.append(name)
    for fixture in ['jev_verified', 'jev_fresh', 'alternatives', 'alternatives_claims', 'alternatives_typed']:
        for name in list(ready):
            if not job(name, fixture, ['-m', 'laya_bench.alternatives_run', name, '--fixture', fixture, '--batch-size', '1']):
                ready.remove(name)
        with (logs / 'decision2-report.log').open('a', encoding='utf-8') as stream:
            subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8', '-m',
                            'laya_bench.alternatives_report'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    state.update(status='finished' if len(ready) == len(MODELS) else 'needs_repair', finished=now())
    save()

if __name__ == '__main__':
    run()
