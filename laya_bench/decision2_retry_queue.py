"""Resume incomplete native fixtures; preserve process crashes and try other suites."""
import argparse
import os
import subprocess
import psutil
from .common import ROOT, read_json, write_json
from .jev_api import now


def run(after_pid, models):
    out = ROOT / 'results/alternatives'
    try:
        previous = psutil.Process(after_pid)
        assert 'laya_bench.decision2_queue' in previous.cmdline()
        previous.wait()
    except psutil.NoSuchProcess:
        pass
    state = {'pid': os.getpid(), 'started': now(), 'status': 'running', 'models': models, 'jobs': [],
             'policy': 'Same checkpoint/runtime/math; resume saved IDs. One process retry for a Windows access violation only. Other frozen suites continue after a failed fixture. No crash is converted into a prediction.'}
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'TMP': str(ROOT / '.cache/decision2-tmp'), 'TEMP': str(ROOT / '.cache/decision2-tmp'),
                   'TRITON_CACHE_DIR': str(ROOT / '.cache/decision2-triton'),
                   'TORCHINDUCTOR_CACHE_DIR': str(ROOT / '.cache/decision2-inductor')}
    def save():
        write_json(out / 'queue-decision2-retry.json', state)
    save()
    for name in models:
        for fixture in ['jev_fresh', 'jev_verified', 'alternatives', 'alternatives_claims', 'alternatives_typed']:
            for attempt in [1, 2]:
                log = out / f'logs/{name}-{fixture}.log'
                job = {'model': name, 'fixture': fixture, 'attempt': attempt, 'started': now(),
                       'status': 'running', 'log': str(log)}
                state['jobs'].append(job)
                with log.open('a', encoding='utf-8') as stream:
                    stream.write('\n=== Exact-runtime resumable retry '+job['started']+' ===\n')
                    stream.flush()
                    process = subprocess.Popen([str(ROOT / '.venv-decision/Scripts/python.exe'), '-X', 'utf8',
                                                '-m', 'laya_bench.alternatives_run', name, '--fixture', fixture,
                                                '--batch-size', '1'], cwd=ROOT, env=environment,
                                               stdout=stream, stderr=subprocess.STDOUT,
                                               creationflags=subprocess.CREATE_NO_WINDOW)
                    job['worker_pid'] = process.pid
                    save()
                    code = process.wait()
                job.update(status='finished' if code == 0 else 'needs_repair', exit_code=code, finished=now())
                save()
                with (out / 'logs/decision2-retry-report.log').open('a', encoding='utf-8') as stream:
                    subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8', '-m',
                                    'laya_bench.alternatives_report'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                if code not in [3221225477, -1073741819]:
                    break
    latest = {(job['model'], job['fixture']): job for job in state['jobs']}
    state.update(status='finished' if all(job['exit_code'] == 0 for job in latest.values()) else 'needs_repair',
                 finished=now())
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-pid', type=int, required=True)
    parser.add_argument('--models', nargs='+', default=['decision2-sol-2b', 'decision2-nox-4b'])
    args = parser.parse_args()
    run(args.after_pid, args.models)
