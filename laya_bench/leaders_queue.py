"""Insert audited leaders at a GPU job boundary and resume the existing queue."""
import argparse
import os
import subprocess
import time
import psutil
from .common import ROOT, read_json, write_json
from .jev_api import now

MODELS = ['decision-4b-v12', 'winnow-12b-q8', 'cygnet-12b-nf4', 'jev-omni-12b-nf4']
OUT = ROOT / 'results/alternatives'


def run(supervisor_pid):
    parent = psutil.Process(supervisor_pid)
    assert 'laya_bench.alternatives_queue' in parent.cmdline()
    queue = read_json(OUT / 'queue.json')
    assert queue['pid'] == supervisor_pid
    state = {'pid': os.getpid(), 'models': MODELS, 'started': now(), 'supervisor_pid': supervisor_pid,
             'status': 'waiting_for_current_worker', 'jobs': []}
    def save():
        write_json(OUT / 'queue-leaders-priority.json', state)
    parent.suspend()
    try:
        # Re-read only after suspending the owned supervisor to avoid a job-boundary race.
        queue = read_json(OUT / 'queue.json')
        save()
        for job in queue['jobs']:
            if job['status'] != 'running':
                continue
            try:
                worker = psutil.Process(job['worker_pid'])
                assert 'laya_bench.alternatives_run' in worker.cmdline()
                print('Waiting for current GPU worker', job['model'], job['fixture'], flush=True)
                worker.wait()
            except psutil.NoSuchProcess:
                pass
        state['status'] = 'running'
        save()
        for name in MODELS:
            dependencies = [OUT / f'models/{name}.json']
            if name == 'winnow-12b-q8':
                dependencies.append(OUT / 'winnow-runtime.json')
            if not all(p.exists() for p in dependencies):
                state['jobs'].append({'model': name, 'status': 'preparation_not_ready',
                                      'missing': [str(p) for p in dependencies if not p.exists()]})
                save()
                continue
            commands = [('integration', ['-m', 'laya_bench.leaders_smoke', name])]
            commands += [('public_jevbench_231', ['-m', 'laya_bench.alternatives_run', name,
                          '--fixture', 'alternatives_claims', '--suite', 'jevbench_public/', '--batch-size', '1'])]
            commands += [(fixture, ['-m', 'laya_bench.alternatives_run', name, '--fixture', fixture, '--batch-size', '1'])
                         for fixture in ['jev_verified', 'jev_fresh']]
            for stage, command in commands:
                log_path = OUT / f'logs/leader-{name}-{stage}.log'
                job = {'model': name, 'stage': stage, 'status': 'running', 'started': now(), 'log': str(log_path)}
                state['jobs'].append(job)
                save()
                with log_path.open('a', encoding='utf-8') as log:
                    proc = subprocess.Popen([str(ROOT / '.venv-decision/Scripts/python.exe'), '-X', 'utf8', *command],
                                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    job['worker_pid'] = proc.pid
                    save()
                    code = proc.wait()
                job.update(status='finished' if code == 0 else 'needs_repair', exit_code=code, finished=now())
                save()
                if code:
                    break
                with (OUT / 'logs/leaders-report.log').open('a', encoding='utf-8') as report:
                    for module in ['laya_bench.alternatives_report', 'laya_bench.jev_live_report']:
                        subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8', '-m', module],
                                       cwd=ROOT, stdout=report, stderr=subprocess.STDOUT)
        state.update(status='finished', finished=now())
        save()
    finally:
        if parent.is_running():
            parent.resume()
        print('Resumed existing GPU supervisor', supervisor_pid, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('supervisor_pid', type=int)
    run(p.parse_args().supervisor_pid)
