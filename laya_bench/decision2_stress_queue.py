"""Give stress tests one GPU job boundary; always resume the standard queue."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import time
import psutil
import requests
from .common import ROOT, read_json, write_json
from .jev_api import now

OUT = ROOT / 'results/decision2-stress'
IMAGE = 'vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90'

def run(supervisor_pid, models):
    if (OUT / 'queue.json').exists():
        previous = read_json(OUT / 'queue.json')
        try:
            process = psutil.Process(previous['pid'])
            if 'laya_bench.decision2_stress_queue' in process.cmdline():
                raise RuntimeError('A Decision 2.0 stress coordinator is already running')
        except psutil.NoSuchProcess:
            pass
    parent = psutil.Process(supervisor_pid)
    assert 'laya_bench.decision2_queue' in parent.cmdline()
    state = {'pid': os.getpid(), 'started': now(), 'status': 'waiting_for_standard_worker',
             'supervisor_pid': supervisor_pid, 'models': models, 'jobs': [], 'vllm_image': IMAGE}
    OUT.mkdir(exist_ok=True)
    def save():
        write_json(OUT / 'queue.json', state)
    def execute(stage, command, timeout=None):
        job = {'stage': stage, 'started': now(), 'status': 'running', 'command': command}
        state['jobs'].append(job)
        with (OUT / (stage+'.log')).open('a', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            job['worker_pid'] = process.pid
            save()
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
                if stage.startswith('vllm-'):
                    subprocess.run(['docker', 'stop', 'laya-decision2-'+stage.removeprefix('vllm-')],
                                   stdout=stream, stderr=subprocess.STDOUT, timeout=30)
                code = -1
        job.update(status='finished' if code == 0 else 'needs_repair', exit_code=code, finished=now())
        save()
        return code
    parent.suspend()
    save()
    try:
        queue = read_json(ROOT / 'results/alternatives/queue-decision2.json')
        assert queue['pid'] == supervisor_pid
        for job in queue['jobs']:
            if job['status'] != 'running':
                continue
            try:
                process = psutil.Process(job['worker_pid'])
                assert 'laya_bench.alternatives_run' in process.cmdline()
                process.wait()
            except psutil.NoSuchProcess:
                pass
        state['status'] = 'running'
        save()
        model = read_json(ROOT / 'results/alternatives/models/decision2-sol-2b.json')
        relative = Path(model['path']).relative_to(ROOT).as_posix()
        for implementation in ['auto', 'transformers']:
            stage = 'vllm-'+implementation
            command = ['docker', 'run', '--rm', '--name', 'laya-decision2-'+implementation,
                       '--gpus', 'all', '--network', 'none', '--shm-size', '2g',
                       '--mount', f'type=bind,source={ROOT},target=/workspace,readonly',
                       '--mount', f'type=bind,source={OUT},target=/output',
                       '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
                       '-e', 'VLLM_NO_USAGE_STATS=1', '-e', 'DO_NOT_TRACK=1',
                       '-e', 'HF_HOME=/tmp/huggingface', '--entrypoint', 'python3', IMAGE,
                       '/workspace/laya_bench/decision2_vllm_probe.py', '--model', '/workspace/'+relative,
                       '--implementation', implementation, '--output', '/output/'+stage+'.json']
            execute(stage, command, timeout=240)
        environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'TMP': str(ROOT / '.cache/decision2-tmp'), 'TEMP': str(ROOT / '.cache/decision2-tmp'),
                       'TRITON_CACHE_DIR': str(ROOT / '.cache/decision2-triton'),
                       'TORCHINDUCTOR_CACHE_DIR': str(ROOT / '.cache/decision2-inductor')}
        python = str(ROOT / '.venv-decision/Scripts/python.exe')
        for name in models:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            with (OUT / (name+'-server.log')).open('a', encoding='utf-8') as stream:
                server = subprocess.Popen([python, '-X', 'utf8', '-m', 'laya_bench.decision2_stress',
                                           'serve', '--model', name, '--port', str(port)], cwd=ROOT,
                                          env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                          creationflags=subprocess.CREATE_NO_WINDOW)
                state['server_pid'] = server.pid
                save()
                try:
                    deadline = time.monotonic()+240
                    with requests.Session() as session:
                        session.trust_env = False
                        while time.monotonic() < deadline:
                            if server.poll() is not None:
                                raise RuntimeError('Native server exited: '+name)
                            try:
                                if session.get(f'http://127.0.0.1:{port}/status', timeout=2).status_code == 200:
                                    break
                            except requests.RequestException:
                                pass
                            time.sleep(2)
                        else:
                            raise RuntimeError('Native server readiness timeout: '+name)
                    execute(name+'-native-stress', [str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8',
                            '-m', 'laya_bench.decision2_stress', 'run', '--model', name, '--port', str(port)], timeout=1800)
                finally:
                    try:
                        for child in psutil.Process(server.pid).children(recursive=True):
                            child.terminate()
                        server.terminate()
                        server.wait()
                    except psutil.NoSuchProcess:
                        pass
                    state.pop('server_pid', None)
                    save()
        state.update(status='finished' if all(job['exit_code'] == 0 for job in state['jobs']) else 'needs_repair', finished=now())
        save()
    except Exception as error:
        state.update(status='needs_repair', error=type(error).__name__+': '+str(error), finished=now())
        save()
        raise
    finally:
        if parent.is_running():
            parent.resume()
        state['standard_supervisor_resumed'] = now()
        save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervisor-pid', type=int, required=True)
    parser.add_argument('--models', nargs='+', default=['decision2-eos-08b', 'decision2-sol-2b'])
    args = parser.parse_args()
    run(args.supervisor_pid, args.models)
