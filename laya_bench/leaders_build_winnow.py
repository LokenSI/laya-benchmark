"""Build Winnow's hash-checked native CUDA engine in the project cache."""
import hashlib
import subprocess
from .common import ROOT, read_json, write_json, digest


def run(*args, cwd):
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def main():
    root = ROOT / '.cache/leader_research/EldanRing--winnow-inference'
    lock = read_json(root / 'runtime.lock.json')
    source = root / '.runtime/llama.cpp'
    source.mkdir(parents=True, exist_ok=True)
    if not (source / '.git').exists():
        run('git', 'init', cwd=source)
        run('git', 'config', 'core.autocrlf', 'false', cwd=source)
        run('git', 'fetch', '--depth', '1', lock['repository'], lock['commit'], cwd=source)
        run('git', 'checkout', '--detach', 'FETCH_HEAD', cwd=source)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    assert revision == lock['commit']
    for item in lock['patches']:
        patch = root / item['file']
        assert digest(patch) == item['sha256']
        check = subprocess.run(['git', 'apply', '--check', str(patch)], cwd=source, capture_output=True)
        if check.returncode == 0:
            run('git', 'apply', patch, cwd=source)
        else:
            run('git', 'apply', '--reverse', '--check', patch, cwd=source)
    changed = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=source, text=True).splitlines()
    assert set(changed) == set(lock['source_sha256'])
    for filename, expected in lock['source_sha256'].items():
        assert digest(source / filename) == expected, filename
    build = root / '.build-windows'
    configure = ['cmake', '--fresh', '-S', str(root), '-B', str(build), '-G', 'Visual Studio 17 2022',
                 '-A', 'x64', '-T', 'cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1',
                 '-DCUDAToolkit_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1',
                 '-DGGML_CUDA=ON', '-DGGML_METAL=OFF', '-DCMAKE_CUDA_ARCHITECTURES=120',
                 '-DLLAMA_OPENSSL=OFF', '-DLLAMA_CURL=OFF']
    run(*configure, cwd=root)
    run('cmake', '--build', build, '--config', 'Release', '--target', 'runtime/tools/server/llama-server', '--parallel', '4', cwd=root)
    run('cmake', '--build', build, '--config', 'Release', '--target', 'winnow-unit', '--parallel', '4', cwd=root)
    run('ctest', '--test-dir', build, '-C', 'Release', '--output-on-failure', cwd=root)
    executable = next(build.rglob('winnow-server.exe'))
    write_json(ROOT / 'results/alternatives/winnow-runtime.json', {
        'source_revision': read_json(root / 'metadata.json')['rev'], 'runtime_lock': lock,
        'executable': str(executable), 'sha256': digest(executable), 'configure': configure,
        'platform': 'Windows MSVC CUDA 13.1 SM120',
        'scope': 'Author engine and locked llama.cpp patches; Windows is outside the publisher validated profiles. TLS disabled for owned loopback-only inference.'})
    print('Verified native Windows build', executable, flush=True)


if __name__ == '__main__':
    main()
