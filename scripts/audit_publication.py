"""Review the exact publication allowlist and optionally the Git staging area."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SENSITIVE = re.compile(rb'apikey_[0-9a-f]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', re.I)


def audit(staged=False):
    manifest = json.loads((ROOT/'docs/public-files.json').read_text(encoding='utf-8'))
    public = set(manifest['files'])
    assert len(public) == len(manifest['files'])
    names = set(public)
    if staged:
        payload = subprocess.check_output(['git','diff','--cached','--name-only','-z'], cwd=ROOT)
        names = {name.decode('utf-8') for name in payload.split(b'\0') if name}
        assert names, 'Nothing is staged'
    total = 0
    for name in sorted(names):
        path = ROOT/name
        assert path.resolve().is_relative_to(ROOT)
        assert not any(part.startswith('.venv') or part in {'.cache','data','web','.aws','.codex','.agents'} for part in Path(name).parts), f'Unwanted path: {name}'
        if name.startswith('results/'):
            assert name in public, f'Result is outside the explicit publication allowlist: {name}'
            assert not re.search(r'predictions|payload|secrets|embedding.cache|workers/|baselines/', name, re.I)
        elif staged:
            assert name in {'README.md','sources.json','requirements.txt','requirements-lock.txt','.gitignore','.gitattributes'} or name.startswith(('laya_bench/','tests/','scripts/','docs/','.github/workflows/')), f'Unexpected staged file: {name}'
        data = (subprocess.check_output(['git','show',':'+name], cwd=ROOT) if staged else path.read_bytes())
        assert not SENSITIVE.search(data), f'Credential-like content detected in {name}'
        assert len(data) < 50*2**20, f'Unexpectedly large publication artifact: {name}'
        if path.suffix == '.zip':
            import io
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for entry in archive.infolist():
                    assert Path(entry.filename).name == entry.filename
                    assert entry.file_size < 50*2**20
                    assert not SENSITIVE.search(archive.read(entry))
        total += len(data)
    print(f'Publication audit passed: {len(names)} files, {total/2**20:.1f} MiB; no cache, environments, raw predictions or credentials.')
    return {'files': len(names), 'bytes': total}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--staged', action='store_true')
    audit(parser.parse_args().staged)
