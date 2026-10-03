"""Pin and prepare additional open-code/open-weight JevBench leaders."""
import argparse
import hashlib
from pathlib import Path

from .common import ROOT, read_json, write_json, digest

RESEARCH = ROOT / '.cache/leader_research'
CANDIDATES = {
    'winnow-12b-q8': ('EldanRing/Winnow-12B', ['gguf/Winnow-12B-Q8_0.gguf', 'SHA256SUMS', 'release-manifest.json', 'README.md', 'LICENSE', 'NOTICE']),
    'cygnet-12b-nf4': ('google/gemma-4-12B-it', ['*.json', '*.jinja', '*.safetensors', 'README.md']),
    'jev-omni-12b-nf4': ('akhilaaa3/Jev-Omni', ['*.json', '*.jinja', '*.safetensors', 'head.pt', 'jev_omni.py', 'README.md']),
    'decision-4b-v12': ('flymy-ai/decision-4b-v1.2', ['*']),
}


def prepare(name):
    from huggingface_hub import snapshot_download
    import huggingface_hub.file_download as fd
    fd.are_symlinks_supported = lambda cache_dir=None: False
    repo, patterns = CANDIDATES[name]
    research = RESEARCH / repo.replace('/', '--')
    info = read_json(research / 'api.json')
    assert info.get('cardData', {}).get('license') == 'apache-2.0'
    revision = info['sha']
    print('Downloading pinned open weights', name, revision, flush=True)
    path = Path(snapshot_download(repo, revision=revision, allow_patterns=patterns, max_workers=3))
    checks = {}
    if name == 'winnow-12b-q8':
        model = path / 'gguf/Winnow-12B-Q8_0.gguf'
        checksum_lines = (path / 'SHA256SUMS').read_text().splitlines()
        expected = next(line.split()[0] for line in checksum_lines if line.split()[-1].lstrip('*').endswith(model.name))
        actual = digest(model)
        assert actual == expected, 'Winnow model checksum mismatch'
        checks[str(model.relative_to(path))] = actual
    bases = []
    if name == 'decision-4b-v12':
        config = read_json(path / 'model.json')
        for filename, expected in read_json(path / 'manifest.json')['files'].items():
            candidate = path / filename
            assert candidate.resolve().is_relative_to(path.resolve()) and not candidate.is_symlink()
            assert candidate.stat().st_size == expected['bytes'] and digest(candidate) == expected['sha256'], filename
        base_path = snapshot_download(config['base_model'], revision=config['base_revision'],
                                      allow_patterns=['*.json', '*.jinja', '*.safetensors', '*.txt', '*.model'], max_workers=3)
        bases.append({'repo': config['base_model'], 'revision': config['base_revision'], 'path': base_path})
    scope = ('published Q8 weights and native decision engine; Windows CUDA port' if name.startswith('winnow')
             else 'native BF16 adapter, publisher temperature and prompts; eager GPU, no CUDA graphs' if name.startswith('decision-')
             else 'local NF4 variant; not an exact reproduction of the published precision/runtime')
    entry = {'repo': repo, 'revision': revision, 'path': str(path), 'bases': bases,
             'license': 'apache-2.0', 'model_card_sha256': digest(research / 'README.md'),
             'verified_files': checks,
             'comparison_scope': scope}
    write_json(ROOT / f'results/alternatives/models/{name}.json', entry)
    print('Checkpoint verified', name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', choices=CANDIDATES)
    args = parser.parse_args()
    for name in args.models:
        prepare(name)
