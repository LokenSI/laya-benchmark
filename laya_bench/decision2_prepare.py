"""Download and hash-pin native Decision 2.0 packages for the existing fixtures."""
import argparse
import importlib.util
import importlib
import hashlib
import sys
from pathlib import Path
from .common import ROOT, read_json, write_json, digest

MODELS = {
    'decision2-kai-06b': 'vllm-sr/Decision-2.0-Kai-0.6B',
    'decision2-eos-08b': 'vllm-sr/Decision-2.0-Eos-0.8B',
    'decision2-sol-2b': 'vllm-sr/Decision-2.0-Sol-2B',
    'decision2-nox-4b': 'vllm-sr/Decision-2.0-Nox-4B',
}

def portable_identity(result, identity, canonical):
    files = {key.replace('\\', '/'): value for key, value in result['files_sha256'].items()}
    if len(files) != len(result['files_sha256']) or files != identity['fingerprint_files']:
        raise ValueError('Model identity file roster differs from the scored checkpoint')
    checksum = hashlib.sha256(canonical(files).encode('utf-8')).hexdigest()
    if checksum != identity['model_sha256']:
        raise ValueError('Portable model identity differs from the scored checkpoint')
    return {'model_sha256': checksum, 'files_sha256': files}

def native_package(path, revision):
    """Verify source against the pinned manifest before importing it."""
    path = Path(path)
    manifest = read_json(path / 'MODEL_MANIFEST.json')
    for filename, expected in manifest['files_sha256'].items():
        target = path / filename
        if not target.resolve().is_relative_to(path.resolve()) or target.is_symlink():
            raise ValueError('Unsafe package path: ' + filename)
        if digest(target) != expected:
            raise ValueError('Package hash mismatch: ' + filename)
    name = 'decision2_pinned_' + revision
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path / 'decision2/__init__.py',
                                                submodule_search_locations=[str(path / 'decision2')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if sys.platform == 'win32':
        # Publisher fingerprints use str(relative_path), which emits backslashes
        # on Windows. Canonicalize names only, and still enforce the exact file
        # roster and scored digest. Package bytes and inference remain unchanged.
        infer = importlib.import_module(name + '._vendor.dev2model.infer')
        original = infer.checkpoint_fingerprint
        data = importlib.import_module(name + '._vendor.dev2model.data')
        def portable_fingerprint(model_path, source_path=None):
            result = original(model_path, source_path)
            return portable_identity(result, manifest['identity'], data.canonical)
        infer.checkpoint_fingerprint = portable_fingerprint
    return module

def prepare(name):
    from huggingface_hub import snapshot_download
    import huggingface_hub.file_download as fd
    fd.are_symlinks_supported = lambda cache_dir=None: False
    repo = MODELS[name]
    research = ROOT / '.cache/decision2_research' / repo.split('/')[-1]
    info = read_json(research / 'api.json')
    assert info['cardData']['license'] == 'apache-2.0'
    revision = info['sha']
    print('Downloading', name, revision, flush=True)
    path = Path(snapshot_download(repo, revision=revision, max_workers=3))
    native = native_package(path, revision)
    manifest = native.verify_bundle(path)
    assert manifest['profile'] == 'qwen-full'
    entry = {'repo': repo, 'revision': revision, 'path': str(path), 'bases': [],
             'license': 'apache-2.0', 'model_card_sha256': digest(path / 'README.md'),
             'manifest_sha256': digest(path / 'MODEL_MANIFEST.json'),
             'verified_files': manifest['files_sha256'],
             'parameters': manifest['parameters'], 'max_input_tokens': manifest['max_input_tokens'],
             'comparison_scope': 'Publisher native exact path on Windows NVIDIA CUDA with installed Transformers/FLA attention, eager execution. Not the publisher ROCm latency environment.'}
    write_json(ROOT / f'results/alternatives/models/{name}.json', entry)
    print('Verified', name, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', choices=MODELS)
    for name in parser.parse_args().models:
        prepare(name)
