"""Actual vLLM engine construction against the complete Decision 2.0 package.

Run inside the official Linux image. Never substitute the backbone checkpoint
or an unrelated language-model head for the published decision model.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import traceback

def run(model, implementation, output):
    import vllm
    import torch
    import transformers
    from vllm import LLM
    result = {'vllm_version': vllm.__version__, 'model_path': model,
              'python': platform.python_version(), 'platform': platform.platform(),
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'model_revision': Path(model).name,
              'manifest_sha256': hashlib.sha256((Path(model)/'MODEL_MANIFEST.json').read_bytes()).hexdigest(),
              'model_impl': implementation, 'runner': 'pooling',
              'architecture': json.loads((Path(model)/'config.json').read_text())['architectures'],
              'scope': 'Complete published Decision 2.0 package; no architecture override or backbone substitution'}
    try:
        engine = LLM(model=model, trust_remote_code=True, model_impl=implementation,
                     runner='pooling', enforce_eager=True, gpu_memory_utilization=.65,
                     max_model_len=512, max_num_seqs=32)
        result['status'] = 'engine_constructed'
        result['supported_tasks'] = list(engine.llm_engine.get_supported_tasks())
        result['note'] = 'Engine construction alone does not establish System One/head parity.'
    except Exception as error:
        result.update(status='unsupported', error=type(error).__name__+': '+str(error),
                      traceback=traceback.format_exc())
    Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--implementation', choices=['auto', 'transformers'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.model, args.implementation, args.output)
