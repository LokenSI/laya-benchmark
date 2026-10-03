"""Download pinned public checkpoints and benchmark sources for the October study."""
import argparse
import concurrent.futures
import json
from pathlib import Path

from .common import ROOT,read_json,write_json,digest

RESEARCH=ROOT/".cache/alternatives_research"
MODELS={
    "gliner-decide":"fastino/GLiNER2.5-Decide",
    "gliner-decide-multi":"fastino/GLiNER2.5-multi-Decide",
    "gliner-decide-1b":"fastino/GLiNER2.5-Decide-1B",
    "von":"wfzyx/von",
    "kev-08b":"jaredpalmer/kev-0.8b",
    "kev-4b":"jaredpalmer/kev-4b",
    "decider-08b":"Mapika/decider-0.8b",
    "decider-2b":"Mapika/decider-2b",
    "decider-4b":"Mapika/decider-4b",
    "jevk5":"alibiserikbay/JevK5",
    "jevk5-2b":"alibiserikbay/JevK5-2B",
    "tev1":"togethercomputer/Tev1-4B-experimental",
    "nimble-9b":"bespokelabs/Bespoke-Nimble-9B",
    "nev-2b":"EryriLabs/Nev-2B-GGUF",
    "plumb-4b":"crh225/plumb-4b",
    "imajev-4b":"mohit67890/imajev-4b",
    "imajev-2b":"mohit67890/imajev-2b",
    "intern-decision-4b":"internlm/Intern-Decision-4B",
    "wald-4b-v12":"org2ai/Wald-4B",
    "julia":"SupersonicLabs/Julia-1",
}


def download_model(name):
    from huggingface_hub import snapshot_download,HfApi
    import huggingface_hub.file_download as fd
    # Windows may allow the probe but reject the actual symlink. Cache by copy.
    fd.are_symlinks_supported = lambda cache_dir=None: False
    repo=MODELS[name]
    api=read_json(RESEARCH/repo.replace('/','--')/"api.json")
    rev=api["sha"]
    patterns=["*.json","*.txt","*.safetensors","*.py","*.pt","*.model","*.jinja","LICENSE*","requirements.txt"]
    if name=="nev-2b":
        files=[x['rfilename'] for x in api['siblings'] if x['rfilename'].endswith('.gguf') and 'Q8_0' in x['rfilename']]
        if len(files)!=1:
            raise ValueError(f"Choose exactly one Q8 Nev file: {files}")
        patterns += files
    print('Downloading',name,rev[:10],flush=True)
    path=snapshot_download(repo,revision=rev,allow_patterns=patterns,max_workers=3)
    bases=[]
    adapter=Path(path)/"adapter_config.json"
    if adapter.exists():
        config=read_json(adapter)
        base=config["base_model_name_or_path"]
        # Some releases serialize the author's cache path, including the exact
        # upstream revision. Resolve that provenance instead of downloading main.
        import re
        cached=re.search(r'models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{40})',base.replace('\\','/'))
        if cached:
            owner,model,base_rev=cached.groups();base=f'{owner}/{model}'
        else:
            base_rev=config.get("revision") or HfApi().model_info(base).sha
        if (Path(path)/"head.pt").exists() and name.startswith("kev"):
            import torch
            head=torch.load(Path(path)/"head.pt",map_location="cpu",weights_only=True)
            base_rev=head.get("base_revision") or base_rev
        base_path=snapshot_download(base,revision=base_rev,allow_patterns=["*.json","*.txt","*.safetensors","*.model","*.jinja"],max_workers=3)
        bases.append({"repo":base,"revision":base_rev,"path":base_path})
    entry={"repo":repo,"revision":rev,"path":path,"bases":bases,"license":api.get('cardData',{}).get('license'),"model_card_sha256":digest(RESEARCH/repo.replace('/','--')/'README.md')}
    write_json(ROOT/f"results/alternatives/models/{name}.json",entry)
    print('Ready',name,flush=True)
    return entry


def download_data():
    from huggingface_hub import snapshot_download
    for name in ['fastino/fast-decisions','jaredpalmer/kev-suites']:
        api=read_json(RESEARCH/(name.replace('/','--')+'-dataset.json'))
        rev=api['sha']
        patterns=['*.jsonl','*.json','README.md'] if name.startswith('fastino') else ['v7/*/test.jsonl','v7/*/development.jsonl','v7/*/manifest.json','transfer-v4/*','README.md']
        path=snapshot_download(name,repo_type='dataset',revision=rev,allow_patterns=patterns,max_workers=4)
        write_json(ROOT/f"results/alternatives/data/{name.replace('/','--')}.json",{'repo':name,'revision':rev,'path':path})
        print('Data',name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='*');p.add_argument('--data',action='store_true');a=p.parse_args()
    if a.data:download_data()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        futures={pool.submit(download_model,n):n for n in a.names}
        for future in concurrent.futures.as_completed(futures):
            try: future.result()
            except Exception as e: print('FAILED',futures[future],repr(e),flush=True)
