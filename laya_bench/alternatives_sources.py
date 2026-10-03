"""Fetch public, commit-pinned runtime and claim fixtures (no credentials)."""
import concurrent.futures
from pathlib import Path
import requests
from .common import ROOT,read_json,write_json,digest

R=ROOT/'.cache/alternatives_research'

def main():
    s=requests.Session()
    for repo in ['allebee/jevk5','jaredpalmer/kev','Mapika/decider','togethercomputer/tev1','bespokelabsai/nimble','jabr/classifier-benchmark']:
        dest=R/('code--'+repo.replace('/','--'));dest.mkdir(parents=True,exist_ok=True)
        if not (dest/'tree.json').exists():
            info=s.get('https://api.github.com/repos/'+repo,timeout=30);info.raise_for_status()
            branch=info.json()['default_branch']
            rr=s.get(f'https://api.github.com/repos/{repo}/commits/{branch}',timeout=30);rr.raise_for_status();rev=rr.json()['sha']
            rr=s.get(f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1',timeout=30);rr.raise_for_status()
            write_json(dest/'tree.json',rr.json());(dest/'revision.txt').write_text(rev)
        rev=(dest/'revision.txt').read_text().strip()
        paths=[x['path'] for x in read_json(dest/'tree.json')['tree'] if x['type']=='blob']
        selected=[]
        for p in paths:
            wanted=p.endswith('.py') or p in ['README.md','pyproject.toml']
            if repo=='jaredpalmer/kev':wanted |= p.startswith(('evals/documents-v1/','evals/hard-v1/','evals/devtools-v1/','experiments/releases/kev-08b-r15','experiments/releases/kev-4b-r10')) and not p.endswith('train.jsonl')
            if repo=='togethercomputer/tev1':wanted |= p.startswith('evaluation/') or p=='runs/new-v1/dataset-manifest.json'
            if repo=='bespokelabsai/nimble':wanted |= p in ['data/eval.jsonl','data/manifest.json']
            if repo in ['allebee/jevk5','jabr/classifier-benchmark']:wanted |= p.endswith(('.json','.jsonl','.yaml','.yml','.toml')) and not any(k in p for k in ['lock','train','node_modules'])
            if wanted and not (dest/p).exists():selected.append(p)
        print(repo,'new files',len(selected),flush=True)
        def get(p):
            rr=requests.get(f'https://raw.githubusercontent.com/{repo}/{rev}/{p}',timeout=90);rr.raise_for_status()
            target=dest/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(rr.content)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(get,selected))
        write_json(dest/'downloaded_files.json',{str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file() and p.name!='downloaded_files.json'})
        print(repo,'ready',flush=True)

if __name__=='__main__':main()
