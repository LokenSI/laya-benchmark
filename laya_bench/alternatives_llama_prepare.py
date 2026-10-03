"""Download and verify the pinned official Windows Vulkan llama.cpp runtime."""
import hashlib
import zipfile
import requests
from .common import ROOT,read_json,write_json

def main():
    release=read_json(ROOT/'.cache/alternatives_research/llama-windows-release.json')
    asset=next(a for a in release['assets'] if a['name'].endswith('bin-win-vulkan-x64.zip'))
    root=(ROOT/'.cache/llama-runtime').resolve();root.mkdir(exist_ok=True)
    dest=root/asset['name']
    if not dest.exists() or dest.stat().st_size!=asset['size']:
        with requests.get(asset['browser_download_url'],stream=True,timeout=(30,120)) as res:
            res.raise_for_status()
            with dest.open('wb') as f:
                for chunk in res.iter_content(2**20):f.write(chunk)
    assert dest.stat().st_size==asset['size']
    sha=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()
    if asset.get('digest'):assert asset['digest']=='sha256:'+sha
    with zipfile.ZipFile(dest) as z:
        for info in z.infolist():assert (root/info.filename).resolve().is_relative_to(root)
        assert z.testzip() is None
        z.extractall(root)
    write_json(root/'manifest.json',{'release':release['tag_name'],'asset':asset['name'],'url':asset['browser_download_url'],'sha256':sha,'github_digest':asset.get('digest')})
    print('Runtime ready',flush=True)

if __name__=='__main__':main()
