"""Download pinned upstream models; EchoLens adapters are installed separately."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    inventory=json.loads((ROOT/'configs/releases/v0.1-upstream-files.json').read_text(encoding='utf-8'))
    models = json.loads((ROOT/'configs/models.json').read_text(encoding='utf-8'))
    models['encoder'] = json.loads((ROOT/'configs/stage1.json').read_text(encoding='utf-8'))['encoder']
    for name,model in models.items():
        listed=inventory[name]
        if any(model[key]!=listed[key] for key in ['repo_id','revision','local_dir']):
            raise ValueError('Model configuration differs from release inventory')
        if not args.verify_only:
            from huggingface_hub import snapshot_download
            print(f"Downloading {name}: {model['repo_id']} @ {model['revision']}",flush=True)
            snapshot_download(repo_id=model['repo_id'],revision=model['revision'],local_dir=ROOT/model['local_dir'],
                              allow_patterns=[f['file'] for f in listed['files']],max_workers=2)
        for item in listed['files']:
            path=ROOT/model['local_dir']/item['file']
            with path.open('rb') as stream:
                actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if path.stat().st_size!=item['bytes'] or actual!=item['sha256']:
                raise ValueError(f'Model file changed: {path}')
        print(f"Verified {name}: {len(listed['files'])} files",flush=True)


if __name__=='__main__':
    main()
