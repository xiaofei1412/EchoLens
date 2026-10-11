"""Create a local source review archive from public paths; never upload."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ['app','src','scripts','tests','configs','docs','examples','licenses','.streamlit']
FILES = ['README.md','LICENSE','NOTICE','THIRD_PARTY_NOTICES.md','MODEL_CARD.md',
         'CHANGELOG.md','AGENTS.md','PROJECT_PROPOSAL.md','pyproject.toml','.gitignore','.gitattributes']
EXTENSIONS = {'.py','.md','.json','.jsonl','.txt','.toml','.ps1','.html','.css','.svg'}


def public_files(root):
    selected = [root/name for name in FILES]+list(root.glob('requirements*.txt'))
    for directory in DIRECTORIES:
        for path in (root/directory).rglob('*'):
            relative = path.relative_to(root)
            if any(p in {'__pycache__','.pytest_cache'} or p.endswith('.egg-info') for p in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f'Symlink in public tree: {relative}')
            if path.is_file() and (path.suffix in EXTENSIONS or path.name=='.nojekyll'):
                if path.suffix=='.jsonl' and directory!='examples':
                    raise ValueError(f'Unexpected record file in source: {relative}')
                if path.stat().st_size>5*1024*1024:
                    raise ValueError(f'Unexpected large source file: {relative}')
                selected.append(path)
    return sorted(set(selected))


def build(root, output):
    selected=public_files(root)
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest=[]
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            data=path.read_bytes(); name=path.relative_to(root).as_posix()
            manifest.append({'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            info=zipfile.ZipInfo(name,date_time=(2026,10,12,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    receipt={'archive':output.name,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
             'bytes':output.stat().st_size,'files':manifest}
    output.with_suffix('.manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    return {k:(len(v) if k=='files' else v) for k,v in receipt.items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(ROOT,parser.parse_args().output),ensure_ascii=False,indent=2))
