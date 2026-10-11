"""Build/install the reviewed local model bundle; never publish or download."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'configs/releases/v0.1-model-files.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f'Path outside installation root: {relative}')
    return path


def entries(root):
    files = json.loads((root/MANIFEST).read_text(encoding='utf-8'))['files']
    if len({f['member'] for f in files}) != len(files):
        raise ValueError('Duplicate manifest member')
    for item in files:
        safe_path(root,item['source'])
        safe_path(root,item['destination'])
    return files


def matches(path, item):
    return path.is_file() and path.stat().st_size==item['bytes'] and sha256(path)==item['sha256']


def build(root, output):
    files = entries(root)
    for item in files:
        if not matches(root/item['source'],item):
            raise ValueError(f"Source missing or changed: {item['source']}")
    output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation protects an already-reviewed distribution artifact.
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_STORED) as archive:
        for item in files:
            info = zipfile.ZipInfo(item['member'],date_time=(2026,10,12,0,0,0))
            with archive.open(info,'w',force_zip64=True) as target, (root/item['source']).open('rb') as source:
                shutil.copyfileobj(source,target)
    return {'archive':str(output),'sha256':sha256(output),'bytes':output.stat().st_size,'files':len(files)}


def install(root, bundle):
    files = entries(root)
    # Trust only the manifest shipped with the reviewed source, not archive metadata.
    with zipfile.ZipFile(bundle) as archive, tempfile.TemporaryDirectory(prefix='echolens-model-') as temp:
        members = archive.infolist()
        if len(members)!=len(files) or {m.filename for m in members}!={f['member'] for f in files}:
            raise ValueError('Unexpected, missing or duplicate archive members')
        for index,item in enumerate(files):
            info = archive.getinfo(item['member'])
            if info.file_size!=item['bytes'] or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Invalid archive size or symlink')
            staged = Path(temp)/str(index)
            with archive.open(info) as source, staged.open('wb') as target:
                shutil.copyfileobj(source,target)
            if not matches(staged,item):
                raise ValueError(f"Checksum mismatch: {item['member']}")
            target = safe_path(root,item['destination'])
            if target.exists() and not matches(target,item):
                raise ValueError(f'Refusing to overwrite changed file: {target}')
        # All checks finish before writing any target. Existing identical files are kept.
        for index,item in enumerate(files):
            target = safe_path(root,item['destination'])
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream, (Path(temp)/str(index)).open('rb') as source:
                    shutil.copyfileobj(source,stream)
    return verify(root)


def verify(root):
    files = entries(root)
    failed = [item['destination'] for item in files if not matches(safe_path(root,item['destination']),item)]
    if failed:
        raise ValueError('Missing or changed installed files: '+', '.join(failed))
    return {'verified_files':len(files)}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command',required=True)
    sub.add_parser('build').add_argument('--output',type=Path,required=True)
    sub.add_parser('install').add_argument('--archive',type=Path,required=True)
    sub.add_parser('verify')
    args = parser.parse_args()
    result = build(ROOT,args.output) if args.command=='build' else install(ROOT,args.archive) if args.command=='install' else verify(ROOT)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
