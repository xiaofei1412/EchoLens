import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('model_bundle',ROOT/'scripts/model_bundle.py')
bundle=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundle)


def fixture(root):
    source=root/'source.bin';source.write_bytes(b'fixed weights')
    item={'source':'source.bin','member':'weights.bin','destination':'artifacts/model/weights.bin',
          'bytes':source.stat().st_size,'sha256':bundle.sha256(source)}
    manifest=root/bundle.MANIFEST;manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'files':[item]}),encoding='utf-8')
    return item


def test_model_install_idempotent_and_refuses_overwrite(tmp_path):
    item=fixture(tmp_path);archive=tmp_path/'bundle.zip'
    bundle.build(tmp_path,archive);bundle.install(tmp_path,archive)
    installed=tmp_path/item['destination'];mtime=installed.stat().st_mtime_ns
    assert bundle.install(tmp_path,archive)=={'verified_files':1}
    assert installed.stat().st_mtime_ns==mtime
    installed.write_bytes(b'user change')
    with pytest.raises(ValueError,match='overwrite'):
        bundle.install(tmp_path,archive)
    assert installed.read_bytes()==b'user change'


@pytest.mark.parametrize('failure',['checksum','extra','missing','duplicate','symlink'])
def test_bad_bundle_writes_nothing(tmp_path,failure):
    item=fixture(tmp_path);archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:
        if failure!='missing':
            info=zipfile.ZipInfo('weights.bin')
            if failure=='symlink':info.external_attr=0o120777 << 16
            z.writestr(info,b'wrong weights' if failure=='checksum' else b'fixed weights')
        if failure=='extra':z.writestr('../escape','x')
        if failure=='duplicate':
            with pytest.warns(UserWarning):z.writestr('weights.bin',b'fixed weights')
    with pytest.raises(ValueError):bundle.install(tmp_path,archive)
    assert not (tmp_path/'artifacts').exists()


def test_distribution_manifest_matches_frozen_adapter_contract():
    manifest=json.loads((ROOT/bundle.MANIFEST).read_text(encoding='utf-8'))
    config=json.loads((ROOT/'configs/stage3-candidate.json').read_text(encoding='utf-8'))
    items={i['destination']:i for i in manifest['files']}
    for name,expected in config['adapter_files'].items():
        assert items[config['adapter']+'/'+name]['sha256']==expected
    assert 'artifacts/stage-01/encoder-clean/best.pt' in items
    assert not any(i['member'].startswith(('data/','models/')) for i in manifest['files'])


def test_source_bundle_excludes_local_data_and_credentials(tmp_path):
    spec=importlib.util.spec_from_file_location('source_bundle',ROOT/'scripts/prepare_source_release.py')
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    for name in source.FILES:
        (tmp_path/name).write_text('public',encoding='utf-8')
    for name in ['src/echolens/app.py','data/reviews.jsonl','artifacts/model.bin','models/config.json',
                 '.env','src/echolens/__pycache__/app.py','configs/.env']:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('test',encoding='utf-8')
    output=tmp_path/'release.zip';source.build(tmp_path,output)
    with zipfile.ZipFile(output) as archive:
        names=archive.namelist()
    assert 'src/echolens/app.py' in names
    assert not any(name.startswith(('data/','models/','artifacts/')) or '.env' in name or '__pycache__' in name for name in names)
