"""Local runtime entry points. No model imports and no automatic downloads."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ASSETS = {'prompt':'configs/prompts/restaurant-v0.1.txt', 'aspects':'configs/labels/restaurant-v0.1.json'}


def python_executable(role, root=ROOT):
    variable = 'ECHOLENS_INFERENCE_PYTHON' if role=='inference' else 'ECHOLENS_APP_PYTHON'
    override = os.environ.get(variable)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise ValueError(f'{variable}指向的解释器不存在：{path}')
        return str(path.resolve())
    environment = '.venv' if role=='inference' else '.venv-stage1'
    local = root/environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    return str(local if local.is_file() else Path(sys.executable))


def prepare_runtime_assets(root=ROOT):
    """Restore only missing non-private assets to the frozen worker's paths."""
    config = json.loads((root/'configs/stage3-candidate.json').read_text(encoding='utf-8'))
    pending = []
    for name,source in ASSETS.items():
        data = (root/source).read_bytes()
        expected = config[name+'_sha256']
        if hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError(f'公开{name}文件与冻结版本不一致')
        target = root/config[name+'_path']
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'本地{name}文件已改变，拒绝覆盖：{target}')
        else:
            pending.append((target,data))
    for target,data in pending:
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)


def missing_analysis_assets(folder, root=ROOT):
    metadata = json.loads((folder/'dataset.json').read_text(encoding='utf-8'))
    models = json.loads((root/'configs/models.json').read_text(encoding='utf-8'))
    paths = ['artifacts/stage-01/encoder-clean/best.pt',
             models['generation']['local_dir']+'/config.json',
             models['embedding']['local_dir']+'/config.json']
    if not (folder/'classifications.json').exists():
        encoder = json.loads((root/'configs/stage1.json').read_text(encoding='utf-8'))['encoder']
        paths.append(encoder['local_dir']+'/config.json')
    if metadata.get('analysis_config')=='configs/stage3-candidate.json':
        config = json.loads((root/'configs/stage3-candidate.json').read_text(encoding='utf-8'))
        paths += [config['adapter']+'/'+name for name in config['adapter_files']]
        paths += list(ASSETS.values())
    return [p for p in paths if not (root/p).is_file()]


def start_analysis(folder):
    from echolens.workbench import atomic_json
    prepare_runtime_assets()
    missing = missing_analysis_assets(folder)
    if missing:
        raise ValueError('缺少分析资源：'+'、'.join(missing))
    python_executable('inference')  # fail before starting a child for an invalid override
    with (folder/'worker.log').open('a',encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable,str(ROOT/'scripts/run_workbench_job.py'),'--dataset',folder.name],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    atomic_json(folder/'job.json',{'status':'running','pid':process.pid})
    return process.pid
