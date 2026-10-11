import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from streamlit.testing.v1 import AppTest
from echolens import runtime, workbench as w
from echolens.workbench_catalog import visible_datasets, preferred_dataset


def test_runtime_selection_and_invalid_explicit_override(tmp_path,monkeypatch):
    monkeypatch.delenv('ECHOLENS_INFERENCE_PYTHON',raising=False)
    assert runtime.python_executable('inference',tmp_path)==sys.executable
    local = tmp_path/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    local.parent.mkdir(parents=True)
    local.touch()
    assert runtime.python_executable('inference',tmp_path)==str(local)
    monkeypatch.setenv('ECHOLENS_INFERENCE_PYTHON',str(tmp_path/'missing'))
    with pytest.raises(ValueError,match='解释器不存在'):
        runtime.python_executable('inference',tmp_path)


def copy_configs(root):
    shutil.copytree(w.ROOT/'configs',root/'configs')


def test_public_assets_restore_exactly_without_overwriting(tmp_path):
    copy_configs(tmp_path)
    runtime.prepare_runtime_assets(tmp_path)
    config = w.read_json(tmp_path/'configs/stage3-candidate.json')
    paths = [tmp_path/config[n+'_path'] for n in runtime.ASSETS]
    before = [p.stat().st_mtime_ns for p in paths]
    runtime.prepare_runtime_assets(tmp_path)
    assert [p.stat().st_mtime_ns for p in paths]==before
    for name in runtime.ASSETS:
        assert hashlib.sha256((tmp_path/config[name+'_path']).read_bytes()).hexdigest()==config[name+'_sha256']
    paths[0].write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='拒绝覆盖'):
        runtime.prepare_runtime_assets(tmp_path)
    assert paths[0].read_text()=='changed'


def test_history_filter_keeps_real_candidate_visible():
    items = [dict(dataset_id=i,kind=k,analysis_config=c) for i,k,c in [
        ('asap-demo-1000','train_demo','configs/stage2.json'),
        ('asap-demo-1000-v3','train_demo','configs/stage2-v3.json'),
        ('probe','integration_probe','configs/stage3-candidate.json'),
        ('user','uploaded','configs/stage3-candidate.json')]]
    assert [m['dataset_id'] for m in visible_datasets(items)]==['asap-demo-1000-v3','user']
    assert preferred_dataset(visible_datasets(items))=='user'
    assert visible_datasets(items,True)==items


def test_empty_workbench_and_unavailable_model(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'STORE',tmp_path/'store')
    app = AppTest.from_file(str(w.ROOT/'app/streamlit_app.py')).run()
    assert not app.exception and app.title[0].value=='数据导入'
    assert '报告导出' in app.sidebar.radio[0].options
    raw = tmp_path/'feedback.jsonl'
    raw.write_text('{"id":"1","text":"汤太咸。"}',encoding='utf-8')
    source,_ = w.import_upload(raw,{'source_record_id':'id','text_raw':'text'},'test','v1','synthetic')
    monkeypatch.setattr(runtime,'missing_analysis_assets',lambda folder:['models/not-installed/config.json'])
    app.sidebar.radio[0].set_value('分析概览').run()
    assert not app.exception
    assert next(b for b in app.button if b.label=='开始 / 继续本地分析').disabled
    assert any('资源尚未就绪' in x.value for x in app.warning)
    app.sidebar.radio[0].set_value('主题与证据').run()
    next(r for r in app.radio if r.label=='查看内容').set_value('原文检索').run()
    assert not app.exception and any('原文 · 1' in x.value for x in app.subheader)
    app.sidebar.radio[0].set_value('报告导出').run()
    assert not app.exception
    assert not w.events(w.dataset_path(source))


def test_hidden_engineering_dataset_does_not_break_selection(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'STORE',tmp_path)
    raw = tmp_path/'feedback.jsonl'
    raw.write_text('{"id":"1","text":"味道好。"}',encoding='utf-8')
    source,_ = w.import_upload(raw,{'source_record_id':'id','text_raw':'text'},'test','v1','synthetic')
    folder=w.dataset_path(source)
    meta=w.read_json(folder/'dataset.json');meta['kind']='integration_probe'
    w.atomic_json(folder/'dataset.json',meta)
    app=AppTest.from_file(str(w.ROOT/'app/streamlit_app.py')).run()
    assert not app.sidebar.selectbox and not app.exception
    app.sidebar.checkbox[0].check().run()
    assert app.sidebar.selectbox[0].value==source
    app.sidebar.checkbox[0].uncheck().run()
    assert not app.exception and not app.sidebar.selectbox


def test_clean_checkout_ui_has_no_private_data_dependency(tmp_path):
    for name in ['app','src','configs','.streamlit']:
        shutil.copytree(w.ROOT/name,tmp_path/name,ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    command = "from streamlit.testing.v1 import AppTest; a=AppTest.from_file('app/streamlit_app.py').run(); assert not a.exception; assert a.title[0].value=='数据导入'; print('clean checkout UI passed')"
    env={**os.environ,'PYTHONPATH':str(tmp_path/'src'),'PYTHONUTF8':'1'}
    result=subprocess.run([sys.executable,'-c',command],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert not (tmp_path/'data').exists() and not (tmp_path/'models').exists()


@pytest.mark.parametrize('fail',[False,True])
def test_candidate_dispatch_uses_selected_runtime_and_releases_lock(tmp_path,monkeypatch,fail):
    spec=importlib.util.spec_from_file_location('release_job',w.ROOT/'scripts/run_workbench_job.py')
    job=importlib.util.module_from_spec(spec);spec.loader.exec_module(job)
    folder=tmp_path/'dataset';folder.mkdir()
    w.atomic_json(folder/'dataset.json',{'analysis_config':'configs/stage3-candidate.json'})
    monkeypatch.setattr(job,'ROOT',tmp_path)
    monkeypatch.setattr(job,'dataset_path',lambda value:folder)
    monkeypatch.setattr(job,'prepare_runtime_assets',lambda:None)
    monkeypatch.setattr(job,'python_executable',lambda role:'selected-inference-python')
    monkeypatch.setattr(sys,'argv',['run_workbench_job.py','--dataset','sample'])
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if fail and len(calls)==2:
            raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(job.subprocess,'run',run)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            job.main()
        assert w.read_json(folder/'progress.json')['status']=='failed'
    else:
        job.main()
    assert w.read_json(folder/'job.json')['status']==('failed' if fail else 'complete')
    assert [Path(c[2]).name for c in calls]==['analyze_stage2.py','analyze_stage3.py']+([] if fail else ['topics_stage2.py'])
    assert calls[0][-1]=='--classification-only'
    assert calls[1][0]=='selected-inference-python'
    assert not (tmp_path/'artifacts/stage-02/gpu-job.lock').exists()


def test_launcher_passes_checkout_sources_to_child(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('release_launcher',w.ROOT/'scripts/start_workbench.py')
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    monkeypatch.setattr(launcher,'prepare_runtime_assets',lambda:None)
    monkeypatch.setattr(launcher,'python_executable',lambda role:sys.executable)
    monkeypatch.setattr(sys,'argv',['start_workbench.py','--port','8502'])
    monkeypatch.setenv('PYTHONPATH','existing-source')
    captured={}
    def call(command,**kwargs):
        captured.update(command=command,**kwargs)
        return 0
    monkeypatch.setattr(launcher.subprocess,'call',call)
    with pytest.raises(SystemExit) as exited:
        launcher.main()
    assert exited.value.code==0
    assert captured['cwd']==tmp_path
    assert captured['command'][-1]=='8502'
    assert captured['env']['PYTHONPATH']==str(tmp_path/'src')+os.pathsep+'existing-source'
