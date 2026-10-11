"""One background GPU job at a time; Streamlit never holds a GPU model."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import psutil
from echolens.experiments import ROOT
from echolens.workbench import dataset_path, read_json, atomic_json
from echolens.runtime import python_executable, prepare_runtime_assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    folder = dataset_path(args.dataset)
    lock = ROOT / "artifacts/stage-02/gpu-job.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        pid = int(lock.read_text())
        if psutil.pid_exists(pid):
            raise RuntimeError("已有分析任务正在运行")
        lock.unlink()
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        candidate = read_json(folder/'dataset.json').get('analysis_config')=='configs/stage3-candidate.json'
        if candidate:
            prepare_runtime_assets()
        steps = [(sys.executable,'analyze_stage2.py',['--classification-only'] if candidate else [])]
        if candidate:
            steps.append((python_executable('inference'),'analyze_stage3.py',[]))
        steps.append((sys.executable,'topics_stage2.py',[]))
        for python,script,extra in steps:
            subprocess.run([python, "-u", str(ROOT / "scripts" / script), "--dataset", args.dataset,*extra], cwd=ROOT, check=True)
        atomic_json(folder / "job.json", {"status": "complete", "pid": os.getpid()})
    except Exception as exc:
        atomic_json(folder / "job.json", {"status": "failed", "error": str(exc), "pid": os.getpid()})
        atomic_json(folder / "progress.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
