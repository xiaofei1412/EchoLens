"""Start the local UI with an explicit runtime; never install or download assets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from echolens.runtime import python_executable, prepare_runtime_assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true',help='Print runtime paths without starting the server')
    parser.add_argument('--port',type=int,default=8501)
    args = parser.parse_args()
    runtimes = {role:python_executable(role) for role in ['app','inference']}
    if args.check:
        print(json.dumps(runtimes,ensure_ascii=False,indent=2))
        return
    prepare_runtime_assets()
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src') + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    raise SystemExit(subprocess.call([runtimes['app'],'-m','streamlit','run',str(ROOT/'app/streamlit_app.py'),
        '--server.address','127.0.0.1','--server.port',str(args.port)],cwd=ROOT,env=env))


if __name__=='__main__':
    main()
