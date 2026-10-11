$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& "$projectRoot\.venv-stage1\Scripts\python.exe" -m streamlit run app/annotation_app.py --server.address 127.0.0.1 --server.port 8502
