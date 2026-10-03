$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = (Get-Location).Path
$python = "C:\Users\pooja\AppData\Local\Programs\Python\Python311\python.exe"
& $python -m streamlit run frontend/app.py
