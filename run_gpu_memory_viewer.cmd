@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
"%~dp0.venv-qwen\Scripts\python.exe" "%~dp0gpu_memory_viewer.py"
if errorlevel 1 pause
