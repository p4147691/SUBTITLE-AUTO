@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
"%~dp0.venv-qwen\Scripts\python.exe" "%~dp0subtitle_client.py" whisper "%~1"
if errorlevel 1 pause
