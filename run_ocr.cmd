@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title SUBTITLE-AUTO OCR
"%~dp0.venv-qwen\Scripts\python.exe" "%~dp0ocr_client.py" "%~1" --mode speed
if errorlevel 1 pause
