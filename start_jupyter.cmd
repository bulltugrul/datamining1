@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Proje ortami bulunamadi. Once setup_environment.ps1 dosyasini calistirin.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "scripts\launch_jupyter.py"
set "launchResult=%errorlevel%"
if not "%launchResult%"=="0" pause
exit /b %launchResult%
