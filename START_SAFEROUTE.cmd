@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python environment not found.
    echo Please run SETUP_SAFEROUTE.cmd first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "tools\launch.py"

pause