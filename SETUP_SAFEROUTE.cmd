@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python "tools\setup_environment.py"
) else (
  py -3.12 "tools\setup_environment.py"
)
if errorlevel 1 (
  echo Setup failed. Check Python 3.12 and network access.
  pause
  exit /b 1
)
pause
