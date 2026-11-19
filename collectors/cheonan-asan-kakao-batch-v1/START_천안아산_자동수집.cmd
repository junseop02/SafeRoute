@echo off
chcp 65001 >nul
title 천안·아산 카카오내비 자동 수집기
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
if errorlevel 1 pause
