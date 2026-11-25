@echo off
setlocal
cd /d "%~dp0"

title SafeRoute - T09 P01 National Service

echo ======================================
echo SafeRoute National Service
echo Model: T09
echo Policy: P01
echo ======================================
echo.

if not exist "runtime\node.exe" (
    echo [ERROR] Node.js runtime not found.
    pause
    exit /b 1
)

if not exist "service\server.js" (
    echo [ERROR] SafeRoute server not found.
    pause
    exit /b 1
)

if not exist "service\data\national_manifest.json" (
    echo [ERROR] National manifest not found.
    pause
    exit /b 1
)

if not exist "service\data\tiles\" (
    echo [ERROR] National tile directory not found.
    pause
    exit /b 1
)

set "SAFEROUTE_NO_BROWSER="
set "KAKAO_REST_API_KEY="
set "SAFEROUTE_PORT=8090"

echo Starting SafeRoute...
echo Enter API keys in the browser.
echo Stop the server with Ctrl+C.
echo.

"runtime\node.exe" "service\server.js"

set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo Server stopped. Exit code: %EXIT_CODE%
pause

exit /b %EXIT_CODE%
