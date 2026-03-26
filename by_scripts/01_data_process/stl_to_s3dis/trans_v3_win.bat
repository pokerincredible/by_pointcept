@echo off
setlocal

REM Windows convenience wrapper for trans_v3_win.ps1
REM Run: trans_v3_win.bat

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0trans_v3_win.ps1"

endlocal

