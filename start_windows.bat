@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  start "" pyw -3 launch_windows.pyw
) else (
  start "" pythonw launch_windows.pyw
)
