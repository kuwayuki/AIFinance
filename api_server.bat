@echo off
cd /d %~dp0

set hours=5
set /a wait_time=%hours% * 3600

start "API Server" /min python api_server.py

rem 30分（1800秒）待機
timeout /t %wait_time% /nobreak

rem Pythonプロセスを終了させる
taskkill /f /im python.exe
