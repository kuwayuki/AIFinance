@echo off
cd /d %~dp0

set hours=5
set /a wait_time=%hours% * 3600

start "API Server" /min python api_server.py

rem 30���i1800�b�j�ҋ@
timeout /t %wait_time% /nobreak

rem Python�v���Z�X���I��������
taskkill /f /im python.exe
