cd /d %~dp0
git pull
pip install yfinance -U
python .\create_bat.py
call aifinance_custom.bat
