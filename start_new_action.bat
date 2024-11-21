cd /d %~dp0
pip install yfinance -U
python .\create_bat.py
call aifinance_custom.bat
