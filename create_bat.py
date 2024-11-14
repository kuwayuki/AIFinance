import utils

mark_arrays = utils.g_spread_read()

bat_content = f"""cd /d %~dp0
python .\main_CAN_SLIM.py "{mark_arrays}"
"""

with open("aifinance_custom.bat", "w", encoding="utf-8") as bat_file:
        bat_file.write(bat_content)