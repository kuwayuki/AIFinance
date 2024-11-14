import sys
import utils
import output_csv

tickers = [ticker.strip().upper() for ticker in sys.argv[1].split(',')]
output_csv.mains(tickers)
utils.g_spread_notice(is_buy=True)

