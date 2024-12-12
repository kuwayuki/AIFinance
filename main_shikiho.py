import sys
import utils

tickers = [ticker.strip().upper() for ticker in sys.argv[1].split(',')] if len(sys.argv) > 1 else ['AAPL']
tickers = utils.ensure_t_suffix(tickers)

def main(tickers):
    # 事前準備：四季報で目星のティッカーシンボルを確認しておく

    # 作業共通フォルダを作成
    # yfinanceを確認し、点数と満たしているかの確認を行う
    for ticker in tickers:
        print(utils.shikiho(ticker))
        print('&')

if __name__ == "__main__":
    main(tickers)

