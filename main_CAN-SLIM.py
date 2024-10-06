import os
from datetime import datetime
import re
import json
import sys
import utils
import get_news
import output_csv
# プロンプトをインポート
from prompts import PROMPT_CAN_SLIM_SYSTEM, PROMPT_CAN_SLIM_USER

tickers = sys.argv[1].split(',') if len(sys.argv) > 1 else ['RLX']  # 複数ティッカーをカンマ区切りで受け取る
folder_path = ''

def main(tickers):
    global folder_path
    folder_path = utils.create_folder_path(','.join(tickers))

    # 1. CAN-SLIM法で選定された銘柄を配列指定
    can_slim_tickers = tickers
    # can_slim_tickers = filter_can_slim(tickers)

    # 2. 上方チャネルラインから売買価格を判断
    get_buy_sell_prices(can_slim_tickers)

# CAN-SLIM法の銘柄から直近のニュースを考慮して銘柄を絞り込む
def filter_can_slim(tickers):
    file_path = os.path.join(folder_path, 'research.csv')
    output_csv.mains(tickers, file_path)
    
    prompt = PROMPT_CAN_SLIM_USER.format(
        current_date=datetime.now(),
        news='なし',
        # news=utils.read_news_from_csv('./csv/news_data.csv'),
        research=utils.read_news_from_csv(file_path, 'shift_jis', "ALL"),
    )
    print(prompt)
    return utils.get_ai_opinion(prompt, PROMPT_CAN_SLIM_SYSTEM)

def get_buy_sell_prices(tickers):
    for ticker in tickers:
        # industry, sector = utils.get_industry_tickers(ticker)
        # if industry and sector:
        #     top_3_stocks = utils.get_top_3_stocks_by_industry_and_sector(industry, sector)
        #     # top_3_stocks = utils.get_top_3_stocks_by_industry_and_sector(industry, sector)
        #     print(f"業界 '{industry}', セクター '{sector}' のトップ3銘柄: {top_3_stocks}")
        if not utils.filter_can_slim(ticker):
            print("CAN-SLIMの条件を満たしません")
        else:
            utils.get_buy_sell_price(ticker)
        # print(f"{ticker} の買い価格: {buy_price}, 売り価格: {sell_price}")

if __name__ == "__main__":
    main(tickers)

