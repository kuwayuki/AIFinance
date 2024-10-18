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

tickers = sys.argv[1].split(',') if len(sys.argv) > 1 else ['IMMR']  # 複数ティッカーをカンマ区切りで受け取る
folder_path = ''

def main(tickers):
    global folder_path
    folder_path = utils.create_folder_path(','.join(tickers))

    # 1. CAN-SLIM法で選定された銘柄を配列指定
    # can_slim_tickers = filter_can_slim(tickers)
    can_slim_tickers = tickers

    # 2. 各手法で売買価格を判断
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
    return utils.get_ai_opinion(prompt, PROMPT_CAN_SLIM_SYSTEM)

def get_buy_sell_prices(tickers, is_output_all_info = False):
    for ticker in tickers:
        # industry, sector = utils.get_industry_tickers(ticker)
        # if industry and sector:
        #     top_3_stocks = utils.get_top_3_stocks_by_industry_and_sector(industry, sector)
        #     # top_3_stocks = utils.get_top_3_stocks_by_industry_and_sector(industry, sector)
        #     print(f"業界 '{industry}', セクター '{sector}' のトップ3銘柄: {top_3_stocks}")
        # utils.get_buy_sell_price(ticker)
        try:

            if is_output_all_info:
                utils.set_output_log_file_path(ticker, 'all_info', True)
                utils.all_print(ticker)

            utils.set_output_log_file_path(ticker, 'can_slim', True)
            utils.output_log(f"\n★★★{ticker} Start★★★")

            score, failed_conditions = utils.filter_can_slim(ticker)
            if score > 2:
                utils.get_buy_sell_price(ticker)
                utils.send_line_log_text()
            else:
                utils.output_log(f"評価が低いので確認しません。")
            # print(f"{ticker} の買い価格: {buy_price}, 売り価格: {sell_price}")

        finally:
            utils.output_log(f"\n★★★{ticker} End★★★")
            utils.set_output_log_file_path(is_clear=True)

if __name__ == "__main__":
    main(tickers)

