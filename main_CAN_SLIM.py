import os
from datetime import datetime
import re
import json
import sys
import utils
import time
import get_news
import main as MainPy
import output_csv
# プロンプトをインポート
from prompts import PROMPT_CAN_SLIM_SYSTEM, PROMPT_CAN_SLIM_USER

# 複数ティッカーをカンマ区切りで受け取り、空白を削除し大文字に変換
tickers = [ticker.strip().upper() for ticker in sys.argv[1].split(',')] if len(sys.argv) > 1 else ['AAPL']
tickers = utils.ensure_t_suffix(tickers)

folder_path = ''
is_future = sys.argv[2] != '' if len(sys.argv) > 3 else False # 未来予測も確認
WATCH_COUNT = 6
TIME_MINUTE = 60
TIME_MINUTE_INIT = 10

START_TIME_USA = (23, 30)
START_TIME_JAPAN = (9, 0)

def should_run_now(is_usa):
    now = datetime.now()
    current_time = (now.hour, now.minute)
    if is_usa:
        return current_time <= START_TIME_USA
    else:
        return current_time <= START_TIME_JAPAN

def main(tickers, is_output_all_info = False, is_send_line = False, is_write_g_spread = True):
    global folder_path

    utils.move_bkup_folder()
    is_usa = utils.set_Sheet_name(tickers)

    # 1. CAN-SLIM法で選定された銘柄を配列指定
    # can_slim_tickers = filter_can_slim(tickers)
    can_slim_tickers = tickers
    output_csv.mains(tickers)

    # 2. 各手法で売買価格を判断
    get_buy_sell_prices(can_slim_tickers, is_output_all_info, is_send_line, is_write_g_spread)

    utils.g_spread_copy_columns()

    # アメリカは22:00-06:30、日本は08:00-15:30以外は監視しない
    if utils.is_in_time_range(is_usa):
        # CSVデータをスプレッドシートにコピー
        # utils.g_spread_write_data_multi(tickers)

        # マークがついているもののみ全て評価
        mark_arrays = utils.g_spread_notice()
        mark_tickers = [item[0] for item in mark_arrays if item[0] in tickers]

        for i in range(WATCH_COUNT):
            print(f"{i+1}回目の実行")

            # 初回は開始時間になるまで待機
            if i == 0:
                # 指定された時間になるまで待機
                while should_run_now(is_usa):
                    print("指定された時刻ではないため待機中...")
                    time.sleep(180)  # 3分待機

            # マークがついているものから直近で動きがありそうなもののみ、一定時間実行
            spread_arrays = utils.g_spread_notice(False)
            future_tickers = [item[0] for item in spread_arrays if item[0] in mark_tickers]

            # CSVの現在価格のみ更新
            utils.get_current_price_multi(future_tickers, False)
            # utils.get_current_price_multi(future_tickers, is_usa)
            # スプレッドシートデータ更新
            utils.g_spread_write_data_multi(future_tickers)
            # 倍率の厳密チェック
            utils.g_spread_notice(is_buy=True)

            if i < (WATCH_COUNT - 1):
                print(f"{TIME_MINUTE}分待機中...")
                time.sleep(TIME_MINUTE * 60) 

# CAN-SLIM法の銘柄から直近のニュースを考慮して銘柄を絞り込む
def filter_can_slim(tickers):
    folder_path = utils.create_folder_path(','.join(tickers))
    file_path = os.path.join(folder_path, 'research.csv')
    output_csv.mains(tickers, file_path)
    
    prompt = PROMPT_CAN_SLIM_USER.format(
        current_date=datetime.now(),
        news='なし',
        # news=utils.read_news_from_csv('./csv/news_data.csv'),
        research=utils.read_news_from_csv(file_path, 'shift_jis', "ALL"),
    )
    return utils.get_ai_opinion_gemini(prompt)
    # return utils.get_ai_opinion(prompt, PROMPT_CAN_SLIM_SYSTEM)

def get_buy_sell_prices(tickers, is_output_all_info = False, is_send_line = False, is_write_g_spread = False):
    for ticker in tickers:
        # industry, sector = utils.get_industry_tickers(ticker)
        # if industry and sector:
        #     top_3_stocks = utils.get_top_3_stocks_by_industry_and_sector(industry, sector)
        #     # top_3_stocks = utils.get_top_3_stocks_by_industry_and_sector(industry, sector)
        #     print(f"業界 '{industry}', セクター '{sector}' のトップ3銘柄: {top_3_stocks}")
        # utils.get_buy_sell_price(ticker)
        try:
            # CSVデータをスプレッドシートに更新
            utils.g_spread_write_data(ticker)

            if is_output_all_info:
                utils.set_output_log_file_path(ticker, 'all_info', True)
                utils.all_print(ticker)

            utils.set_output_log_file_path(ticker, 'can_slim', True)
            utils.analyst_eval_send(ticker, is_write_g_spread)

            utils.output_log(f"\n{ticker} Start")

            score, failed_conditions = utils.filter_can_slim(ticker)
            if score > 2:
                if is_future:
                    MainPy.future(ticker, False)

                utils.get_buy_sell_price(ticker)
                if is_send_line:
                    utils.send_line_log_text(False)
            else:
                utils.output_log(f"評価が低いので確認しません。")
            # print(f"{ticker} の買い価格: {buy_price}, 売り価格: {sell_price}")
        except Exception as e:
            print(e)

        finally:
            utils.output_log(f"\n{ticker} End")
            utils.set_output_log_file_path(is_clear=True)

if __name__ == "__main__":
    main(tickers)

