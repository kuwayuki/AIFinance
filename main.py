import os
from datetime import datetime
import re
import json
import sys
import utils
import get_news
import output_csv
# プロンプトをインポート
from prompts import PROMPT_PROMISING_GROW, PROMPT_BASE_ALL, PROMPT_BASE_DETAIL, PROMPT_BASE_SHORT, PROMPT_SYSTEM_BASE, PROMPT_RELATIONS_CUT, PROMPT_BASE_PROMISING,PROMPT_PROMISING_FUTURE, PROMPT_SYSTEM_GROW

# https://developer.ft.com/portal/docs-start-commence-making-requests

config = utils.load_config()
tickers = sys.argv[1].split(',') if len(sys.argv) > 1 else ['SMH']  # 複数ティッカーをカンマ区切りで受け取る

IS_COMP = config.get("IS_COMP", False)
comp_ticker = config.get("comp_ticker", "VOO")
HISTORY_DAYS = config.get("HISTORY_DAYS", 108)
IS_INCLUDE_HISTORY = config.get("IS_INCLUDE_HISTORY", True)
FORCE_GET = config.get("FORCE_GET", False)
IS_DETAIL = config.get("IS_DETAIL", True)

# IS_ALLの設定
IS_ALL = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False  # "true"の場合True、それ以外はFalse
IS_SHORT_CONTINUE_DETAIL = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else False  # "true"の場合True、それ以外はFalse

# GoogleニュースのRSSフィードURL
NEWS_URL = "https://news.google.com/news/rss/headlines/section/topic/BUSINESS?hl=ja&gl=JP&ceid=JP:ja"

# メイン処理
def main_action(ticker, force_news="", is_short_continue_detail=False):
    is_short = ticker == "SHORT"
    # force_news が指定されている場合、us_news を取得しない
    if not force_news:
        if is_short:
            us_news = utils.get_news_google(2)
            print(us_news)
        else:
            us_news = utils.get_news_google()
    else: 
        us_news = force_news

    # フォルダの作成
    date_str = datetime.now().strftime('%Y%m%d')
    time_str = datetime.now().strftime('%H%M')
    folder_path = f'./history/{date_str}/{ticker}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    historical_section = ""
    if not is_short:
        historical_data = utils.load_history_data(ticker, FORCE_GET)
        json_file_path = os.path.join(folder_path, 'etf_extra_data.json')
        json_data = utils.load_json_data(json_file_path)

        # JSONデータをプロンプト用に整形
        json_section = "\n\n追加情報:\n" + json.dumps(json_data, indent=2)

        if HISTORY_DAYS != 0:
            if IS_INCLUDE_HISTORY:
                historical_section = f"\n過去{HISTORY_DAYS}日間のデータも考慮していますが、直近1ヶ月のデータには特に注目しています。"
                historical_section += f"\n過去データの一部: {historical_data.to_string(index=False)}"

                if ticker != comp_ticker and IS_COMP:
                    comp_historical_data = utils.comp_load(ticker)
                    historical_section += f"\n\n比較用銘柄<{comp_ticker}>の一部: {comp_historical_data.to_string()}"
                # historical_section += f"\n過去データの一部: {historical_data.tail(5).to_string(index=False)}"

    if is_short:
        print('SHORT!')
        comp_historical_data = utils.comp_load(ticker)
        historical_section += f"\n\n比較用銘柄<{comp_ticker}>の一部: {comp_historical_data.to_string()}"
        explane = utils.process_ai_opinion(False, ticker, us_news, "", "", PROMPT_BASE_SHORT, os.path.join(folder_path, f'result_detail_{time_str}.txt'))

        if is_short_continue_detail:
            quotes_list = re.findall(r'「(.*?)」', explane)
            print(quotes_list)
            for quote in quotes_list:
                main_action(quote, us_news, False)  # それぞれのquoteをtickerとして使用
        return
    if IS_ALL:
        utils.process_ai_opinion(True, ticker, us_news, historical_section, json_section, PROMPT_BASE_ALL, os.path.join(folder_path, f'result_all_{time_str}.txt'))
        print('\n\n')
    if IS_DETAIL:
        # process_ai_opinion(False, ticker, us_news, historical_section, json_section, PROMPT_BASE_PROMISING, os.path.join(folder_path, f'result_detail_{time_str}.txt'))
        utils.process_ai_opinion(False, ticker, us_news, historical_section, json_section, PROMPT_BASE_DETAIL, os.path.join(folder_path, f'result_detail_{time_str}.txt'))

def future(ticker, is_include_history_data = False, is_grow = False, file_path = os.path.join(f'./csv/', 'research.csv')):
    historical_section = ""
    if is_include_history_data:
        historical_data = utils.load_history_data(ticker, False)
        historical_data = utils.data_filter_date(historical_data, HISTORY_DAYS)
        historical_section += f"\n過去{HISTORY_DAYS}日間のデータも考慮していますが、直近1ヶ月のデータには特に注目しています。"
        historical_section += f"\n{historical_data.to_string(index=False)}"

    prompt_base = PROMPT_PROMISING_FUTURE
    prompt_system = PROMPT_SYSTEM_BASE
    if is_grow:
        prompt_base = PROMPT_PROMISING_GROW
        prompt_system = PROMPT_SYSTEM_GROW

    csv_ticer = ticker
    if is_grow:
        csv_ticer = "ALL"
        
    prompt = prompt_base.format(
        ticker=ticker,
        historical_section=historical_section,
        current_date=datetime.now(),
        news=utils.read_news_from_csv('./csv/news_data.csv'),
        research=utils.read_news_from_csv(file_path, 'shift_jis', csv_ticer),
    )
    # return prompt
    ai_opinion = utils.get_ai_opinion(prompt, prompt_system)
    ai_opinion_cleaned = re.sub(r'[\*\#\_]+', '', ai_opinion)
    utils.send_line_notify(ai_opinion_cleaned)

    date_str = datetime.now().strftime('%Y%m%d')
    time_str = datetime.now().strftime('%H%M')
    folder_path = f'./history/{date_str}/{ticker}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    try:
        with open(os.path.join(folder_path, f'research_{time_str}.txt'), 'w', encoding='utf-8') as file:
            file.write(ai_opinion_cleaned)
    except Exception as e:
        print(f"ファイルの保存中にエラーが発生しました: {e}")


def main():
    is_future = sys.argv[2] == "FUTURE" if len(sys.argv) > 2 else False 
    is_grow = sys.argv[2] == "GROW" if len(sys.argv) > 2 else False 
    if is_future or is_grow:
        # TODO:トークン量がえげつないので一旦コメントアウト
        # get_news.main(use_latest_csv_date=True)
        print("トークン量がえげつないので一旦コメントアウト")
    else:
        # Google News
        soup = utils.fetch_news_soup(NEWS_URL)
        news_headlines = utils.fetch_news_titles(soup, True)
        print(news_headlines)
        news_all_relations = utils.get_news_all_relations(soup)
        # yahoo_headlines = fetch_news_titles(soup, "Yahoo!ファイナンス")
        print(news_all_relations)

    if is_grow:
        date_str = datetime.now().strftime('%Y%m%d')
        tickers_str = ','.join(tickers)
        folder_path = f'./history/{date_str}/{tickers_str}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, 'research.csv')
        output_csv.mains(tickers, file_path)
        future(tickers_str, False, True, file_path)
    else:
        for ticker in tickers:
            if is_future:
                # python main.py "NVDA" "FUTURE"
                output_csv.main(ticker)
                future(ticker, False)
                # 過去データも含める場合
                # future(ticker, True)
            else:
                # python main.py "NVDA"
                main_action(ticker, news_all_relations, IS_SHORT_CONTINUE_DETAIL)
            # main(ticker)import yfinance as yf

# 複数のティッカーに対してループ処理
if __name__ == "__main__":
    main()