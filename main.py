import yfinance as yf
import pandas as pd
import openai
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import json
from newsapi import NewsApiClient
import csv
import sys
import get_news
import ouptut_csv
# プロンプトをインポート
from prompts import PROMPT_BASE_ALL, PROMPT_BASE_DETAIL, PROMPT_BASE_SHORT, PROMPT_SYSTEM_BASE, PROMPT_RELATIONS_CUT, PROMPT_BASE_PROMISING,PROMPT_PROMISING_FUTURE
# import fredapi

# https://developer.ft.com/portal/docs-start-commence-making-requests

CONFIG_FILE_PATH = "./config/config.json"
def load_config(config_path):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config

def load_ticker_urls(json_file_path):
    with open(json_file_path, 'r') as json_file:
        ticker_urls = json.load(json_file)
    return ticker_urls
ticker_urls = load_ticker_urls('./config/ticker_urls.json')


config = load_config(CONFIG_FILE_PATH)
tickers = sys.argv[1].split(',') if len(sys.argv) > 1 else ['SMH']  # 複数ティッカーをカンマ区切りで受け取る

API_KEY = config.get("API_KEY", "")
openai.api_key = API_KEY
OSUSUME = config.get("OSUSUME", [])
IS_COMP = config.get("IS_COMP", False)
comp_ticker = config.get("comp_ticker", "VOO")
HISTORY_DAYS = config.get("HISTORY_DAYS", 108)
IS_INCLUDE_HISTORY = config.get("IS_INCLUDE_HISTORY", True)
FORCE_GET = config.get("FORCE_GET", False)
IS_DETAIL = config.get("IS_DETAIL", True)
NEWS_API = config.get("NEWS_API", "")
GPT_MODEL = config.get("GPT_MODEL", "gpt-4o")
NEWS_COUNT = config.get("NEWS_COUNT", 40)

# IS_ALLの設定
IS_ALL = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False  # "true"の場合True、それ以外はFalse
IS_SHORT_CONTINUE_DETAIL = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else False  # "true"の場合True、それ以外はFalse

# FRED_API = '8cf1aece6a69ed3ab3e88d8460390faf'
# fred = fredapi.Fred(api_key=FRED_API)
newsapi = NewsApiClient(api_key=NEWS_API)

# GoogleニュースのRSSフィードURL
NEWS_URL = "https://news.google.com/news/rss/headlines/section/topic/BUSINESS?hl=ja&gl=JP&ceid=JP:ja"

# RSSフィードからニュースヘッドラインを取得
def fetch_news_soup(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'xml')
        return soup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return ""

from datetime import datetime

def fetch_news_titles(soup, detailed=False, filter_keyword=None, num_titles=None, ):
    try:
        items = soup.find_all('item', limit=num_titles)
        titles = []
        seen_times = set()  # 重複した時刻を避けるためのセット

        for item in items:
            # pubDateを取得してフォーマット
            pub_date = item.pubDate.text.strip()
            utc_date = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
            jst_date = utc_date + timedelta(hours=9)  # UTCからJSTに変換
            formatted_date = jst_date.strftime('%Y/%m/%d %H:%M')

            # フィルタリングがある場合は条件に合致するものだけ追加
            if filter_keyword and filter_keyword not in item.description.text:
                continue

            if detailed:
                # description内の重要な部分を抽出
                description = item.description.text.strip()
                soup_description = BeautifulSoup(description, 'html.parser')
                links = soup_description.find_all('a')
                important_info_list = []
                for link in links:
                    text = link.get_text()
                    if len(text) >= 20:  # 文字数が20文字以上かチェック
                        important_info_list.append(text)
                
                # すでに同じ時刻がリストにある場合は、時刻を省略して情報だけ追加
                if formatted_date in seen_times:
                    titles[-1] += ": " + ": ".join(important_info_list)
                else:
                    # 新しい時刻としてリストに追加
                    seen_times.add(formatted_date)
                    titles.append(f"{formatted_date} " + "。".join(important_info_list))
            else:
                if formatted_date not in seen_times:
                    seen_times.add(formatted_date)
                    titles.append(f"{formatted_date} {item.title.text}")

        return "\n".join(titles)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return ""

# ETFデータを保存する関数
def save_etf_data(ticker, start_date, end_date, file_path):
    etf = yf.Ticker(ticker)
    hist = etf.history(start=start_date, end=end_date)
    hist.to_csv(file_path)

    # ティッカーに関する追加情報を取得
    info = etf.info
    financials = etf.financials.to_dict()  # DataFrameを辞書形式に変換
    balance_sheet = etf.balance_sheet.to_dict()
    cashflow = etf.cashflow.to_dict()
    # earnings = etf.earnings.to_dict()

    # keysがTimestampになっている部分を文字列に変換
    def convert_keys_to_str(d):
        return {str(k): v for k, v in d.items()}

    financials = {str(k): convert_keys_to_str(v) for k, v in financials.items()}
    balance_sheet = {str(k): convert_keys_to_str(v) for k, v in balance_sheet.items()}
    cashflow = {str(k): convert_keys_to_str(v) for k, v in cashflow.items()}
    # earnings = etf.earnings.to_dict()

    # 保存するデータをまとめる
    json_data = {
        "info": info,
        "financials": financials,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow,
        # "earnings": earnings
    }

    # # JSONファイル名を生成
    # json_file_path = os.path.splitext(file_path)[0] + "_extra_data.json"# JSON形式で保存
    # with open(json_file_path, 'w') as json_file:
    #     json.dump(json_data, json_file, indent=4)

    def filter_info_data(info):
        keys_to_keep = ['longBusinessSummary', 'marketCap', 'currentPrice', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'trailingPE', 'forwardPE', 'beta', 'dividendRate', 'dividendYield', 'recommendationKey', 'profitMargins', 'grossMargins']
        return {key: info[key] for key in keys_to_keep if key in info}

    def filter_financial_data(financials):
        keys_to_keep = ['Total Revenue', 'Net Income', 'Operating Income', 'EBITDA', 'Gross Profit', 'Diluted EPS']
        return {str(date): {key: data[key] for key in keys_to_keep if key in data} for date, data in financials.items()}

    def filter_balance_sheet_data(balance_sheet):
        keys_to_keep = ['Total Assets', 'Total Liabilities', 'Total Equity']
        return {str(date): {key: data[key] for key in keys_to_keep if key in data} for date, data in balance_sheet.items()}

    def filter_cashflow_data(cashflow):
        keys_to_keep = ['Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditure']
        return {str(date): {key: data[key] for key in keys_to_keep if key in data} for date, data in cashflow.items()}

    # JSONデータから不要な情報を省略する
    filtered_data = {
        "info": filter_info_data(json_data.get("info", {})),
        "financials": filter_financial_data(json_data.get("financials", {})),
        "balance_sheet": filter_balance_sheet_data(json_data.get("balance_sheet", {})),
        "cashflow": filter_cashflow_data(json_data.get("cashflow", {}))
    }

    # JSONファイル名を生成
    json_file_path = os.path.splitext(file_path)[0] + "_extra_data.json"# JSON形式で保存
    with open(json_file_path, 'w') as json_file:
        json.dump(filtered_data, json_file, indent=4)

# JSONファイルからデータを読み込む関数
def load_json_data(json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)
        return json_data
    else:
        return {}

# ETFデータをファイルから読み込む関数
def load_etf_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'])
    return data

# AIの意見を取得
def get_ai_opinion(prompt, prompt_system = PROMPT_SYSTEM_BASE):
    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "system", "content": prompt_system}
        ],
        temperature=0.01
    )

    print(response.usage)
    return response.choices[0].message.content

def send_line_notify(notification_message):
    line_notify_token = 'Z392GKtaICiiiSndtfrKqDmYliv8df5S9AIekyPpSoa'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    
    # メッセージが1000文字を超える場合、分割して送信
    max_length = 1000
    print(notification_message)
    for i in range(0, len(notification_message), max_length):
        chunk = notification_message[i:i + max_length]
        data = {'message': f'{chunk}'}
        requests.post(line_notify_api, headers=headers, data=data)


def get_news_google(fromDate = 3):
    try:
        start_date = (datetime.now() - timedelta(days=fromDate)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')  # 本日の日付を取得
        # news_data = newsapi.get_top_headlines(
        #                         q='finance',
        #                             category='business',
        #                             language='en',
        #                             country='us')
        
        # /v2/everything
        news_data = newsapi.get_everything(q='finance',
                                      from_param=start_date,
                                      to=end_date,
                                      language='en',
                                      sort_by='popularity'
                                    #   sort_by='relevancy'
                                    #   sort_by='publishedAt'
                                      )
        
        # ニュースのタイトルと説明を取得し、キーワードに一致する記事をフィルタリング
        articles = news_data.get('articles', [])
        relevant_headlines = []
        for article in articles:
            relevant_headlines.append(article['publishedAt'] + ":"+ article['title'] + '\n' + article['description'])

        return"\n".join(relevant_headlines)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return""

def convert_to_numeric(column):
    try:
        return pd.to_numeric(column, errors='raise').round(2)  # エラーが出たら例外に処理が飛ぶ
    except ValueError:
        return column

def convert_to_date(column):
    try:
        # 列を文字列型に変換し、タイムゾーン情報を削除してから日時に変換
        column = pd.to_datetime(column.astype(str), errors='coerce')  # 日時に変換
        return column.dt.date  # 日付のみを取得
        # return pd.to_datetime(column.str.replace(r'(\+|\-)\d{2}:\d{2}$', '', regex=True), errors='coerce').dt.date
    except Exception as e:
        return column

# 不要な列を除外する関数
def filter_historical_data(historical_data, file_path):
    # 必要な列だけを残す
    filtered_data = historical_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    # Date列を適切なdatetime型に変換（エラーを無視して続行）
    filtered_data.loc[:, 'Date'] = convert_to_date(filtered_data['Date'])
    
    # 小数点以下を2桁に丸める
    filtered_data.loc[:, 'Open'] = convert_to_numeric(filtered_data['Open'])
    filtered_data.loc[:, 'High'] = convert_to_numeric(filtered_data['High'])
    filtered_data.loc[:, 'Low'] = convert_to_numeric(filtered_data['Low'])
    filtered_data.loc[:, 'Close'] = convert_to_numeric(filtered_data['Close'])
    
    # NaNが発生した行を削除
    filtered_data = filtered_data.dropna()
    filtered_data.to_csv(file_path, index=False)
    return filtered_data

def process_ai_opinion(is_all, ticker, us_news, historical_section, json_section, prompt_base, output_file_path=None):
    current_time = datetime.now().strftime('%H:%M')
    company_homepage_url = ticker_urls.get(ticker, '')

    # プロンプトの生成
    prompt = prompt_base.format(
        ticker=ticker,
        current_date=datetime.now(),
        news_headlines=us_news,
        company_homepage_url=company_homepage_url,
        historical_section=historical_section + json_section,
        osusume_stocks=", ".join(OSUSUME)
    )

    # AIの意見を取得
    ai_opinion = get_ai_opinion(prompt)
    ai_opinion_cleaned = re.sub(r'[\*\#\_]+', '', ai_opinion)

    # LINE通知を送信
    title = "全体" if is_all else ticker
    message = f"\n★{title}：{current_time}★\n\n{ai_opinion_cleaned}"
    send_line_notify(message)
    print(f"{title}： {ai_opinion_cleaned}")

    if output_file_path:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(message)
        except Exception as e:
            print(f"ファイルの保存中にエラーが発生しました: {e}")
    return ai_opinion_cleaned

def comp_load():
    date_str = datetime.now().strftime('%Y%m%d')
    folder_path = f'./history/{date_str}/{comp_ticker}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    comp_file_path = os.path.join(folder_path, 'etf.csv')
    if not os.path.exists(comp_file_path):
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        save_etf_data(ticker, start_date, end_date, comp_file_path)

    historical_data = load_etf_data(comp_file_path)
    historical_data = filter_historical_data(historical_data, os.path.join(folder_path, 'etf_filter.csv'))
    return historical_data

# ニュースのキャッシュを保存するファイルパス
CACHE_FILE_PATH = './news_cache.json'

# キャッシュの保存
def save_news_cache(news_all_relations):
    cache_data = {
        'timestamp': time.time(),
        'news_all_relations': news_all_relations
    }
    with open(CACHE_FILE_PATH, 'w') as cache_file:
        json.dump(cache_data, cache_file)

# キャッシュの読み込み
def load_news_cache():
    if os.path.exists(CACHE_FILE_PATH):
        with open(CACHE_FILE_PATH, 'r') as cache_file:
            cache_data = json.load(cache_file)
        return cache_data
    return None


# キャッシュの読み込み
def load_history_data(ticker, is_force = False):
    date_str = datetime.now().strftime('%Y%m%d')
    folder_path = f'./history/{date_str}/{ticker}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, 'etf.csv')

    # ETFデータを保存（ファイルが存在しない場合のみ）
    if not os.path.exists(file_path) or is_force:
        start_date = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')  # 本日の日付を取得
        save_etf_data(ticker, start_date, end_date, file_path)

    historical_data = load_etf_data(file_path)
    historical_data = filter_historical_data(historical_data, os.path.join(folder_path, 'etf_filter.csv'))
    return historical_data

# ニュースヘッドラインを取得してAIに送信する関数
def get_news_all_relations(soup):
    # キャッシュを読み込む
    cache_data = load_news_cache()
    if cache_data:
        # 現在の時間から1時間前の時間を計算
        one_hour_ago = time.time() - 3600
        # キャッシュのタイムスタンプが1時間以内なら再利用
        if cache_data['timestamp'] > one_hour_ago:
            return cache_data['news_all_relations']

    # キャッシュがないか1時間以上経過している場合は新しいデータを取得
    news_headlines = fetch_news_titles(soup, True)
    news_all_relations = get_ai_opinion(PROMPT_RELATIONS_CUT + '\n' + news_headlines, '')

    # 新しいデータをキャッシュに保存
    save_news_cache(news_all_relations)
    
    return news_all_relations

# メイン処理
def main(ticker, force_news="", is_short_continue_detail=False):
    is_short = ticker == "SHORT"
    # force_news が指定されている場合、us_news を取得しない
    if not force_news:
        if is_short:
            us_news = get_news_google(2)
            print(us_news)
        else:
            us_news = get_news_google()
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
        historical_data = load_history_data(ticker, FORCE_GET)
        json_file_path = os.path.join(folder_path, 'etf_extra_data.json')
        json_data = load_json_data(json_file_path)

        # JSONデータをプロンプト用に整形
        json_section = "\n\n追加情報:\n" + json.dumps(json_data, indent=2)

        if HISTORY_DAYS != 0:
            if IS_INCLUDE_HISTORY:
                historical_section = f"\n過去{HISTORY_DAYS}日間のデータも考慮していますが、直近1ヶ月のデータには特に注目しています。"
                historical_section += f"\n過去データの一部: {historical_data.to_string(index=False)}"

                if ticker != comp_ticker and IS_COMP:
                    comp_historical_data = comp_load()
                    historical_section += f"\n\n比較用銘柄<{comp_ticker}>の一部: {comp_historical_data.to_string()}"
                # historical_section += f"\n過去データの一部: {historical_data.tail(5).to_string(index=False)}"

    if is_short:
        print('SHORT!')
        comp_historical_data = comp_load()
        historical_section += f"\n\n比較用銘柄<{comp_ticker}>の一部: {comp_historical_data.to_string()}"
        explane = process_ai_opinion(False, ticker, us_news, "", "", PROMPT_BASE_SHORT, os.path.join(folder_path, f'result_detail_{time_str}.txt'))

        if is_short_continue_detail:
            quotes_list = re.findall(r'「(.*?)」', explane)
            print(quotes_list)
            for quote in quotes_list:
                main(quote, us_news, False)  # それぞれのquoteをtickerとして使用
        return
    if IS_ALL:
        process_ai_opinion(True, ticker, us_news, historical_section, json_section, PROMPT_BASE_ALL, os.path.join(folder_path, f'result_all_{time_str}.txt'))
        print('\n\n')
    if IS_DETAIL:
        # process_ai_opinion(False, ticker, us_news, historical_section, json_section, PROMPT_BASE_PROMISING, os.path.join(folder_path, f'result_detail_{time_str}.txt'))
        process_ai_opinion(False, ticker, us_news, historical_section, json_section, PROMPT_BASE_DETAIL, os.path.join(folder_path, f'result_detail_{time_str}.txt'))

def read_news_from_csv(file_path, encoding='utf-8', ticker=None):
    news_list = []
    try:
        with open(file_path, mode='r', encoding=encoding) as file:
            reader = list(csv.reader(file))  # 全行をリストとして読み込む
            headers = ",".join(reader[0])  # ヘッダー行を文字列として結合
            if ticker:
                for row in reversed(reader[1:]):  # データ行を逆順に処理
                    if row[0] == ticker:  # 最初の列がティッカーと一致するかチェック
                        news_content = ",".join(row)  # 行全体を連結
                        news_list.append(f"{headers}\n{news_content}")
                        break  # 一致するティッカーが見つかったら終了
            else:
                for row in reader[1:]:
                    # 各行のデータを適切に処理し、リストに追加
                    news_date = row[0]
                    news_content = ",".join(row[1:])  # 2番目以降の全ての列を連結
                    news_list.append(f"{news_date} - {news_content}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    return "\n".join(news_list)

def future(ticker, is_include_history_data = False):
    historical_section = ""
    if is_include_history_data:
        historical_data = load_history_data(ticker, False)
        historical_section += f"\n過去{HISTORY_DAYS}日間のデータも考慮していますが、直近1ヶ月のデータには特に注目しています。"
        historical_section += f"\n{historical_data.to_string(index=False)}"

    prompt = PROMPT_PROMISING_FUTURE.format(
        ticker=ticker,
        historical_section=historical_section,
        current_date=datetime.now(),
        news=read_news_from_csv('./news_data.csv'),
        research=read_news_from_csv('./research.csv', 'shift_jis', ticker),
    )

    ai_opinion = get_ai_opinion(prompt)
    ai_opinion_cleaned = re.sub(r'[\*\#\_]+', '', ai_opinion)
    send_line_notify(ai_opinion_cleaned)

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

# 複数のティッカーに対してループ処理
if __name__ == "__main__":
    is_future = sys.argv[2] == "FUTURE" if len(sys.argv) > 2 else False 
    if not is_future:
        # Google News
        soup = fetch_news_soup(NEWS_URL)
        news_headlines = fetch_news_titles(soup, True)
        print(news_headlines)
        news_all_relations = get_news_all_relations(soup)
        # yahoo_headlines = fetch_news_titles(soup, "Yahoo!ファイナンス")
        print(news_all_relations)
    else:
        get_news.main(use_latest_csv_date=True)

    for ticker in tickers:
        if is_future:
            # python main.py "NVDA" "FUTURE"
            ouptut_csv.main(ticker)
            future(ticker, False)
        else:
            # python main.py "NVDA"
            main(ticker, news_all_relations, IS_SHORT_CONTINUE_DETAIL)
        # main(ticker)import yfinance as yf