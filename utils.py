import yfinance as yf
import pandas as pd
import openai
import matplotlib.pyplot as plt
import os
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import json
from newsapi import NewsApiClient
import csv
import sys
import util_can_slim_type
from prompts import PROMPT_SYSTEM_BASE, PROMPT_RELATIONS_CUT
from scipy.stats import linregress
from scipy.signal import argrelextrema
import numpy as np
 # pip install scipy
# import fredapi

# https://developer.ft.com/portal/docs-start-commence-making-requests

def load_config(config_path = "./config/config.json"):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config

def load_ticker_urls(json_file_path):
    with open(json_file_path, 'r') as json_file:
        ticker_urls = json.load(json_file)
    return ticker_urls
ticker_urls = load_ticker_urls('./config/ticker_urls.json')


config = load_config()
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


def create_folder_path(folder_name):
    date_str = datetime.now().strftime('%Y%m%d')
    folder_path = f'./history/{date_str}/{folder_name}'
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

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
def save_history_data(ticker, start_date, end_date, file_path):
    etf = yf.Ticker(ticker)
    hist = etf.history(start=start_date, end=end_date)
    hist = hist.reset_index()
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist.to_csv(file_path, index=False)
    data = load_etf_data(file_path)
    return data

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

def data_filter(data, days=None):
    if days is not None:
        # 日付でソートして最新のデータが一番下になるように
        data = data.sort_values(by='Date', ascending=True).reset_index(drop=True)
        
        # 指定された日数分のデータを取得（最新の日付から指定日数分）
        data = data.tail(days)
    return data

# AIの意見を取得
def get_ai_opinion(prompt, prompt_system = PROMPT_SYSTEM_BASE):
    # print(prompt)

    # gpt-4o-2024-08-06 or o1-preview
    if GPT_MODEL != "o1-preview":
        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "system", "content": prompt_system}, {"role": "user", "content": prompt}],
            temperature=0.01)
    else:
        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}])
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

def comp_load(ticker):
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

def read_news_from_csv(file_path, encoding='utf-8', ticker=None):
    news_list = []
    try:
        with open(file_path, mode='r', encoding=encoding) as file:
            reader = list(csv.reader(file))  # 全行をリストとして読み込む
            headers = ",".join(reader[0])  # ヘッダー行を文字列として結合
            if ticker:
                if ticker == "ALL":
                    # ヘッダーを一度だけ追加
                    news_list.append(headers)
                    for row in reversed(reader[1:]):  # データ行を逆順に処理
                        news_content = ",".join(row)  # 行全体を連結
                        news_list.append(news_content)
                else:
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

# 取っ手なしカップ型を検出する関数
def detect_cup_without_handle(data, window=20):
    # ローリングウィンドウで最小値を検出（カップの底）
    minima = data['Close'].rolling(window=window).min()

    # 高値の局所的な極大を検出（カップの両側）
    max_idx = argrelextrema(data['Close'].values, np.greater, order=window)[0]

    if len(max_idx) >= 2:
        # 最低2つの極大値があればカップ型を形成する可能性がある
        left_peak = max_idx[0]
        right_peak = max_idx[-1]
        cup_bottom = minima.idxmin()  # カップの底

        if data['Close'][left_peak] > data['Close'][cup_bottom] and data['Close'][right_peak] > data['Close'][cup_bottom]:
            return True, left_peak, cup_bottom, right_peak
    return False, None, None, None

def convert_to_weekly(data):
    weekly_data = data.resample('W', on='Date').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'  # 週ごとの取引量を合計
    }).dropna()
    
    return weekly_data

# 過去データから売買価格を決定（上方チャネルラインを計算）
def get_buy_sell_price(ticker, date = 720):
    # 過去データを取得
    start_date = (datetime.now() - timedelta(days=date)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    folder_path = create_folder_path(ticker)
    comp_file_path = os.path.join(folder_path, f'{ticker}_etf.csv')
    data = save_history_data(ticker, start_date, end_date, comp_file_path)
    # data = load_etf_data(comp_file_path)

    # 買い価格を決定
    get_buy_price(data, folder_path)

    # 売り価格を決定
    # get_sell_price(data)

    # 損切価格を決定
    # get_buy_price(data)

def get_buy_price(data, image_folder=None):
    weekly_data = convert_to_weekly(data)
    pattern_found, purchase_price, left_peak, cup_bottom, right_peak = util_can_slim_type.detect_cup_with_handle(weekly_data, image_folder=image_folder)
    if pattern_found:
        print(f"取っ手付きカップ型が検出されました。購入価格は {purchase_price} です。")

    # pattern_found, purchase_price, left_peak, saucer_bottom, right_peak = util_can_slim_type.detect_saucer_with_handle(data_filter(data, 360), image_folder=image_folder)
    # if pattern_found:
    #     print(f"取っ手付きソーサー型が検出されました。購入価格は {purchase_price} です。")

    # pattern_found, purchase_price, first_bottom, second_bottom = util_can_slim_type.detect_double_bottom(data_filter(data, 180), image_folder=image_folder)
    # if pattern_found:
    #     print(f"ダブルボトム型が検出されました。購入価格は {purchase_price} です。")

    # pattern_found, purchase_price = util_can_slim_type.detect_flat_base(data_filter(data, 60), image_folder=image_folder)
    # if pattern_found:
    #     print(f"フラットベース型が検出されました。購入価格は {purchase_price} です。")

    # pattern_found, purchase_price = util_can_slim_type.detect_ascending_base(data_filter(data, 180), image_folder=image_folder)
    # if pattern_found:
    #     print(f"上昇トライアングル型が検出されました。購入価格は {purchase_price} です。")

    # pattern_found, purchase_price = util_can_slim_type.detect_consolidation(data_filter(data, 90), image_folder=image_folder)
    # if pattern_found:
    #     print(f"コンソリデーション型が検出されました。購入価格は {purchase_price} です。")

    # pattern_found, purchase_price = util_can_slim_type.detect_vcp(data_filter(data, 180), image_folder=image_folder)
    # if pattern_found:
    #     print(f"VCPパターンが検出されました。購入価格は {purchase_price} です。")

    # is_cup_without_handle, left, bottom, right = util_can_slim_type.detect_cup_without_handle(data)
    # if is_cup_without_handle:
    #     print(f"取っ手なしカップ型検出: 左ピーク={left}, カップ底={bottom}, 右ピーク={right}")
    #     buy_price = calculate_buy_price(data, right)
    #     print(f"推奨購入価格: {buy_price:.2f}")

def get_sell_price(data, date = 360):
    util_can_slim_type.detect_upper_channel_line(data)
