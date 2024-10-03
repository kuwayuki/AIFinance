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

def calculate_buy_price(data, right_peak, buffer_percent=1.02):
    # 右ピークの価格を取得し、購入価格を計算（例: 右ピークの価格を2%上回る）
    right_peak_price = data['Close'][right_peak]
    buy_price = right_peak_price * buffer_percent
    return buy_price

def get_sell_price(data, date = 360):

    # 高値と日付を抽出
    highs = data['High']
    dates = pd.to_datetime(data['Date'])
    
    # 最高値の抽出（過去数ヶ月分、最低3つの高値を検出）
    high_points = []
    num_highs_needed = 3
    for i in range(1, len(highs) - 1):
        # 高値の条件: 前後の日よりも高い
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            high_points.append((dates[i], highs[i]))
        
        if len(high_points) >= num_highs_needed:
            break
    
    # 高値が3つ以上あるかチェック
    if len(high_points) < num_highs_needed:
        print(f"高値が十分にありません")
        return None, None
    
    # 高値の座標を取得
    high_dates = [point[0].toordinal() for point in high_points]  # 日付を数値化
    high_values = [point[1] for point in high_points]  # 高値の価格
    
    # 高値の直線回帰を行い、チャネルラインを作成
    slope, intercept, _, _, _ = linregress(high_dates, high_values)
    
    # 直線（上方チャネルライン）の式: y = slope * x + intercept
    upper_channel_line = slope * dates.apply(lambda date: date.toordinal()) + intercept
    
    # 現在の価格
    current_price = data['Close'].iloc[-1]
    
    # 売りサイン: 現在の価格が上方チャネルラインを超えた場合
    sell_signal = current_price > upper_channel_line.iloc[-1]
    
    # 次の売り価格を予測 (例えば、1週間後の価格を予測)
    future_date = dates.iloc[-1] + pd.Timedelta(days=7)
    future_date_ordinal = future_date.toordinal()
    future_sell_price = slope * future_date_ordinal + intercept

    # 結果を出力
    print(f'現在価格: {current_price}')
    print(f'売りサイン: {sell_signal} (上方チャネルライン: {upper_channel_line.iloc[-1]})')
    print(f'次の売り価格（1週間後予測）: {future_sell_price}（予測日: {future_date.strftime("%Y-%m-%d")}）')
    
    # グラフの描画（オプション）
    plt.figure(figsize=(10, 6))
    plt.plot(dates, highs, label="高値", color="blue")
    plt.plot(dates, upper_channel_line, label="上方チャネルライン", color="red")
    plt.scatter([point[0] for point in high_points], [point[1] for point in high_points], color="green", label="高値のポイント")
    plt.axhline(y=future_sell_price, color='orange', linestyle='--', label="次の売り価格予測")
    plt.title(f'高値と上方チャネルライン')
    plt.xlabel("日付")
    plt.ylabel("価格")
    plt.legend()
    plt.show()
    
    return upper_channel_line.iloc[-1], sell_signal

def plot_pattern(data, points, lines, title, image_name, image_folder):
    """
    共通のパターンプロット関数。

    Parameters:
        data (DataFrame): 株価データ。
        points (list of dict): プロットするポイントのリスト。
            各ポイントは {'index': int, 'label': str, 'color': str} の辞書。
        lines (list of dict): プロットする水平線のリスト。
            各ラインは {'y': float, 'label': str, 'color': str, 'linestyle': str} の辞書。
        title (str): グラフのタイトル。
        image_name (str): 保存する画像のファイル名。
        image_folder (str): 画像を保存するフォルダのパス。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price')

    # ポイントをプロット
    for point in points:
        plt.scatter(data.index[point['index']], data['Close'].iloc[point['index']],
                    color=point['color'], label=point['label'])

    # ラインをプロット
    for line in lines:
        plt.axhline(y=line['y'], color=line['color'], linestyle=line['linestyle'], label=line['label'])

    plt.legend()
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')

    # 画像を保存
    if image_folder is not None:
        image_path = os.path.join(image_folder, image_name)
        plt.savefig(image_path)
        plt.close()
        print(f"{title} の図を保存しました: {image_path}")
    else:
        plt.show()

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
    pattern_found, purchase_price, left_peak, cup_bottom, right_peak = detect_cup_with_handle(weekly_data, image_folder=image_folder)
    if pattern_found:
        print(f"取っ手付きカップ型が検出されました。購入価格は {purchase_price} です。")

    # pattern_found, purchase_price, left_peak, saucer_bottom, right_peak = detect_saucer_with_handle(data_filter(data, 360), image_folder=image_folder)
    # if pattern_found:
    #     print(f"取っ手付きソーサー型が検出されました。購入価格は {purchase_price} です。")

    # pattern_found, purchase_price, first_bottom, second_bottom = detect_double_bottom(data_filter(data, 180), image_folder=image_folder)
    # if pattern_found:
    #     print(f"ダブルボトム型が検出されました。購入価格は {purchase_price} です。")

    # pattern_found, purchase_price = detect_flat_base(data_filter(data, 60), image_folder=image_folder)
    # if pattern_found:
    #     print(f"フラットベース型が検出されました。購入価格は {purchase_price} です。")

    # pattern_found, purchase_price = detect_ascending_base(data_filter(data, 180), image_folder=image_folder)
    # if pattern_found:
    #     print(f"上昇トライアングル型が検出されました。購入価格は {purchase_price} です。")

    # pattern_found, purchase_price = detect_consolidation(data_filter(data, 90), image_folder=image_folder)
    # if pattern_found:
    #     print(f"コンソリデーション型が検出されました。購入価格は {purchase_price} です。")

    # pattern_found, purchase_price = detect_vcp(data_filter(data, 180), image_folder=image_folder)
    # if pattern_found:
    #     print(f"VCPパターンが検出されました。購入価格は {purchase_price} です。")

    # カップ型パターンを検出
    # is_cup_without_handle, left, bottom, right = detect_cup_without_handle(data)
    # if is_cup_without_handle:
    #     print(f"取っ手なしカップ型検出: 左ピーク={left}, カップ底={bottom}, 右ピーク={right}")
    #     buy_price = calculate_buy_price(data, right)
    #     print(f"推奨購入価格: {buy_price:.2f}")


# カップの高値から安値までの深さは12～35％が好ましい
def detect_cup_with_handle(data, window=4, image_folder=None):
    # 高値の局所的な極大を検出（カップの両側）
    max_idx = argrelextrema(data['Close'].values, np.greater, order=window)[0]

    # 安値の局所的な極小を検出（カップの底）
    min_idx = argrelextrema(data['Close'].values, np.less, order=window)[0]

    if len(max_idx) >= 2 and len(min_idx) >= 1:
        left_peak = max_idx[0]
        right_peak = max_idx[-1]
        cup_bottom = min_idx[np.argmin(data['Close'].values[min_idx])]

        # カップの深さを確認（高値から安値までの比率）
        left_peak_price = data['Close'].values[left_peak]
        right_peak_price = data['Close'].values[right_peak]
        cup_bottom_price = data['Close'].values[cup_bottom]

        # カップの深さ（%）
        depth = (left_peak_price - cup_bottom_price) / left_peak_price * 100

        # 深さが12〜35%の範囲にあるかチェック
        if depth < 12 or depth > 35:
            print(f"カップの深さが12〜35％の範囲外です: {depth:.2f}%")
            return False, None, None, None, None

        print(depth)
        # 取っ手部分の開始と終了を推定（右のピークより前で下落があるか確認）
        handle_start = cup_bottom
        handle_end = right_peak

        # 取っ手部分のデータを取得
        handle_data = data.iloc[handle_start:handle_end + 1]

        # 取っ手部分の長さが1週間以上であることを確認
        if len(handle_data) < 1:
            print("取っ手部分が1週間未満です。パターンを無効とします。")
            return False, None, None, None, None

        # 取っ手部分の値動きが下降しているか確認（右ピークまでに一定の下落があること）
        handle_min_price = handle_data['Close'].min()
        if handle_min_price >= right_peak_price:
            print("取っ手部分の下降が確認できません。パターンを無効とします。")
            return False, None, None, None, None

        # 取っ手部分の最新の価格（右端の値）を購入価格に設定
        handle_high = handle_data['Close'].iloc[-1]  # 最新（右端）の値を使用
        purchase_price = handle_high * 1

        # プロット用のポイントとラインを準備
        points = [
            {'index': left_peak, 'label': 'Left Peak', 'color': 'g'},
            {'index': cup_bottom, 'label': 'Cup Bottom', 'color': 'r'},
            {'index': right_peak, 'label': 'Right Peak', 'color': 'g'}
        ]
        lines = [
            {'y': purchase_price, 'label': 'Purchase Price', 'color': 'purple', 'linestyle': '--'}
        ]

        # max_idx の他の局所的極大値も追加（色を区別）
        for idx in max_idx:
            if idx != left_peak and idx != right_peak:
                points.append({'index': idx, 'label': 'Local Max', 'color': 'b'})  # 局所的極大値

        # min_idx の他の局所的極小値も追加（色を区別）
        for idx in min_idx:
            if idx != cup_bottom:
                points.append({'index': idx, 'label': 'Local Min', 'color': 'orange'})  # 局所的極小値

        # 共通のプロット関数を呼び出す
        plot_pattern(
            data=data,
            points=points,
            lines=lines,
            title='Cup with Handle Pattern',
            image_name='1_cup_with_handle.png',
            image_folder=image_folder
        )

        return True, purchase_price, left_peak, cup_bottom, right_peak
    else:
        return False, None, None, None, None

def detect_saucer_with_handle(data, window=60, image_folder=None):
    # 高値と安値の局所的な極大・極小を検出
    max_idx = argrelextrema(data['Close'].values, np.greater, order=window)[0]
    min_idx = argrelextrema(data['Close'].values, np.less, order=window)[0]

    if len(max_idx) >= 2 and len(min_idx) >= 1:
        left_peak = max_idx[0]
        right_peak = max_idx[-1]
        saucer_bottom = min_idx[np.argmin(data['Close'].values[min_idx])]

        # 取っ手部分の開始と終了を推定
        handle_start = saucer_bottom
        handle_end = right_peak

        # 取っ手部分のデータを取得
        handle_data = data.iloc[handle_start:handle_end+1]

        # 取っ手部分の高値を取得
        handle_high = handle_data['Close'].max()
        purchase_price = handle_high * 1.02  # 高値の2%上

        # プロット用のポイントとラインを準備
        points = [
            {'index': left_peak, 'label': 'Left Peak', 'color': 'g'},
            {'index': saucer_bottom, 'label': 'Saucer Bottom', 'color': 'r'},
            {'index': right_peak, 'label': 'Right Peak', 'color': 'g'}
        ]
        lines = [
            {'y': purchase_price, 'label': 'Purchase Price', 'color': 'purple', 'linestyle': '--'}
        ]

        # 共通のプロット関数を呼び出す
        plot_pattern(
            data=data,
            points=points,
            lines=lines,
            title='Saucer with Handle Pattern',
            image_name='2_saucer_with_handle.png',
            image_folder=image_folder
        )

        return True, purchase_price, left_peak, saucer_bottom, right_peak
    else:
        return False, None, None, None, None
def detect_double_bottom(data, window=20, image_folder=None):
    # 局所的な極小値（ボトム）を検出
    min_idx = argrelextrema(data['Close'].values, np.less, order=window)[0]

    if len(min_idx) >= 2:
        first_bottom = min_idx[0]
        second_bottom = min_idx[1]

        # ネックライン（2つのボトム間の高値）を取得
        neckline = data['Close'][first_bottom:second_bottom+1].max()
        purchase_price = neckline * 1.02  # ネックラインの2%上

        # プロット用のポイントとラインを準備
        points = [
            {'index': first_bottom, 'label': 'First Bottom', 'color': 'r'},
            {'index': second_bottom, 'label': 'Second Bottom', 'color': 'r'}
        ]
        lines = [
            {'y': neckline, 'label': 'Neckline', 'color': 'orange', 'linestyle': '--'},
            {'y': purchase_price, 'label': 'Purchase Price', 'color': 'purple', 'linestyle': '--'}
        ]

        # 共通のプロット関数を呼び出す
        plot_pattern(
            data=data,
            points=points,
            lines=lines,
            title='Double Bottom Pattern',
            image_name='3_double_bottom.png',
            image_folder=image_folder
        )

        return True, purchase_price, first_bottom, second_bottom
    else:
        return False, None, None, None
def detect_flat_base(data, period=30, tolerance=0.03, image_folder=None):
    recent_data = data[-period:]
    max_price = recent_data['Close'].max()
    min_price = recent_data['Close'].min()

    # 変動幅が許容範囲内か確認
    if (max_price - min_price) / max_price < tolerance:
        purchase_price = max_price * 1.02  # ベースの高値の2%上

        # プロット用のラインを準備
        points = []
        lines = [
            {'y': max_price, 'label': 'Flat Base High', 'color': 'orange', 'linestyle': '--'},
            {'y': min_price, 'label': 'Flat Base Low', 'color': 'green', 'linestyle': '--'},
            {'y': purchase_price, 'label': 'Purchase Price', 'color': 'purple', 'linestyle': '--'}
        ]

        # 共通のプロット関数を呼び出す
        plot_pattern(
            data=data,
            points=points,
            lines=lines,
            title='Flat Base Pattern',
            image_name='4_flat_base.png',
            image_folder=image_folder
        )

        return True, purchase_price
    else:
        return False, None
def detect_ascending_base(data, window=60, image_folder=None):
    # 移動平均線を使用してトレンドを確認
    data['MA'] = data['Close'].rolling(window=window).mean()

    # 直近の価格が移動平均線より上にあるか確認
    if data['Close'].iloc[-1] > data['MA'].iloc[-1]:
        recent_high = data['Close'][-window:].max()
        purchase_price = recent_high * 1.02  # 高値の2%上

        # プロット用のラインを準備
        points = []
        lines = [
            {'y': recent_high, 'label': 'Recent High', 'color': 'orange', 'linestyle': '--'},
            {'y': purchase_price, 'label': 'Purchase Price', 'color': 'purple', 'linestyle': '--'}
        ]

        # 共通のプロット関数を呼び出す
        plot_pattern(
            data=data,
            points=points,
            lines=lines,
            title='Ascending Base Pattern',
            image_name='5_ascending_base.png',
            image_folder=image_folder
        )

        return True, purchase_price
    else:
        return False, None
def detect_consolidation(data, period=30, tolerance=0.05, image_folder=None):
    recent_data = data[-period:]
    max_price = recent_data['Close'].max()
    min_price = recent_data['Close'].min()

    # 変動幅が許容範囲内か確認
    if (max_price - min_price) / max_price < tolerance:
        breakout_price = max_price * 1.02  # レンジ上限の2%上

        # プロット用のラインを準備
        points = []
        lines = [
            {'y': max_price, 'label': 'Consolidation High', 'color': 'orange', 'linestyle': '--'},
            {'y': min_price, 'label': 'Consolidation Low', 'color': 'green', 'linestyle': '--'},
            {'y': breakout_price, 'label': 'Purchase Price', 'color': 'purple', 'linestyle': '--'}
        ]

        # 共通のプロット関数を呼び出す
        plot_pattern(
            data=data,
            points=points,
            lines=lines,
            title='Consolidation Pattern',
            image_name='6_consolidation.png',
            image_folder=image_folder
        )

        return True, breakout_price
    else:
        return False, None
def detect_vcp(data, window_sizes=[60, 40, 20, 10], image_folder=None):
    volatilities = []
    for window in window_sizes:
        recent_data = data[-window:]
        volatility = recent_data['Close'].std() / recent_data['Close'].mean()
        volatilities.append(volatility)

    # ボラティリティが縮小しているか確認
    if all(x > y for x, y in zip(volatilities, volatilities[1:])):
        recent_high = data['Close'][-10:].max()
        purchase_price = recent_high * 1.02  # 高値の2%上

        # プロット用のラインを準備
        points = []
        lines = [
            {'y': recent_high, 'label': 'Recent High', 'color': 'orange', 'linestyle': '--'},
            {'y': purchase_price, 'label': 'Purchase Price', 'color': 'purple', 'linestyle': '--'}
        ]

        # 共通のプロット関数を呼び出す
        plot_pattern(
            data=data,
            points=points,
            lines=lines,
            title='Volatility Contraction Pattern (VCP)',
            image_name='7_vcp.png',
            image_folder=image_folder
        )

        return True, purchase_price
    else:
        return False, None
