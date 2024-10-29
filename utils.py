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
from prompts import PROMPT_SYSTEM_BASE, PROMPT_RELATIONS_CUT, PROMPT_SYSTEM, PROMPT_USER
from constant import CONSTANT
from scipy.stats import linregress
from scipy.signal import argrelextrema
import numpy as np
from yahooquery import Ticker
# from sec_edgar_downloader import Downloader
 # pip install scipy
# import fredapi

# https://developer.ft.com/portal/docs-start-commence-making-requests

log_file_path = ''
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

def save_ticker_auto(ticker, date = 720):
    start_date = (datetime.now() - timedelta(days=date)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    folder_path = create_folder_path(ticker)
    comp_file_path = os.path.join(folder_path, f'{ticker}_etf.csv')
    data = save_history_data(ticker, start_date, end_date, comp_file_path)
    return data

# ETFデータを保存する関数
def save_history_data(ticker, start_date, end_date, file_path):
    if os.path.exists(file_path):
        data = load_etf_data(file_path)
        return data

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

def data_filter_date(data, days=None):
    if days is not None:
        # 現在の日付を取得
        current_date = datetime.now()
        
        # Date列をdatetime形式に変換
        data['Date'] = pd.to_datetime(data['Date'])
        
        # 指定された日数分のデータをフィルタリング（現在日時からdays日前以降）
        start_date = current_date - timedelta(days=days)
        data = data[data['Date'] >= start_date].reset_index(drop=True)
    
    return data

def data_filter_index(data, days=None):
    if days is not None:
        # 日付でソートして最新のデータが一番下になるように
        data = data.sort_values(by='Date', ascending=True).reset_index(drop=True)
        
        # 指定された日数分のデータを取得（最新の日付から指定日数分）
        data = data.tail(days)
    return data

# AIの意見を取得
def get_ai_opinion(prompt, prompt_system = PROMPT_SYSTEM_BASE, is_print = True):
    # gpt-4o-2024-08-06 or o1-preview
    if GPT_MODEL != "o1-preview":
        if prompt_system is not None:
            response = openai.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "system", "content": prompt_system}, {"role": "user", "content": prompt}],
                temperature=0.01)
        else:
            response = openai.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.01)
    else:
        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}])
    print(response.usage)
    result = response.choices[0].message.content
    if is_print:
        print(result)
    return result

def send_line_notify(notification_message, title = None):
    line_notify_token = 'Z392GKtaICiiiSndtfrKqDmYliv8df5S9AIekyPpSoa'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    
    if title is not None:
        notification_message = f"<{title}>\n{notification_message}"

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


def get_industry_tickers(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        industry = info.get('industry', 'N/A')
        sector = info.get('sector', 'N/A')

        print(f"業界: {industry}, セクター: {sector}")
        return industry, sector
    
    except Exception as e:
        print(f"{ticker_symbol} のデータ取得中にエラーが発生しました: {e}")
        return None, None

# TODO: finviz eliteで取得
def get_top_3_stocks_by_industry_and_sector(industry, sector):
    try:
        # Yahoo Financeのスクリーナーを使って同じ業界とセクターに属する銘柄を取得
        screener = Ticker(screen=True)
        screener_data = screener.summary_profile

        # 業界とセクターが一致する銘柄を選定
        industry_sector_stocks = [symbol for symbol, data in screener_data.items() 
                                  if data.get('industry') == industry and data.get('sector') == sector]
        
        # 銘柄ごとにリターンを計算し、その値でランキング
        stock_performance = {}
        for symbol in industry_sector_stocks:
            try:
                ticker = Ticker(symbol)
                data = ticker.history(period="1y")
                if data.empty:
                    continue
                stock_return = data['adjclose'].pct_change().cumsum().iloc[-1]
                stock_performance[symbol] = stock_return
            except:
                continue
        
        # トップ3の銘柄を取得
        top_3_stocks = sorted(stock_performance.items(), key=lambda x: x[1], reverse=True)[:3]
        return [symbol for symbol, _ in top_3_stocks]
    
    except Exception as e:
        print(f"業界内の銘柄取得中にエラーが発生しました: {e}")
        return []

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

# TODO: 取引量の追加
def convert_to_weekly(data):
    weekly_data = data.resample('W', on='Date').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'  # 週ごとの取引量を合計
    }).dropna()
    
    return weekly_data

# keysがTimestampになっている部分を文字列に変換
def convert_keys_to_str(d):
    return {str(k): v for k, v in d.items()}

def convert_dict(d):
    if hasattr(d, 'to_dict'):
        d = d.to_dict()
        d = {str(k): convert_keys_to_str(v) for k, v in d.items()}
    return d

def jsonOutput(data, title="", is_output=True):
    try:
        if data is None:
            return None

        if hasattr(data, 'to_dict'):
            data = data.to_dict()
            data = {str(k): convert_keys_to_str(v) for k, v in data.items()}
        
        if is_output:
            output_log(title, True)
            output_log(data, True)
    except:
        print('err')
    return data

def all_print(ticker):
    etf = yf.Ticker(ticker)
    # tech = yf.Sector(etf.info.get('sectorKey'))
    # software = yf.Industry(etf.info.get('industryKey'))
    # Sector and Industry to Ticker
    # tech_ticker = tech.ticker
    # tech_ticker.info
    # jsonOutput(tech_ticker.info, "関係基本情報（会社名、業界、株価など）")
    # software_ticker = software.ticker
    # jsonOutput(software_ticker.history(), "関係AA基本情報（会社名、業界、株価など）")

    # 基本情報
    jsonOutput(etf.info, "基本情報（会社名、業界、株価など）")
    
    # 財務諸表
    jsonOutput(etf.financials, "財務諸表")
    jsonOutput(etf.balance_sheet, "バランスシート")
    jsonOutput(etf.cashflow, "キャッシュフロー")
    jsonOutput(etf.earnings, "収益情報")
    
    # 配当金と株式分割
    jsonOutput(etf.dividends, "配当金情報")
    jsonOutput(etf.splits, "株式分割情報")
    
    # サステナビリティ
    jsonOutput(etf.sustainability, "サステナビリティスコア")
    
    # 推奨情報
    jsonOutput(etf.recommendations, "アナリスト推奨")
    jsonOutput(etf.recommendations_summary, "推奨サマリー")
    jsonOutput(etf.upgrades_downgrades, "アップグレード/ダウングレード情報")
    
    # 会社の予定
    output_log(etf.calendar, title="会社の予定（カレンダー）")

    # 株式保有情報
    jsonOutput(etf.major_holders, "主要株主")

    max_retries = 3
    retries = 0
    while retries < max_retries:
        try: 
            institutional_holders = etf.institutional_holders
            if institutional_holders is not None and not institutional_holders.empty:
                output_log(institutional_holders, title="機関投資家の保有株")
                break
        except Exception as e:
            retries += 1
            time.sleep(2)  # 2秒待ってから再試行

    retries = 0
    while retries < max_retries:
        try: 
            mutualfund_holders = etf.mutualfund_holders
            if mutualfund_holders is not None and not mutualfund_holders.empty:
                output_log(mutualfund_holders, title="投資信託保有者")
                break
        except Exception as e:
            retries += 1
            time.sleep(2)  # 2秒待ってから再試行
    
    # ニュース
    jsonOutput(etf.news, "最新ニュース")
    
    # オプション
    jsonOutput(etf.options, "オプションの有効期限")
    
    # 追加の情報
    jsonOutput(etf.earnings_dates, "収益日")
    jsonOutput(etf.history_metadata, "履歴メタデータ")
    jsonOutput(etf.analyst_price_targets, "アナリスト価格目標")
    jsonOutput(etf.revenue_estimate, "収益見積もり")
    jsonOutput(etf.earnings_estimate, "収益予測")
    jsonOutput(etf.eps_trend, "収益トレンド")

    # インサイダー取引情報
    output_log(etf.insider_purchases, title="インサイダーの株式購入情報")
    output_log(etf.insider_transactions, title="インサイダー取引情報")
    output_log(etf.insider_roster_holders, title="インサイダーホルダー")
    
    # キャピタルゲイン情報
    output_log(etf.capital_gains, title="キャピタルゲイン情報")
    
    # 株式アクション（配当・分割など）
    jsonOutput(etf.actions, "株式アクション情報（配当・分割など）")
    
    # ISIN情報
    jsonOutput(etf.isin, "国際証券識別番号（ISIN）")

# データ取得の再試行設定
def get_data_with_retry(fetch_function, retries=3, wait=10):
    attempts = 0
    while attempts < retries:
        try:
            data = fetch_function()
            if data is not None:
                return convert_dict(data)
        except Exception as e:
            print(f"Error fetching data: {e}. Retrying {attempts + 1}/{retries}...")
            attempts += 1
            time.sleep(wait)
    return None  # 取得失敗の場合はNoneを返す

def filter_can_slim(ticker, is_send_news = True):
    # return all_print(ticker)
    output_log(f"\n☆☆CAN-SLIM評価：開始☆☆")
    tickerInfo = yf.Ticker(ticker)
    
    score = 0
    failed_conditions = 0
    eps_under = CONSTANT["EPS"]

    # C (Current Quarterly Earnings and Sales): 四半期EPS成長を確認
    quarterly_earnings = tickerInfo.quarterly_earnings
    if quarterly_earnings is not None and not quarterly_earnings.empty:
        # 四半期ごとのEPS成長率を計算し、25%以上の成長が複数回確認できるかをチェック
        quarterly_growth_count = 0
        latest_growth_rate = None
        for i in range(1, len(quarterly_earnings)):
            growth_rate = (quarterly_earnings['Earnings'][i] - quarterly_earnings['Earnings'][i-1]) / abs(quarterly_earnings['Earnings'][i-1]) * 100
            if growth_rate > eps_under:
                quarterly_growth_count += 1
            if i == len(quarterly_earnings) - 1:
                latest_growth_rate = growth_rate
        
        if latest_growth_rate is not None and latest_growth_rate > eps_under and quarterly_growth_count >= 2:  # 直近の成長も含めて評価
            output_log(f"C：◎四半期EPS成長率が{eps_under}%以上を超えています。直近は{latest_growth_rate}%でした。")
            score += 1
        else:
            output_log(f"C：四半期EPS成長率が{eps_under}%以上の期間が足りませんでした。直近は{latest_growth_rate}%でした。")
            failed_conditions += 1
    else:
        output_log(f"C：四半期EPSデータがありません。Reported EPSデータを代用します。")
        # Reported EPSデータを代用
        earnings_data = tickerInfo.earnings_dates
        if earnings_data is not None:
            # Reported EPS から有効な四半期EPSデータを取得
            reported_eps = earnings_data.get('Reported EPS', {})
            current_time = pd.Timestamp.now().tz_localize(None)  # 現在のタイムスタンプをtz-naiveに変換
            quarterly_eps = {k: v for k, v in reported_eps.items() if not pd.isna(v) and pd.to_datetime(k).tz_localize(None) < current_time}
            sorted_eps = dict(sorted(quarterly_eps.items(), key=lambda x: pd.to_datetime(x[0]).tz_localize(None)))

            if len(sorted_eps) >= 2:
                eps_values = list(sorted_eps.values())
                quarterly_growth_count = 0
                latest_growth_rate = None
                for i in range(1, len(eps_values)):
                    growth_rate = (eps_values[i] - eps_values[i-1]) / abs(eps_values[i-1]) * 100
                    if growth_rate > eps_under:
                        quarterly_growth_count += 1
                    if i == len(eps_values) - 1:
                        latest_growth_rate = growth_rate
                
                if latest_growth_rate is not None and latest_growth_rate > eps_under:  # 直近の成長も含めて評価
                # if latest_growth_rate is not None and latest_growth_rate > eps_under and quarterly_growth_count >= 2:  # 直近の成長も含めて評価
                    output_log(f"C：◎四半期EPS成長率が{eps_under}%以上を超えています。直近は{latest_growth_rate}%でした。")
                    score += 1
                else:
                    output_log(f"C：四半期EPS成長率が{eps_under}%以上の期間が足りませんでした。直近は{latest_growth_rate}%でした。")
                    failed_conditions += 1
            else:
                output_log(f"C：有効な四半期EPSデータが不足しています。")
                failed_conditions += 1
        else:
            output_log(f"C：四半期EPSデータがありません。")
            failed_conditions += 1

    # A (Annual Earnings Increases): 年間EPSの成長を確認
    income_statement = tickerInfo.income_stmt
    if income_statement is not None and not income_statement.empty:
        # EPSデータを取得し、なければNet Incomeを代用
        if 'Diluted EPS' in income_statement.index:
            annual_earnings = income_statement.loc['Diluted EPS'].sort_index(ascending=True)
        elif 'Basic EPS' in income_statement.index:
            annual_earnings = income_statement.loc['Basic EPS'].sort_index(ascending=True)
        elif 'Reported EPS' in income_statement.index:
            annual_earnings = income_statement.loc['Reported EPS'].sort_index(ascending=True)
        elif 'Net Income' in income_statement.index:
            annual_earnings = income_statement.loc['Net Income'].sort_index(ascending=True)
        else:
            annual_earnings = None

        if annual_earnings is not None:
            annual_growth_count = 0
            latest_growth_rate = None
            # 年間EPS成長率を計算し、25%以上かどうかを確認
            for i in range(1, len(annual_earnings)):
                growth_rate = (annual_earnings[i] - annual_earnings[i-1]) / abs(annual_earnings[i-1]) * 100
                if growth_rate > eps_under:
                    annual_growth_count += 1
                if i == len(annual_earnings) - 1:
                    latest_growth_rate = growth_rate
            
            if latest_growth_rate is not None and latest_growth_rate > eps_under and annual_growth_count >= 2:  # 直近の成長も含めて評価
                output_log(f"A：◎年間EPS成長率が{eps_under}%以上です。")
                score += 1
            else:
                output_log(f"A：年間EPS成長率が{eps_under}%以上の期間が足りませんでした。{latest_growth_rate}%でした。")
                failed_conditions += 1
        else:
            output_log("A：年間EPSデータがありません。")
            failed_conditions += 1
    else:
        output_log("A：年間EPSデータがありません。")
        failed_conditions += 1

    # N (New Products, Management, or Conditions): 最近のニュースから新製品や経営の変化をチェック
    news = tickerInfo.news
    if news is not None:
        new_product_or_management = False
        recent_weeks = pd.Timestamp.now() - pd.Timedelta(weeks=2)
        news_list = []
        news_link_list = []
        for item in news:
            news_date = pd.to_datetime(item['providerPublishTime'], unit='s')
            if news_date > recent_weeks:
                news = f"{pd.to_datetime(item['providerPublishTime'], unit='s').strftime('%Y-%m-%d')}: {item['title']}"
                news_list.append(news)
                news_link_list.append(item['link'])
                new_product_or_management = True

        if new_product_or_management and is_new_news("\n".join(news_list)):
            output_log(f"N：◎新製品または経営の変化に関するニュースが見つかりました")
            news_summary = get_ai_opinion("\n".join(news_list), PROMPT_SYSTEM["JAPANESE_SUMMARY_ARRAY"])
            if is_send_news:
                split_news_summary = news_summary.split('\n')
                news_list_send = []
                for index, line in enumerate(split_news_summary):
                    news_list_send.append(f"{line}")
                    news_list_send.append(f"{news_link_list[index]}")
                send_line_notify("\n".join(news_list_send), f"{ticker}-News")
            output_log("\n".join(news_list_send))
            for link in news_link_list:
                output_log(link + '\n', is_print_only=True)
            # output_log(get_ai_opinion("\n".join(news_list), PROMPT_SYSTEM["JAPANESE_SUMMARY_ARRAY"]))
            score += 1
        else:
            output_log("N：新製品または経営の変化に関するニュースが見つかりませんでした。")
            failed_conditions += 1
    else:
        output_log("N：ニュースデータがありません。")
        failed_conditions += 1

    # S (Supply and Demand): 過去1年の出来高データを取得して、最近の増加を確認
    historical_data = tickerInfo.history(period="1y")
    if not historical_data.empty:
        volume_data = historical_data['Volume']
        recent_max_volume = volume_data[-4:].max()  # 直近4週間の最大出来高
        average_volume = volume_data.mean()

        # 出来高が増加しているか確認
        if recent_max_volume > 1.05 * average_volume:
            score += 1
            output_log(f"S：◎出来高の増加が確認されました。直近の最大出来高: {recent_max_volume} 平均出来高: {average_volume}")
            # 機関投資家の活動の可能性を間接的に評価
            if recent_max_volume > 1.2 * average_volume:
                score += 0.5 # ボーナス
                output_log("S：〇直近の出来高が大幅に増加しています。")
        else:
            output_log(f"S：出来高の増加が十分ではありません。直近の最大出来高: {recent_max_volume} 平均出来高: {average_volume}")
            failed_conditions += 1
    else:
        output_log("S：過去1年の出来高データがありません。")
        failed_conditions += 1

    # L (Leader or Laggard): 業界リーダーを判断するための指標（ROEやNet Profit Margin）を取得
    if is_leader(ticker):
        output_log("L：◎業界リーダーになりうるとAIが判断します")
        score += 1
    else:
        output_log("L：業界リーダーとAIが判断しません")
        failed_conditions += 1

    # I (Institutional Sponsorship): 機関投資家の直近の動向をチェック
    retries = 0
    max_retries = 3
    while retries < max_retries:
        try: 
            major_holders = tickerInfo.major_holders
            institutional_holders = tickerInfo.institutional_holders
            if institutional_holders is not None and not institutional_holders.empty:
                # Date Reportedで直近のデータを絞り込む
                institutional_holders['Date Reported'] = pd.to_datetime(institutional_holders['Date Reported'])
                recent_date = pd.Timestamp.now() - pd.Timedelta(days=95)  # 直近8週間以内のデータを使用
                recent_data = institutional_holders[institutional_holders['Date Reported'] > recent_date]
                if not recent_data.empty and any(recent_data['Shares'] > 0):
                    output_log("I：◎機関投資家が直近で株を購入しています。")
                    score += 1
                    break
                else:
                    output_log("I：機関投資家の直近での購入が見当たりませんでした。")
                    print(institutional_holders)
                    # output_log(institutional_holders)
                    failed_conditions += 1
                    break
            else:
                output_log("I：機関投資家の保有データがありません。")
                failed_conditions += 1
                break
        except Exception as e:
            output_log(f"I：Error institutional_holders: {e} 再試行します... ({retries + 1}/{max_retries})")
            retries += 1
            time.sleep(2)  # 2秒待ってから再試行

    if retries == max_retries:
        output_log("I：最大試行回数に達しました。データを取得できませんでした。")
        failed_conditions += 1

    # M (Market Direction): 市場全体のトレンドを確認（S&P500とダウ平均）
    dow_data = save_ticker_auto('^GSPC')
    weekly_dow_data = convert_to_weekly(data_filter_date(dow_data, 180))
    dow_uptrend = util_can_slim_type.is_trend(weekly_dow_data, trend_type="uptrend")

    sp500_data = save_ticker_auto('^DJI')
    weekly_sp500_data = convert_to_weekly(data_filter_date(sp500_data, 180))
    sp500_uptrend = util_can_slim_type.is_trend(weekly_sp500_data, trend_type="uptrend")

    if dow_uptrend or sp500_uptrend:
        if dow_uptrend: score += 0.5
        if sp500_uptrend: score += 0.5
        output_log(f"M：◎市場全体が上昇トレンドです。 dow:{dow_uptrend} / sp500:{sp500_uptrend}")
    else:
        failed_conditions += 1
        output_log("M：市場全体が上昇トレンドではありません")

    output_log(f"評価：{score}/{score + failed_conditions}点")
    output_log(f"☆☆CAN-SLIM評価：終了☆☆\n")
    return score, failed_conditions

def analyst_eval(ticker):
    etf = yf.Ticker(ticker)
    # データ取得
    recommendations = get_data_with_retry(lambda: etf.recommendations)
    recommendations_summary = get_data_with_retry(lambda: etf.recommendations_summary)
    earnings_dates = get_data_with_retry(lambda: etf.earnings_dates)
    analyst_price_targets = get_data_with_retry(lambda: etf.analyst_price_targets)
    revenue_estimate = get_data_with_retry(lambda: etf.revenue_estimate)
    earnings_estimate = get_data_with_retry(lambda: etf.earnings_estimate)
    eps_trend = get_data_with_retry(lambda: etf.eps_trend)

    # 各情報を文字列に変換して結合
    combined_info = f"Recommendations: {recommendations}\n" \
                    f"Recommendations Summary: {recommendations_summary}\n" \
                    f"Earnings Dates: {earnings_dates}\n" \
                    f"Analyst Price Targets: {analyst_price_targets}\n" \
                    f"Revenue Estimate: {revenue_estimate}\n" \
                    f"Earnings Estimate: {earnings_estimate}\n" \
                    f"EPS Trend: {eps_trend}"
                    # f"Upgrades Downgrades: {upgrades_downgrades}\n" \

    have_finance_info = ""
    have_finance=read_news_from_csv('./csv/IHaveFinance.csv', 'shift_jis', ticker)
    if have_finance.count("\n") > 0:
        have_finance_info = f"現在の株の保有数は下記です。全てあるいは部分的に売る必要があるときは教えてください。\n{have_finance}"

    prompt = PROMPT_USER["ANALYST_EVAL"].format(
        ticker=ticker,
        analyst=combined_info,
        current_date=datetime.now(),
        have_finance_info=have_finance_info,
    )
    response = get_ai_opinion(prompt, None)
    return response

def is_new_news(news_list):
    prompt = PROMPT_USER["NEW_PRODUCT"].format(
        product=news_list,
    )
    response = get_ai_opinion(prompt, PROMPT_SYSTEM["IS_TRUE"])
    return response

def is_leader(ticker):
    prompt = PROMPT_USER["LEADER"].format(
        ticker=ticker,
    )
    response = get_ai_opinion(prompt, PROMPT_SYSTEM["IS_TRUE"])
    return response

# 過去データから売買価格を決定（上方チャネルラインを計算）
def get_buy_sell_price(ticker, date = 720):
    data = save_ticker_auto(ticker, date)
    sp500_data = save_ticker_auto('^DJI', date)
    dow_data = save_ticker_auto('^GSPC', date)
    folder_path = create_folder_path(ticker)

    # ベースグラフを作成
    create_plot_pattern(data, folder_path)

    # 買い価格を決定
    get_buy_price(data, folder_path)

    # 売り価格を決定
    get_sell_price(data, sp500_data, dow_data, image_folder = folder_path)

    # 損切価格を決定
    # get_buy_price(data)

def create_plot_pattern(data, image_folder=None):
    # ベースの図を作成
    util_can_slim_type.plot_pattern(data=data, title='day', image_name='0A_base_day.png', image_folder=image_folder, is_output_log=False)

    # ベースの図を作成
    weekly_data = convert_to_weekly(data)
    util_can_slim_type.plot_pattern(data=weekly_data, title='week', image_name='0B_base_week.png', image_folder=image_folder, is_output_log=False)

def get_buy_price(data, image_folder=None, is_cup_with_handle=True, is_saucer_with_handle=True, is_double_bottom=True
                  , is_flat_base=False, is_ascending_base=False, is_consolidation=False, is_vcp=True):
    output_log(f"\n☆☆買い価格：開始☆☆")
    if is_cup_with_handle:
        weekly_data = convert_to_weekly(data_filter_date(data, 180))
        pattern_found, purchase_price, left_peak, cup_bottom, right_peak = util_can_slim_type.detect_cup_with_handle(weekly_data, image_folder=image_folder)
        if pattern_found:
            output_log(f"取っ手付きカップ型が検出されました。購入価格は {purchase_price} です。")

    if is_double_bottom:
        weekly_data = convert_to_weekly(data_filter_date(data, 180))
        pattern_found, purchase_price, first_bottom, second_bottom = util_can_slim_type.detect_double_bottom(weekly_data, image_folder=image_folder)
        if pattern_found:
            output_log(f"ダブルボトム型が検出されました。購入価格は {purchase_price} です。")

    if is_vcp:
        weekly_data = convert_to_weekly(data_filter_date(data, 180))
        pattern_found, purchase_price = util_can_slim_type.detect_vcp(weekly_data, image_folder=image_folder)
        if pattern_found:
            output_log(f"VCPパターンが検出されました。購入価格は {purchase_price} です。")

    if is_saucer_with_handle:
        weekly_data = convert_to_weekly(data_filter_date(data, 360))
        pattern_found, purchase_price, left_peak, saucer_bottom, right_peak = util_can_slim_type.detect_saucer_with_handle(weekly_data, image_folder=image_folder)
        if pattern_found:
            output_log(f"取っ手付きソーサー型が検出されました。購入価格は {purchase_price} です。")

    # if is_flat_base:
    #     weekly_data = convert_to_weekly(data_filter_date(data, 60))
    #     pattern_found, purchase_price = util_can_slim_type.detect_flat_base(weekly_data, image_folder=image_folder)
    #     if pattern_found:
    #         output_log(f"フラットベース型が検出されました。購入価格は {purchase_price} です。")

    # if is_ascending_base:
    #     weekly_data = convert_to_weekly(data_filter_date(data, 180))
    #     pattern_found, purchase_price = util_can_slim_type.detect_ascending_base(weekly_data, image_folder=image_folder)
    #     if pattern_found:
    #         output_log(f"上昇トライアングル型が検出されました。購入価格は {purchase_price} です。")

    # is_cup_without_handle, left, bottom, right = util_can_slim_type.detect_cup_without_handle(data)
    # if is_cup_without_handle:
    #     output_log(f"取っ手なしカップ型検出: 左ピーク={left}, カップ底={bottom}, 右ピーク={right}")
    #     buy_price = calculate_buy_price(data, right)
    #     output_log(f"推奨購入価格: {buy_price:.2f}")

    output_log(f"☆☆買い価格：終了☆☆")

def get_sell_price(data, sp500_data, dow_data, image_folder=None, is_upper_channel_line=True, is_climax_top=True, is_exhaustion_gap=True,
                    is_railroad_tracks=True, is_double_top=True, is_market_downtrend=True, is_moving_average_break=True):
    output_log(f"\n☆☆売り価格：開始☆☆")
    if is_market_downtrend:
        weekly_data = convert_to_weekly(data_filter_date(data, 180))
        weekly_sp500_data = convert_to_weekly(data_filter_index(sp500_data, 180))
        weekly_dow_data = convert_to_weekly(data_filter_index(dow_data, 180))
        signal, price = util_can_slim_type.detect_market_downtrend(weekly_data, dow_data=weekly_dow_data, sp500_data=weekly_sp500_data, image_folder=image_folder)
        if signal:
            output_log(f"市場全体の下降トレンドが検出されました。売り価格は成行： {price} です。")

    if is_moving_average_break:
        weekly_data = convert_to_weekly(data_filter_date(data, 200))
        signal, price = util_can_slim_type.detect_moving_average_break(weekly_data, image_folder=image_folder)
        if signal:
            output_log(f"移動平均線を下回りました。売り価格は成行： {price} です。")

    if is_upper_channel_line:
        weekly_data = convert_to_weekly(data_filter_date(data, 360))
        signal, price = util_can_slim_type.detect_upper_channel_line(weekly_data, image_folder=image_folder)
        if signal:
            output_log(f"上方チャネルラインの売りシグナルが検出されました。売り価格は成行： {price} です。")

    if is_climax_top:
        weekly_data = convert_to_weekly(data_filter_date(data, 90))
        signal, price = util_can_slim_type.detect_climax_top(weekly_data, image_folder=image_folder)
        if signal:
            output_log(f"クライマックストップの売りシグナルが検出されました。売り価格は成行： {price} です。")

    # if is_exhaustion_gap:
    #     weekly_data = convert_to_weekly(data_filter(data, 90))
    #     signal, price = util_can_slim_type.detect_exhaustion_gap(weekly_data, image_folder=image_folder)
    #     if signal:
    #         output_log(f"イグゾーストキャップの売りシグナルが検出されました。売り価格は {price} です。")

    # if is_double_top:
    #     weekly_data = convert_to_weekly(data_filter(data, 180))
    #     signal, price = util_can_slim_type.detect_double_top(weekly_data, image_folder=image_folder)
    #     if signal:
    #         output_log(f"ダブルトップの売りシグナルが検出されました。売り価格は {price} です。")

    # if is_railroad_tracks:
    #     weekly_data = convert_to_weekly(data_filter(data, 90))
    #     signal, price = util_can_slim_type.detect_railroad_tracks(weekly_data, image_folder=image_folder)
    #     if signal:
    #         output_log(f"レールロードトラックの売りシグナルが検出されました。売り価格は {price} です。")

    output_log(f"☆☆売り価格：終了☆☆")

def get_output_log_file_path(folder_name=None, file_name=None, is_time_str=False):
    if is_time_str:
        time_str = datetime.now().strftime('%H%M')
        file_name = f'{file_name}_{time_str}.txt'
    else:
        file_name = f'{file_name}.txt'

    file_path = os.path.join(create_folder_path(folder_name), file_name)
    return file_path

def set_output_log_file_path(folder_name=None, file_name=None, is_time_str=False, is_clear=False):
    global log_file_path
    if is_clear:
        log_file_path = None
        return
    log_file_path = get_output_log_file_path(folder_name, file_name, is_time_str)

def output_log(message, is_json = False, title = None, is_print_only = False, tmp_file_path = None):
    global log_file_path
    out_message = ''
    if title is not None:
        out_message = title

    if is_json:
        out_message += json.dumps(message, indent=4, ensure_ascii=False)
    else:
        out_message += str(message)
    print(str(message))

    if is_print_only:
        return

    if log_file_path:
        try:
            if tmp_file_path is None:
                with open(log_file_path, 'a', encoding='utf-8') as file:
                    file.write(out_message + '\n')
            else:
                with open(tmp_file_path, 'a', encoding='utf-8') as file:
                    file.write(out_message + '\n')
        except Exception as e:
            print(f"ファイルの保存中にエラーが発生しました: {e}")
    return message

def get_log_text():
    global log_file_path
    if log_file_path:
        try:
            with open(log_file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print("指定されたファイルが存在しません。")
        except Exception as e:
            print(f"ファイルの読み込み中にエラーが発生しました: {e}")
    else:
        print("ログファイルパスが設定されていません。")
    return None

def send_line_log_text(is_include_https = True):
    text = get_log_text()
    if text:
        if is_include_https is False:
            filtered_text = [line for line in text.split('\n') if 'https' not in line]
            text = '\n'.join(filtered_text)
        if text:
            send_line_notify(text)

def analyst_eval_send(ticker):
    eval = analyst_eval(ticker)
    eval_clean = re.sub(r'[\*\#\_]+', '', eval)
    output_log(eval_clean, tmp_file_path = get_output_log_file_path(ticker, 'eval', True))
    send_line_notify(f"\n★★★{ticker}★★★\n" + eval_clean)

def sample(ticker):
    print(ticker)
    # dl = Downloader("./history/", email_address="ee68028@gmail.com")
    # dl.get("SC 13G", ticker)  # 'AAPL'の13Fフォームをダウンロード
    # dl.get("SC 13G", ticker, limit=1, download_details=False)
