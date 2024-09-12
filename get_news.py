import csv
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import pytz
import time
from prompts import PROMPT_CSV_TO_COMPACT
import openai
from main import API_KEY
openai.api_key = API_KEY
GPT_MODEL = "gpt-4o"

API_KEY = 'WMvFsMAutRouFV1iWzvyDmBfTRqjKKVu'  # ここにAPIキーを入力してください
IS_COMPACT_AI = False

def get_new_york(url):
    news_data = []
    # APIリクエストのURLを生成
    full_url = f"{url}&api-key={API_KEY}"

    try:
        # リクエストを送信
        response = requests.get(full_url)

        # レスポンスのステータスコードを確認
        if response.status_code == 200:
            data = response.json()
            for article in data['response']['docs']:
                try:
                    dt = datetime.fromisoformat(article['pub_date'])
                    dt_japan = dt.astimezone(pytz.timezone('Asia/Tokyo'))
                    published_date_jst = dt_japan.strftime("%Y-%m-%d")
                    # published_date_jst = dt_japan.strftime("%Y-%m-%d %H:%M:%S %Z")
                    
                    news_data.append([published_date_jst, article['abstract']])
                    print(f"{published_date_jst} - {article['abstract']}")
                except ValueError as e:
                    print(f"日付の変換に失敗しました: {e}")
        elif response.status_code == 429:
            print("リクエストが多すぎます。少し待機して再試行します...")
            time.sleep(10)  # 10秒待機して再試行
            return get_new_york(url)
        else:
            print(f"データの取得に失敗しました。ステータスコード: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"リクエストの送信中にエラーが発生しました: {e}")

    return news_data

def fetch_news_soup(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'xml')
        return soup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return None

def fetch_news_titles(soup):
    items = soup.find_all('item')
    news_data = []
    for item in items:
        pub_date = item.pubDate.text.strip()
        utc_date = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
        jst_date = utc_date + timedelta(hours=9)  # UTCからJSTに変換
        formatted_date = jst_date.strftime('%Y/%m/%d %H:%M')
        title = item.title.text.strip()
        news_data.append([formatted_date, title])
    return news_data

def append_to_csv(file_name, news_data):
    # 既存のCSVファイルの内容をリストとして読み込む
    existing_data = []
    try:
        with open(file_name, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            existing_data = list(reader)  # 既存の行を全てリストとして格納
    except FileNotFoundError:
        pass  # ファイルが存在しない場合はスキップ

    # 新しいデータが既に存在するかチェック
    new_data_to_add = []
    for row in news_data:
        if row not in existing_data:  # 新しいデータが既存データに無い場合のみ追加
            new_data_to_add.append(row)

    # 新しいデータを追記モードでCSVに書き込む
    if new_data_to_add:
        with open(file_name, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(new_data_to_add)
        print(f"{len(new_data_to_add)} 件の新しいデータを追加しました。")
    else:
        print("新しいデータはありません。")

def generate_date_ranges(start_date, end_date):
    # 隔月で日付範囲を生成する
    date_ranges = []
    current_date = start_date
    while current_date < end_date:
        next_date = current_date + timedelta(days=30)  # 30日後を次の範囲の開始日とする
        if next_date > end_date:
            next_date = end_date
        date_ranges.append((current_date.strftime('%Y%m%d'), next_date.strftime('%Y%m%d')))
        current_date = next_date
    return date_ranges

def read_news_from_csv(file_path, encoding='utf-8'):
    news_list = []
    latest_date = None
    try:
        with open(file_path, mode='r', encoding=encoding) as file:
            reader = csv.reader(file)
            for row in reader:
                news_date = row[0]
                if not latest_date or news_date > latest_date:
                    latest_date = news_date  # 最新の日付を取得
                news_content = ",".join(row[1:])
                news_list.append(f"{news_date} - {news_content}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    return "\n".join(news_list), latest_date

def main(use_latest_csv_date=False):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)

    # 最新日付を使うオプションが有効な場合、CSVから最新の日付を取得
    if use_latest_csv_date:
        _, latest_date = read_news_from_csv('news_data.csv')
        if latest_date:
            start_date = datetime.strptime(latest_date, '%Y-%m-%d')

    # start_dateとend_dateが同じ場合、returnする
    if start_date.date() == end_date.date():
        print("ニュースの追加はありません。")
        return
    date_ranges = generate_date_ranges(start_date, end_date)

    for start, end in date_ranges:
        print(f"Fetching news from {start} to {end}")
        # base_url = "https://news.google.com/rss/search"
        # query = f"?q=after:{start}+before:{end}&hl=ja&gl=JP&ceid=JP:ja"
        base_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        # query = f"?q=stock+market&begin_date={start}&end_date={end}&api-key={API_KEY}"
        # query = f"?q=business&begin_date={start}&end_date={end}&api-key={API_KEY}"
        query = f"?q=business+OR+stock+market&begin_date={start}&end_date={end}&api-key={API_KEY}"
        url = base_url + query
        print(url)

        news_data = get_new_york(url)
        # news_data = fetch_news_titles(soup)
        append_to_csv('news_data.csv', news_data)
        # 次の処理があるか確認してtime.sleep(10)を実行
        if (start, end) != date_ranges[-1]:
            time.sleep(10)
        # soup = fetch_news_soup(url)
        # if soup:
        #     news_data = get_new_york(soup)
        #     # news_data = fetch_news_titles(soup)
        #     append_to_csv('news_data.csv', news_data)

    if IS_COMPACT_AI:
        # プロンプトの生成
        all_news_data, _ = read_news_from_csv('news_data.csv')
        prompt = PROMPT_CSV_TO_COMPACT + all_news_data
        ai_opinion = get_ai_opinion(prompt)

        print(ai_opinion)
        ai_opinion_cleaned = re.sub(r'[\*\#\_]+', '', ai_opinion)
        print(ai_opinion_cleaned)
        # AIの意見をCSVに出力
        with open('ai_opinion.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['AI Opinion'])  # ヘッダー
            writer.writerow([ai_opinion_cleaned])  # 内容

# AIの意見を取得
def get_ai_opinion(prompt):
    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.1
    )

    print(response.usage)
    return response.choices[0].message.content

if __name__ == "__main__":
    main(use_latest_csv_date=True)
