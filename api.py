import requests
from datetime import datetime
import pytz

API_KEY = 'WMvFsMAutRouFV1iWzvyDmBfTRqjKKVu'  # ここにAPIキーを入力してください

# Top Stories APIのエンドポイント
urls = ["https://api.nytimes.com/svc/search/v2/articlesearch.json?q=business&begin_date=20230101&end_date=20230131",
        ]

# urls = ["https://api.nytimes.com/svc/topstories/v2/business.json",
#         "https://api.nytimes.com/svc/topstories/v2/politics.json",
#         "https://api.nytimes.com/svc/topstories/v2/technology.json",
#         "https://api.nytimes.com/svc/topstories/v2/realestate.json",
#         "https://api.nytimes.com/svc/topstories/v2/world.json",
#         "https://api.nytimes.com/svc/topstories/v2/us.json",
#         ]
# url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q=election&api-key=WMvFsMAutRouFV1iWzvyDmBfTRqjKKVu"
# url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q=business&begin_date=20230101&end_date=20230131&api-key=WMvFsMAutRouFV1iWzvyDmBfTRqjKKVu"
# url = f"https://api.nytimes.com/svc/topstories/v2/business.json?begin_date=20230101&end_date=20230131&api-key=WMvFsMAutRouFV1iWzvyDmBfTRqjKKVu"
# url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q=finance&begin_date=20230101&end_date=20230131&api-key=WMvFsMAutRouFV1iWzvyDmBfTRqjKKVu"

def get_top_story(url):
    # APIリクエストのURLを生成
    full_url = f"{url}&api-key={API_KEY}"

    # APIリクエストを送信
    response = requests.get(full_url)

    # レスポンスのステータスコードを確認
    if response.status_code == 200:
        # レスポンスからJSONデータを取得
        data = response.json()
        
        # トップニュースを表示
        # for article in data['results']:
        #     # published_dateを日本時間に変換
        #     dt = datetime.fromisoformat(article['published_date'])
        #     dt_japan = dt.astimezone(pytz.timezone('Asia/Tokyo'))
        #     published_date_jst = dt_japan.strftime("%Y-%m-%d %H:%M:%S %Z")
            
        #     print(f"{published_date_jst} - {article['abstract']}")
        for article in data['response']['docs']:
            # published_dateを日本時間に変換
            dt = datetime.fromisoformat(article['pub_date'])
            dt_japan = dt.astimezone(pytz.timezone('Asia/Tokyo'))
            published_date_jst = dt_japan.strftime("%Y-%m-%d %H:%M:%S %Z")
            
            print(f"{published_date_jst} - {article['abstract']}")
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")

def get_top_stories(urls):
    for url in urls:
        get_top_story(url)

get_top_stories(urls)