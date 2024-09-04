import requests
import xml.etree.ElementTree as ET

# ニュースURL
NEWS_URL = "https://www.axios.com/"

# RSSフィードを取得
response = requests.get(NEWS_URL)
response.raise_for_status()  # リクエストが成功したかを確認

# XMLの解析
root = ET.fromstring(response.content)

# ニュース記事のタイトルとリンクを取得
for item in root.findall('./channel/item'):
    title = item.find('title').text
    print(f"Title: {title}")
