import yfinance as yf
import os
import sys
from datetime import datetime
import pandas as pd
import json
import math
# https://qiita.com/aguilarklyno/items/51622f9efc33aac88bbf

tickers = sys.argv[1].split(',') if len(sys.argv) > 1 else []

def filter_news(news):
    # Convert the timestamp to a date and extract title only
    return [{"title": article["title"], "date": datetime.utcfromtimestamp(article["providerPublishTime"]).strftime('%Y-%m-%d')} for article in news if "providerPublishTime" in article and "title" in article]

def filter_info(info):
    # 株価の未来予測に直接関連する重要な情報のみ抽出
    important_keys = ['sector', 'industry', 'marketCap', 'beta', 'forwardPE', 'priceToSalesTrailing12Months']
    return {key: info[key] for key in important_keys if key in info}

def filter_financials(financials):
    # 損益計算書から重要な項目のみ抽出
    important_keys = ['Total Revenue', 'Gross Profit', 'Net Income', 'Operating Income', 'EBITDA']
    return {key: financials[key] for key in important_keys if key in financials}

# 再帰的にNaN、None、空のデータを削除する関数
def clean_data(data):
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items() if v is not None and v != {} and v != [] and v != "" and (not isinstance(v, float) or not math.isnan(v))}
    elif isinstance(data, list):
        return [clean_data(item) for item in data if item is not None and item != "" and (not isinstance(item, float) or not math.isnan(item))]
    return data

# 再帰的にデータをJSONシリアライズ可能な形式に変換する関数
def convert_to_serializable(data):
    if isinstance(data, pd.DataFrame):
        return {str(k): convert_to_serializable(v) for k, v in data.to_dict().items()}  # DataFrameのキーと値を再帰的に変換
    elif isinstance(data, pd.Series):
        return {str(k): convert_to_serializable(v) for k, v in data.to_dict().items()}  # Seriesのキーと値を再帰的に変換
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()  # TimestampをISOフォーマットの文字列に変換
    elif isinstance(data, dict):
        return {str(k): convert_to_serializable(v) for k, v in data.items()}  # 辞書のキーと値を再帰的に変換
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]  # リストの要素を再帰的に変換
    return data  # それ以外のデータ型はそのまま返す

def save_etf_json_data(ticker, file_path):
    yfTicker = yf.Ticker(ticker)
    
    # データを取得して辞書にまとめる
    data = {
        # 基本情報 (例: 企業名、業種、市場など)
        "info": filter_info(yfTicker.info),
        # アクション (配当金や株式分割の履歴)
        # "actions": convert_to_serializable(yfTicker.actions),
        # 配当履歴
        # "dividends": convert_to_serializable(yfTicker.dividends),
        # 株式分割の履歴
        # "splits": convert_to_serializable(yfTicker.splits),
        # キャピタルゲイン（保有株の資本利益）
        # "capital_gains": convert_to_serializable(yfTicker.capital_gains),
        # 年次の損益計算書 (例: 売上高、純利益、営業利益など)
        "income_statement": filter_financials(convert_to_serializable(yfTicker.financials)),
        # 四半期ごとの損益計算書
        "quarterly_income_statement": filter_financials(convert_to_serializable(yfTicker.quarterly_financials)),
        # 年次の貸借対照表 (例: 資産、負債、自己資本など)
        "balance_sheet": convert_to_serializable(yfTicker.balance_sheet),
        # 四半期ごとの貸借対照表
        "quarterly_balance_sheet": convert_to_serializable(yfTicker.quarterly_balance_sheet),
        # 年次のキャッシュフロー計算書 (例: 営業活動によるキャッシュフロー、投資活動など)
        "cashflow": convert_to_serializable(yfTicker.cashflow),
        # 四半期ごとのキャッシュフロー計算書
        "quarterly_cashflow": convert_to_serializable(yfTicker.quarterly_cashflow),
        # 主要株主の情報
        # "major_holders": convert_to_serializable(yfTicker.major_holders),
        # 機関投資家の保有株情報
        # "institutional_holders": convert_to_serializable(yfTicker.institutional_holders),
        # 投資信託が保有する株式の情報
        # "mutualfund_holders": convert_to_serializable(yfTicker.mutualfund_holders),
        # インサイダー取引情報
        # "insider_transactions": convert_to_serializable(yfTicker.insider_transactions),
        # インサイダーによる株式購入情報
        # "insider_purchases": convert_to_serializable(yfTicker.insider_purchases),
        # アナリストからの推奨情報 (例: 強い買い、買い、中立など)
        # "recommendations": convert_to_serializable(yfTicker.recommendations),
        # アナリストの推奨のサマリー情報
        "recommendations_summary": convert_to_serializable(yfTicker.recommendations_summary),
        # アナリストによる格上げ・格下げの履歴
        # "upgrades_downgrades": convert_to_serializable(yfTicker.upgrades_downgrades),
        # 収益発表の予定日や過去の収益発表日
        "earnings_dates": convert_to_serializable(yfTicker.earnings_dates),
        # 企業に関連する最新のニュース
        "news": filter_news(yfTicker.news)
    }

    # NaN、None、空のデータ（空の辞書、リスト、文字列など）を削除
    data = {k: v for k, v in data.items() if v is not None and v != {} and v != [] and v != "" and (not isinstance(v, float) or not math.isnan(v))}

    # 再帰的にNaN、None、空のデータを削除
    cleaned_data = clean_data(data)

    # JSONファイルに書き込む
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

# メイン処理
def main(ticker):
    folder_path = f'./'
    # file_path = os.path.join(folder_path, 'research.csv')
    # save_etf_data(ticker, file_path)
    file_path = os.path.join(folder_path, 'research.json')
    save_etf_json_data(ticker, file_path)

# 複数のティッカーに対してループ処理
if __name__ == "__main__":
    for ticker in tickers:
        main(ticker)
