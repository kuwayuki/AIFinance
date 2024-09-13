import yfinance as yf
import os
import sys
from datetime import datetime
import pandas as pd
import json
# https://qiita.com/aguilarklyno/items/51622f9efc33aac88bbf

# DEFAULT = ["CAT"]
DEFAULT = ["CVX","CAT","PEP","KO","BAC","GS","MS","JNJ","MRK","AAPL","NVDA","INTC","AVGO","GOOGL","MSFT","META","BABA","IBM","ORCL","VZ","ASML","LRCX","MCHP","ON","SWKS","CTSH","BIDU","NTES","WIT","EBAY","ABNB","EA","ZM","IQ","TME","AKAM","GEN","FFIV","DBX","CHKP","ZI","BOX","FIS","STNE","FUTU"]
# DEFAULT = ["CVX","CAT","PEP","KO","BAC","GS","MS","JNJ","MRK","AAPL","NVDA","INTC","AVGO","GOOGL","MSFT","META","BABA","IBM","ORCL","VZ"]
# DEFAULT = ["ASML","LRCX","MCHP","ON","SWKS","CTSH","BIDU","NTES","WIT","EBAY","ABNB","EA","ZM","IQ","TME","AKAM","GEN","FFIV","DBX","CHKP","ZI","BOX","FIS","STNE","FUTU"]
tickers = sys.argv[1].split(',') if len(sys.argv) > 1 else DEFAULT

def format_percentage(value):
    """値をパーセンテージ形式にフォーマットし、少数第1位まで表示"""
    try:
        if value is None or value != value:  # value != value はNaNをチェックするための式
            return "N/A"
        return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else value
    except:
        return "N/A"

def format_decimal(value):
    """値を少数第1位までフォーマット"""
    try:
        return f"{value:.1f}" if isinstance(value, (int, float)) else value
    except:
        return "N/A"

def get_yearly_high_low(ticker, year):
    """指定された年の最高値と最安値を取得"""
    etf = yf.Ticker(ticker)
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    try:
        hist = etf.history(start=start_date, end=end_date)
        yearly_high = hist['High'].max()
        yearly_low = hist['Low'].min()
        return {
            "yearly_high": format_decimal(yearly_high),
            "yearly_low": format_decimal(yearly_low)
        }
    except Exception as e:
        print(f"Error retrieving high/low for {ticker} in {year}: {e}")
        return {
            "yearly_high": "N/A",
            "yearly_low": "N/A"
        }

def calculate_profit_margins(ticker):
    """過去2期分と今年の利益率と売上高を計算"""
    etf = yf.Ticker(ticker)
    try:
        # 年次財務データを取得
        financials = etf.financials.T  # 年度ごとの財務データ
        margins = {
            "profit_margin_2y_ago": "N/A",
            "profit_margin_1y_ago": "N/A",
            "profit_margin_current": "N/A",
            "revenue_2y_ago": "N/A",
            "revenue_1y_ago": "N/A",
            "revenue_current": "N/A"
        }

        # 利益率の計算に使用するキーの順序
        profit_keys = ['Operating Income', 'Net Income', 'Gross Profit']

        for i, period in enumerate(["profit_margin_current", "profit_margin_1y_ago", "profit_margin_2y_ago"]):
            if len(financials) > i:
                revenue = financials['Total Revenue'].iloc[i]
                profit = None

                for key in profit_keys:
                    if key in financials.columns:
                        profit = financials[key].iloc[i]
                        break

                if revenue is None or revenue != revenue:  # チェックしてNoneやNaNの場合に備える
                    print(f"Revenue for {ticker} in {period} is missing or NaN.")
                    continue  # スキップして次の期間を確認
                if profit is None or profit != profit:  # チェックしてNoneやNaNの場合に備える
                    print(f"Profit for {ticker} in {period} is missing or NaN.")
                    continue  # スキップして次の期間を確認

                if profit is not None and revenue != 0:
                    margins[period] = format_percentage(profit / revenue)
                margins[f"revenue_{i}y_ago"] = format_decimal(revenue)

        return margins

    except Exception as e:
        print(f"Error calculating profit margins for {ticker}: {e}")
        return {
            "profit_margin_2y_ago": "N/A",
            "profit_margin_1y_ago": "N/A",
            "profit_margin_current": "N/A",
            "revenue_2y_ago": "N/A",
            "revenue_1y_ago": "N/A",
            "revenue_current": "N/A"
        }

def filter_cashflow_for_forecasting(cashflow_data):
    """
    キャッシュフローデータから未来予測に必要な項目だけを抽出する。
    """
    # 未来予測に重要と考えられる項目のみ保持
    keys_to_keep = ['Free Cash Flow', 'Operating Cash Flow', 'Net Income From Continuing Operations', 'Capital Expenditure']
    
    # キャッシュフローデータをフィルタリング
    filtered_cashflow = {}
    for date, values in cashflow_data.items():
        filtered_values = {key: values[key] for key in keys_to_keep if key in values}
        filtered_cashflow[date] = filtered_values
    
    return filtered_cashflow

def get_free_cash_flow_json(ticker):
    """フリーキャッシュフローをJSON形式で取得して、CSVの1セルに保存"""
    etf = yf.Ticker(ticker)
    try:
        cashflow = etf.cashflow
        if cashflow is not None:
            # DataFrameをJSON形式に変換して辞書に変換
            cashflow_dict = json.loads(cashflow.to_json())
            
            # フィルタリング関数を適用
            filtered_cashflow_dict = filter_cashflow_for_forecasting(cashflow_dict)
            
            # 辞書をJSON文字列に変換
            cashflow_json = json.dumps(filtered_cashflow_dict)
            
            # カンマや不要な文字を置換
            cashflow_json_cleaned = cashflow_json.replace(",", " ").replace("—", "-").replace("\xa0", " ")
            
            return cashflow_json_cleaned

        else:
            return "N/A"
    except Exception as e:
        print(f"Error retrieving cash flow for {ticker}: {e}")
        return "N/A"

def get_debt_to_equity_json(ticker):
    """負債比率を計算し、JSON形式で必要なデータをCSV用に整形"""
    etf = yf.Ticker(ticker)
    try:
        balance_sheet = etf.balance_sheet
        if balance_sheet is not None:
            # 必要なデータが存在するかを確認して取得
            total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else None
            total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else None
            net_debt = balance_sheet.loc['Net Debt'].iloc[0] if 'Net Debt' in balance_sheet.index else None

            # データを辞書に変換
            balance_sheet_data = {
                "Total Debt": total_debt,
                "Total Stockholder Equity": total_equity,
                "Net Debt": net_debt
            }

            # 辞書をJSON文字列に変換
            balance_sheet_json = json.dumps(balance_sheet_data)

            # カンマや不要な文字を置換して、CSVに書き込むために整形
            balance_sheet_json_cleaned = balance_sheet_json.replace(",", " ").replace("—", "-").replace("\xa0", " ")

            return balance_sheet_json_cleaned
        else:
            return "N/A"
    except Exception as e:
        print(f"Error retrieving balance sheet for {ticker}: {e}")
        return "N/A"

def clean_json(data):
    if isinstance(data, pd.DataFrame):
        # DataFrameを辞書に変換してからJSONに変換
        data = data.to_dict(orient='records')
    # JSON形式に変換し、カンマを全角カンマに、改行や特殊文字を削除
    recommendations_str = json.dumps(data).replace(',', ';').replace('\n', '').replace('\r', '')
    return recommendations_str

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

def save_etf_data(ticker, file_path):
    etf = yf.Ticker(ticker)
    
    # 現在の年を取得
    current_year = datetime.now().year

    # 利益率と売上高のデータを取得
    profit_margins = calculate_profit_margins(ticker)

    # フリーキャッシュフローと負債比率を取得
    # free_cash_flow = get_free_cash_flow_json(ticker)
    # debt_to_equity_ratio = get_debt_to_equity_json(ticker)
    recommendations_summary = etf.recommendations_summary
    print(recommendations_summary)
    recommendations_summary = clean_json(recommendations_summary)

    # 過去2年分と今年の最高値・最安値を現在の年から取得
    high_low_2y_ago = get_yearly_high_low(ticker, current_year - 2)
    high_low_1y_ago = get_yearly_high_low(ticker, current_year - 1)
    high_low_current = get_yearly_high_low(ticker, current_year)

    # 他の基本的なデータを取得
    per = format_decimal(etf.info.get("forwardPE", "N/A"))  # 予想PERを少数第1位まで
    trailing_per = format_decimal(etf.info.get("trailingPE", "N/A"))  # 実績PERを少数第1位まで
    roe = format_percentage(etf.info.get("returnOnEquity", "N/A"))  # ROE
    website = etf.info.get("website", "N/A")  # 企業URL
    current_price = format_decimal(etf.history(period="1d")['Close'].iloc[-1])  # 現在の株価

    # ニュースリストを";"で区切って連結し、カンマを別の文字（例えば、スペースや「|」）に置換
    news_str = ";".join([news_item['title'].replace(",", " ").replace("—", "-").replace("\u2013", " ").replace("\xa0", " ").replace("\xae", " ").replace("\u014c", " ").replace("\u2122", " ").replace('\u02bb', " ") for news_item in etf.news])

    # CSV ファイルが存在しない場合はヘッダーを追加して新規作成
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='shift_jis', errors='replace') as f:
            f.write("Ticker,利益率(2期前),利益率(1期前),利益率(今年),実績PER,予想PER,ROE,売上高(2期前),売上高(1期前),現在の株価,"
                    "最高値(2期前),最安値(2期前),最高値(1期前),最安値(1期前),最高値(今年),最安値(今年),ニュース,企業URL, アナリスト評価\n")
                    # "フリーキャッシュフロー,負債比率\n")
    
    # データを追記
    with open(file_path, 'a', encoding='shift_jis', errors='replace') as f:
        f.write(f"{ticker},{profit_margins['profit_margin_2y_ago']},{profit_margins['profit_margin_1y_ago']},"
                f"{profit_margins['profit_margin_current']},{trailing_per},{per},{roe},"
                f"{profit_margins['revenue_2y_ago']},{profit_margins['revenue_1y_ago']},{current_price},"
                f"{high_low_2y_ago['yearly_high']},{high_low_2y_ago['yearly_low']},"
                f"{high_low_1y_ago['yearly_high']},{high_low_1y_ago['yearly_low']},"
                f"{high_low_current['yearly_high']},{high_low_current['yearly_low']},"
                f"{news_str},{website}, {recommendations_summary} \n")
                # f"{news_str},{website},{free_cash_flow},{debt_to_equity_ratio}\n")

# メイン処理
def main(ticker):
    folder_path = f'./'
    file_path = os.path.join(folder_path, 'research.csv')
    save_etf_data(ticker, file_path)

# 複数のティッカーに対してループ処理
if __name__ == "__main__":
    for ticker in tickers:
        main(ticker)