import yfinance as yf
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import utils
import json

# デフォルトのティッカーシンボルのリスト
DEFAULT = ["9697.T"]

tickers = sys.argv[1].split(',') if len(sys.argv) > 1 else DEFAULT

def format_percentage(value):
    """値をパーセンテージ形式にフォーマットし、少数第1位まで表示"""
    try:
        if value is None or value != value:  # NaNチェック
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
            "利益率(2期前)": "N/A",
            "利益率(1期前)": "N/A",
            "利益率(今年)": "N/A",
            "売上高(2期前)": "N/A",
            "売上高(1期前)": "N/A",
            "売上高(今年)": "N/A"
        }

        # 利益率の計算に使用するキーの順序
        profit_keys = ['Operating Income', 'Net Income', 'Gross Profit']

        for i, period in enumerate(["利益率(今年)", "利益率(1期前)", "利益率(2期前)"]):
            if len(financials) > i:
                revenue = financials['Total Revenue'].iloc[i]
                profit = None

                for key in profit_keys:
                    if key in financials.columns:
                        profit = financials[key].iloc[i]
                        break

                if revenue is None or revenue != revenue:  # NoneやNaNのチェック
                    print(f"Revenue for {ticker} in {period} is missing or NaN.")
                    continue
                if profit is None or profit != profit:
                    print(f"Profit for {ticker} in {period} is missing or NaN.")
                    continue

                if profit is not None and revenue != 0:
                    margins[period] = format_percentage(profit / revenue)

                if i == 0:
                    margins["売上高(今年)"] = format_decimal(revenue)
                else:
                    margins[f"売上高({i}期前)"] = format_decimal(revenue)

        return margins

    except Exception as e:
        print(f"Error calculating profit margins for {ticker}: {e}")
        return {
            "利益率(2期前)": "N/A",
            "利益率(1期前)": "N/A",
            "利益率(今年)": "N/A",
            "売上高(2期前)": "N/A",
            "売上高(1期前)": "N/A",
            "売上高(今年)": "N/A"
        }

def clean_json(data):
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient='records')
    recommendations_str = json.dumps(data).replace(',', ';').replace('\n', '').replace('\r', '')
    return recommendations_str

def get_canslim_data_dict(ticker):
    """CAN-SLIM法のデータを辞書形式で取得"""
    stock = yf.Ticker(ticker)
    data = {}

    # 発行済株式数を取得（EPS計算に使用）
    shares_outstanding = stock.info.get('sharesOutstanding', None)
    if shares_outstanding is None:
        print(f"{ticker}: 発行済株式数が取得できませんでした。")
        return data  # 空の辞書を返す

    # 現在の四半期の純利益増加率（C）
    quarterly_financials = stock.quarterly_financials
    if quarterly_financials is not None and not quarterly_financials.empty and 'Net Income' in quarterly_financials.index:
        if quarterly_financials.shape[1] >= 2:
            latest_quarter = quarterly_financials.columns[0]
            previous_quarter = quarterly_financials.columns[1]
            current_net_income = quarterly_financials.loc['Net Income', latest_quarter]
            previous_net_income = quarterly_financials.loc['Net Income', previous_quarter]
            if previous_net_income != 0 and current_net_income is not None and previous_net_income is not None:
                net_income_growth = (current_net_income - previous_net_income) / abs(previous_net_income) * 100
                data['四半期純利益成長率(%)'] = format_decimal(net_income_growth)

                # EPSの計算
                current_eps = current_net_income / shares_outstanding
                previous_eps = previous_net_income / shares_outstanding
                if previous_eps != 0:
                    eps_growth = (current_eps - previous_eps) / abs(previous_eps) * 100
                    data['四半期EPS成長率(%)'] = format_decimal(eps_growth)

    # 年次利益の増加（A）
    annual_financials = stock.financials
    if annual_financials is not None and not annual_financials.empty and 'Net Income' in annual_financials.index:
        net_income_history = annual_financials.loc['Net Income']
        if len(net_income_history) >= 2:
            current_year = net_income_history.index[0]
            previous_year = net_income_history.index[1]
            current_net_income = net_income_history[current_year]
            previous_net_income = net_income_history[previous_year]
            if previous_net_income != 0 and current_net_income is not None and previous_net_income is not None:
                annual_net_income_growth = (current_net_income - previous_net_income) / abs(previous_net_income) * 100
                data['年次純利益成長率(%)'] = format_decimal(annual_net_income_growth)

                # EPSの計算
                current_eps = current_net_income / shares_outstanding
                previous_eps = previous_net_income / shares_outstanding
                if previous_eps != 0:
                    annual_eps_growth = (current_eps - previous_eps) / abs(previous_eps) * 100
                    data['年次EPS成長率(%)'] = format_decimal(annual_eps_growth)

    # 株価と52週高値の比率
    history = stock.history(period='1y')
    if not history.empty:
        current_price = history['Close'].iloc[-1]
        high_52week = history['High'].max()
        data['CANSLIM現在の株価'] = format_decimal(current_price)
        data['52週高値'] = format_decimal(high_52week)
        if high_52week != 0:
            price_vs_high = (current_price / high_52week) * 100
            data['株価と52週高値の比率(%)'] = format_decimal(price_vs_high)

    # 需給関係（S）
    data['発行済株式数'] = shares_outstanding

    float_shares = stock.info.get('floatShares', None)
    if float_shares is not None:
        data['浮動株数'] = float_shares

    insider_ownership = stock.info.get('heldPercentInsiders', None)
    if insider_ownership is not None:
        data['インサイダー保有率(%)'] = format_percentage(insider_ownership)

    institutional_ownership = stock.info.get('heldPercentInstitutions', None)
    if institutional_ownership is not None:
        data['機関投資家保有率(%)'] = format_percentage(institutional_ownership)

    # 機関投資家の保有データを取得
    try:
        institutional_holders = stock.institutional_holders
        if institutional_holders is not None and not institutional_holders.empty:
            data['機関投資家の数'] = len(institutional_holders)
            data['機関投資家の合計保有株数'] = institutional_holders['Shares'].sum()
        else:
            data['機関投資家の数'] = "N/A"
            data['機関投資家の合計保有株数'] = "N/A"
    except Exception as e:
        print(f"{ticker}: Error retrieving institutional holders: {e}")
        data['機関投資家の数'] = "N/A"
        data['機関投資家の合計保有株数'] = "N/A"

    # 市場の方向性（M）
    spy = yf.Ticker('^GSPC')
    spy_history = spy.history(period='1y')
    if not spy_history.empty:
        spy_current_price = spy_history['Close'].iloc[-1]
        spy_price_1y_ago = spy_history['Close'].iloc[0]
        if spy_price_1y_ago != 0:
            spy_return = (spy_current_price - spy_price_1y_ago) / spy_price_1y_ago * 100
            data['市場リターン(S&P 500 1年)(%)'] = format_decimal(spy_return)

    return data


def calculate_garp_score(ticker):
    """GARP投資法のデータを辞書形式で取得"""
    stock = yf.Ticker(ticker)
    data = {}
    try:
        # 発行済株式数を取得
        shares_outstanding = stock.info.get('sharesOutstanding', None)
        if shares_outstanding is None:
            print(f"{ticker}: sharesOutstanding not found.")
            data['EPS成長率'] = "N/A"
            data['PER'] = "N/A"
            data['PEGレシオ'] = "N/A"
            return data

        # 過去の四半期EPSを取得
        quarterly_financials = stock.quarterly_financials
        if quarterly_financials.empty or 'Net Income' not in quarterly_financials.index:
            print(f"{ticker}: Quarterly financials data is insufficient.")
            data['EPS成長率'] = "N/A"
            data['PER'] = "N/A"
            data['PEGレシオ'] = "N/A"
            return data

        net_income_history = quarterly_financials.loc['Net Income']
        if len(net_income_history) < 4:
            print(f"{ticker}: Not enough net income data.")
            data['EPS成長率'] = "N/A"
            data['PER'] = "N/A"
            data['PEGレシオ'] = "N/A"
            return data

        # EPSを計算
        eps_history = net_income_history / shares_outstanding
        # EPS成長率を計算
        eps_growth_rates = eps_history.pct_change()
        eps_growth_rate = eps_growth_rates.mean()
        # PERを取得
        per = stock.info.get('forwardPE', None)
        # PEGレシオを計算
        if per and eps_growth_rate and eps_growth_rate != 0:
            peg_ratio = per / (eps_growth_rate * 100)
            data['EPS成長率'] = format_percentage(eps_growth_rate)
            data['Investment PER'] = format_decimal(per)  # 'PER' を 'Investment PER' に変更
            data['PEGレシオ'] = format_decimal(peg_ratio)
        else:
            data['EPS成長率'] = "N/A"
            data['Investment PER'] = format_decimal(per) if per else "N/A"  # 'PER' を 'Investment PER' に変更
            data['PEGレシオ'] = "N/A"
    except Exception as e:
        print(f"Error calculating GARP score for {ticker}: {e}")
        data['EPS成長率'] = "N/A"
        data['PER'] = "N/A"
        data['PEGレシオ'] = "N/A"
    return data

def calculate_magic_formula(ticker):
    """マジックフォーミュラ投資法のデータを辞書形式で取得"""
    stock = yf.Ticker(ticker)
    data = {}
    try:
        # 必要な財務データを取得
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        if financials.empty or balance_sheet.empty:
            print(f"{ticker}: Financials or balance sheet data is empty.")
            data['ROIC'] = "N/A"
            data['利益利回り'] = "N/A"
            return data
        # ROICの計算（簡易版）
        operating_income = financials.loc['Operating Income'].iloc[0]
        total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0
        total_liabilities = balance_sheet.loc['Total Liab'].iloc[0] if 'Total Liab' in balance_sheet.index else 0

        if total_assets == 0 and 'Total Asset' in balance_sheet.index:
            total_assets = balance_sheet.loc['Total Asset'].iloc[0]

        invested_capital = total_assets - total_liabilities
        if invested_capital != 0:
            roic = operating_income / invested_capital
            data['ROIC'] = format_percentage(roic)
        else:
            data['ROIC'] = "N/A"
        # 利益利回りの計算
        enterprise_value = stock.info.get('enterpriseValue', None)
        if enterprise_value and operating_income:
            earnings_yield = operating_income / enterprise_value
            data['利益利回り'] = format_percentage(earnings_yield)
        else:
            data['利益利回り'] = "N/A"
    except Exception as e:
        print(f"Error calculating Magic Formula for {ticker}: {e}")
        data['ROIC'] = "N/A"
        data['利益利回り'] = "N/A"
    return data

def calculate_piotroski_score(ticker):
    """ピオトロスキー・スコアを計算"""
    stock = yf.Ticker(ticker)
    score = 0
    data = {}
    data['更新日'] = datetime.now().strftime("%Y-%m-%d")
    try:
        # 財務諸表を取得
        income_statement = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        if income_statement.empty or balance_sheet.empty or cash_flow.empty:
            print(f"{ticker}: Financial statements are empty.")
            data['Fスコア'] = "N/A"
            return data

        # 過去2期間のデータを取得
        if income_statement.shape[1] < 2 or balance_sheet.shape[1] < 2 or cash_flow.shape[1] < 2:
            print(f"{ticker}: Not enough data for Piotroski Score.")
            data['Fスコア'] = "N/A"
            return data

        # ROA
        net_income = income_statement.loc['Net Income'].iloc[0]
        total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else None
        prev_total_assets = balance_sheet.loc['Total Assets'].iloc[1] if 'Total Assets' in balance_sheet.index else None
        if total_assets is None or prev_total_assets is None:
            print(f"{ticker}: Total Assets data not found.")
            data['Fスコア'] = "N/A"
            return data

        roa = net_income / total_assets if total_assets != 0 else 0
        if roa > 0:
            score += 1

        # CFO
        cfo = cash_flow.loc['Total Cash From Operating Activities'].iloc[0] if 'Total Cash From Operating Activities' in cash_flow.index else None
        if cfo is None:
            print(f"{ticker}: Operating Cash Flow data not found.")
            cfo = 0

        if cfo > 0:
            score += 1

        # ROAの増加
        prev_net_income = income_statement.loc['Net Income'].iloc[1]
        prev_roa = prev_net_income / prev_total_assets if prev_total_assets != 0 else 0
        if roa > prev_roa:
            score += 1

        # CFO > Net Income
        if cfo > net_income:
            score += 1

        # 負債比率の減少
        long_term_debt = balance_sheet.loc['Long Term Debt'].iloc[0] if 'Long Term Debt' in balance_sheet.index else 0
        prev_long_term_debt = balance_sheet.loc['Long Term Debt'].iloc[1] if 'Long Term Debt' in balance_sheet.index else 0
        if long_term_debt < prev_long_term_debt:
            score += 1

        # 流動比率の増加
        current_assets = balance_sheet.loc['Total Current Assets'].iloc[0] if 'Total Current Assets' in balance_sheet.index else None
        current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in balance_sheet.index else None
        prev_current_assets = balance_sheet.loc['Total Current Assets'].iloc[1] if 'Total Current Assets' in balance_sheet.index else None
        prev_current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[1] if 'Total Current Liabilities' in balance_sheet.index else None

        if None in [current_assets, current_liabilities, prev_current_assets, prev_current_liabilities]:
            print(f"{ticker}: Current Assets or Current Liabilities data not found.")
        else:
            current_ratio = current_assets / current_liabilities if current_liabilities != 0 else 0
            prev_current_ratio = prev_current_assets / prev_current_liabilities if prev_current_liabilities != 0 else 0
            if current_ratio > prev_current_ratio:
                score += 1

        # 利益率の増加
        gross_profit = income_statement.loc['Gross Profit'].iloc[0] if 'Gross Profit' in income_statement.index else None
        revenue = income_statement.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_statement.index else None
        prev_gross_profit = income_statement.loc['Gross Profit'].iloc[1] if 'Gross Profit' in income_statement.index else None
        prev_revenue = income_statement.loc['Total Revenue'].iloc[1] if 'Total Revenue' in income_statement.index else None

        if None in [gross_profit, revenue, prev_gross_profit, prev_revenue]:
            print(f"{ticker}: Gross Profit or Total Revenue data not found.")
        else:
            gross_margin = gross_profit / revenue if revenue != 0 else 0
            prev_gross_margin = prev_gross_profit / prev_revenue if prev_revenue != 0 else 0
            if gross_margin > prev_gross_margin:
                score += 1

        # 資産回転率の増加
        if revenue and prev_revenue:
            asset_turnover = revenue / total_assets if total_assets != 0 else 0
            prev_asset_turnover = prev_revenue / prev_total_assets if prev_total_assets != 0 else 0
            if asset_turnover > prev_asset_turnover:
                score += 1

        data['Fスコア'] = score  # 整数値として設定

    except Exception as e:
        print(f"Error calculating Piotroski Score for {ticker}: {e}")
        data['Fスコア'] = "N/A"
    return data

def save_etf_data(ticker, file_path, include_canslim_data=False, include_investment_scores=False):
    etf = yf.Ticker(ticker)
    data = {}
    data['Ticker'] = ticker

    # 現在の年を取得
    current_year = datetime.now().year

    # 利益率と売上高のデータを取得
    profit_margins = calculate_profit_margins(ticker)
    data.update(profit_margins)

    # 過去2年分と今年の最高値・最安値を取得
    high_low_2y_ago = get_yearly_high_low(ticker, current_year - 2)
    high_low_1y_ago = get_yearly_high_low(ticker, current_year - 1)
    high_low_current = get_yearly_high_low(ticker, current_year)

    data['最高値(2期前)'] = high_low_2y_ago['yearly_high']
    data['最安値(2期前)'] = high_low_2y_ago['yearly_low']
    data['最高値(1期前)'] = high_low_1y_ago['yearly_high']
    data['最安値(1期前)'] = high_low_1y_ago['yearly_low']
    data['最高値(今年)'] = high_low_current['yearly_high']
    data['最安値(今年)'] = high_low_current['yearly_low']

    # 他の基本的なデータを取得
    per = format_decimal(etf.info.get("forwardPE", "N/A"))  # 予想PER
    trailing_per = format_decimal(etf.info.get("trailingPE", "N/A"))  # 実績PER
    roe = format_percentage(etf.info.get("returnOnEquity", "N/A"))  # ROE
    website = etf.info.get("website", "N/A")  # 企業URL

    # 現在の株価を取得
    try:
        history = etf.history(period="1d")
        if not history.empty and 'Close' in history.columns:
            current_price = format_decimal(history['Close'].iloc[-1])
        else:
            print(f"{ticker}: No data available for current price.")
            current_price = np.nan
    except Exception as e:
        print(f"{ticker}: Error retrieving current price: {e}")
        current_price = np.nan

    data['予想PER'] = per
    data['実績PER'] = trailing_per
    data['ROE'] = roe
    data['企業URL'] = website
    data['現在の株価'] = current_price

    # ニュースを取得
    news_items = etf.news
    if news_items:
        news_str = ";".join([news_item['title'].replace(",", " ").replace("—", "-").replace("\u2013", " ").replace("\xa0", " ").replace("\xae", " ").replace("\u014c", " ").replace("\u2122", " ").replace('\u02bb', " ") for news_item in news_items])
        data['ニュース'] = news_str
    else:
        data['ニュース'] = "N/A"

    # アナリスト評価を取得
    recommendations_summary = etf.recommendations_summary
    if recommendations_summary is not None:
        recommendations_summary = clean_json(recommendations_summary)
        data['アナリスト評価'] = recommendations_summary
    else:
        data['アナリスト評価'] = "N/A"

    # CAN-SLIMデータを追加する場合
    if include_canslim_data:
        canslim_data = get_canslim_data_dict(ticker)
        data.update(canslim_data)

    # 投資指標データを追加する場合
    if include_investment_scores:
        # GARPスコアを計算
        garp_data = calculate_garp_score(ticker)
        data.update(garp_data)
        # マジックフォーミュラを計算
        magic_formula_data = calculate_magic_formula(ticker)
        data.update(magic_formula_data)
        # ピオトロスキー・スコアを計算
        piotroski_data = calculate_piotroski_score(ticker)
        data.update(piotroski_data)

    # データをDataFrameに変換
    df = pd.DataFrame([data])

    # 列の順序を指定
    columns_order = [
        "Ticker",
        "利益率(2期前)",
        "利益率(1期前)",
        "利益率(今年)",
        "実績PER",
        "予想PER",
        "ROE",
        "売上高(2期前)",
        "売上高(1期前)",
        "売上高(今年)",
        "現在の株価",
        "最高値(2期前)",
        "最安値(2期前)",
        "最高値(1期前)",
        "最安値(1期前)",
        "最高値(今年)",
        "最安値(今年)",
        "ニュース",
        "企業URL",
        "アナリスト評価"
    ]

    if include_canslim_data:
        canslim_columns = [
            '四半期純利益成長率(%)',
            '四半期EPS成長率(%)',
            '年次純利益成長率(%)',
            '年次EPS成長率(%)',
            'CANSLIM現在の株価',
            '52週高値',
            '株価と52週高値の比率(%)',
            '発行済株式数',
            '浮動株数',
            'インサイダー保有率(%)',
            '機関投資家保有率(%)',
            '平均取引量(1年)',
            '株価リターン(1年)(%)',
            'セクター',
            '業界',
            'ベータ',
            '機関投資家の数',
            '機関投資家の合計保有株数',
            '市場リターン(S&P 500 1年)(%)'
        ]
        columns_order.extend(canslim_columns)

    if include_investment_scores:
        investment_columns = [
            'EPS成長率',
            'Investment PER',
            'PEGレシオ',
            'ROIC',
            '利益利回り',
            'Fスコア',
            '更新日'
        ]
        columns_order.extend(investment_columns)

    # 列の順序を適用し、存在しない列は無視
    df = df.reindex(columns=columns_order)

    # CSVファイルが存在しない場合はヘッダーを追加して新規作成
    if os.path.exists(file_path):
        # 既存のCSVを読み込む
        existing_df = pd.read_csv(file_path, encoding='shift_jis', on_bad_lines='skip')
        existing_df.set_index('Ticker', inplace=True)

        # Tickerが既存のデータにある場合は上書き、ない場合は新しい行として追加
        if ticker in existing_df.index:
            existing_df.update(df.set_index('Ticker'))
        else:
            existing_df = pd.concat([existing_df, df.set_index('Ticker')])

        # CSVに書き戻す
        existing_df.reset_index().to_csv(file_path, index=False, encoding='shift_jis', errors='replace')
    else:
        # CSVファイルがない場合は新規作成
        df.to_csv(file_path, index=False, encoding='shift_jis', errors='replace')

    print(f"{file_path} にデータを出力しました。")

# メイン処理
def mains(tickers, file_path = os.path.join(f'./csv/', 'research.csv')):
    for ticker in tickers:
        if utils.read_ticker_csv(ticker):
            print(f"{ticker}のyfinance取得はスキップします。")
            continue

        main(ticker, file_path)

def main(ticker, file_path = os.path.join(f'./csv/', 'research.csv')):
    # CAN-SLIMデータと投資指標データを追加する場合は引数をTrueにする
    save_etf_data(ticker, file_path, include_canslim_data=True, include_investment_scores=True)

# 複数のティッカーに対してループ処理
if __name__ == "__main__":
    mains(tickers)

# https://finviz.com/screener.ashx?v=111&f=cap_midover,fa_pe_u20,fa_peg_low,fa_roe_o15,geo_usa,sh_price_u50,sh_relvol_o0.5&ft=4