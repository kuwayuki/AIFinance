import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from scipy.stats import linregress
from scipy.signal import argrelextrema
import numpy as np

def plot_pattern(data, points, lines, title, image_name, image_folder):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price')

    # ポイントをプロット
    for point in points:
        plt.scatter(data.index[point['index']], data['Close'].iloc[point['index']],
                    color=point['color'], label=point['label'])

   # ラインをプロット
    for line in lines:
        if 'x' in line:
            # x が指定されている場合は plt.plot を使用
            plt.plot(line['x'], line['y'], color=line['color'], linestyle=line['linestyle'], label=line['label'])
        else:
            # 水平線の場合は plt.axhline を使用
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

def detect_cup_with_handle(data, window=2, image_folder=None, depth_check=True):
    # 高値の局所的な極大を検出（カップの両側）
    max_idx = argrelextrema(data['Close'].values, np.greater, order=window)[0]

    # 安値の局所的な極小を検出（カップの底、max_idxの間に限定）
    min_idx = []
    new_max_idx = []
    for i in range(0, len(max_idx) - 1, 1):
        if i + 1 < len(max_idx):
            local_min_idx = argrelextrema(data['Close'].values[max_idx[i]:max_idx[i + 1]], np.less, order=window)[0]
            if len(local_min_idx) > 0:
                local_min_idx += max_idx[i]
                min_price = data['Close'].values[local_min_idx[np.argmin(data['Close'].values[local_min_idx])]]
                left_peak_price = data['Close'].values[max_idx[i]]
                right_peak_price = data['Close'].values[max_idx[i + 1]]
                depth_left = (left_peak_price - min_price) / left_peak_price * 100
                depth_right = (right_peak_price - min_price) / right_peak_price * 100
                # 深さが12〜35%の範囲にある場合のみ追加（depth_checkがTrueの場合）
                if not depth_check or ((12 <= depth_left <= 35) and (12 <= depth_right <= 35)):
                    min_idx.extend(local_min_idx)
                    new_max_idx.append(max_idx[i])
                    new_max_idx.append(max_idx[i + 1])

    # 最後の max_idx 以降にある局所的な極小を追加
    if len(max_idx) > 0:
        last_max_idx = max_idx[-1]
        local_min_idx_after_last_max = argrelextrema(data['Close'].values[last_max_idx:], np.less, order=window)[0]
        if len(local_min_idx_after_last_max) > 0:
            local_min_idx_after_last_max += last_max_idx
            min_idx.extend(local_min_idx_after_last_max)

    max_idx = np.array(new_max_idx)
    min_idx = np.array(min_idx)

    if len(max_idx) >= 2 and len(min_idx) >= 1:
        # 最新２つのピークを選択
        right_peak = max_idx[-1]
        left_peak = max_idx[-2]
        right_bottom_peak = min_idx[-1]

        # カップの底を選択
        cup_bottom_candidates = min_idx[(min_idx > left_peak) & (min_idx < right_peak)]
        if len(cup_bottom_candidates) == 0:
            print("カップの底が見つかりません。")
            return False, None, None, None, None
        cup_bottom = cup_bottom_candidates[np.argmin(data['Close'].values[cup_bottom_candidates])]

        # 取っ手部分の開始点が見つからない場合、カップの底を開始点とする
        handle_start = right_bottom_peak
        handle_end = data.index.get_loc(data.index[-1])

        # 取っ手部分のデータを取得
        handle_data = data.iloc[handle_start:handle_end + 1]

        # 取っ手部分の長さが1週間以上であることを確認
        print(len(handle_data))
        if len(handle_data) < 2:  # 1週間
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

        # 取っ手部分のラインを追加
        lines.append({
            'x': [data.index[handle_start], data.index[handle_end]],
            'y': [data['Close'].iloc[handle_start], data['Close'].iloc[handle_end]],
            'label': 'Handle',
            'color': 'cyan',
            'linestyle': '-'  # 実線で取っ手部分を表示
        })

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


def calculate_buy_price(data, right_peak, buffer_percent=1.02):
    # 右ピークの価格を取得し、購入価格を計算（例: 右ピークの価格を2%上回る）
    right_peak_price = data['Close'][right_peak]
    buy_price = right_peak_price * buffer_percent
    return buy_price


def detect_upper_channel_line(data, date = 360):

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