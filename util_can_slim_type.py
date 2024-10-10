import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
from scipy.signal import argrelextrema
import numpy as np

def plot_pattern(data, title, image_name, image_folder, points=None, lines=None):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price')

    # ポイントをプロット
    if points is not None:
        for point in points:
            plt.scatter(data.index[point['index']], data['Close'].iloc[point['index']],
                        color=point['color'], label=point['label'])

    # ラインをプロット
    if lines is not None:
        data_min = data['Close'].min()
        for line in lines:
            if 'x' in line:
                # x が指定されている場合、最小値以上の部分のみプロット
                x_filtered = [x for x, y in zip(line['x'], line['y']) if y >= data_min]
                y_filtered = [y for y in line['y'] if y >= data_min]
                if x_filtered and y_filtered:
                    plt.plot(x_filtered, y_filtered, color=line['color'], linestyle=line['linestyle'], label=line['label'])
            else:
                # 水平線の場合、dataの最小値より大きい場合のみ plt.axhline を使用
                if line['y'] > data_min:
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

# OK!!
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
    purchase_price = 0
    lines = []
    if len(max_idx) >= 2 and len(min_idx) >= 1:
        # 最新２つのピークを選択
        right_peak = max_idx[-1]
        left_peak = max_idx[-2]

        # カップの底を選択
        cup_bottom_candidates = min_idx[(min_idx > left_peak) & (min_idx < right_peak)]
        if len(cup_bottom_candidates) == 0:
            print("カップの底が見つかりません。")
            # return False, None, None, None, None
        cup_bottom = cup_bottom_candidates[np.argmin(data['Close'].values[cup_bottom_candidates])]

        points = [
            {'index': left_peak, 'label': 'Left Peak', 'color': 'g'},
            {'index': cup_bottom, 'label': 'Cup Bottom', 'color': 'r'},
            {'index': right_peak, 'label': 'Right Peak', 'color': 'g'}
        ]
        # 取っ手部分の開始点を設定（カップの右ピークから軽い調整部分）
        handle_start_candidates = min_idx[min_idx > right_peak]
        if len(handle_start_candidates) == 0:
            print("取っ手部分の開始点が見つかりません。")
            # return False, None, None, None, None
        else:
            handle_start = handle_start_candidates[0]
            handle_end = len(data) - 1  # データの最終インデックスを終了点とする

            # 取っ手部分のデータを取得
            handle_data = data.iloc[handle_start:handle_end + 1]

            if len(handle_data) < 2:
                print("取っ手部分が1週間未満です。パターンを無効とします。")
                # return False, None, None, None, None

            handle_min_price = handle_data['Close'].min()
            if handle_min_price >= data['Close'].values[right_peak]:
                print("取っ手部分の下降が確認できません。パターンを無効とします。")
                # return False, None, None, None, None

            # 最新カップの右高値からその後の最小値の下降が12%の範囲にあるかを確認
            right_peak_price = data['Close'].values[right_peak]
            handle_depth = (right_peak_price - handle_min_price) / right_peak_price * 100
            if handle_depth > 12:
                print(f"取っ手部分の下降が12%の範囲外です: {handle_depth:.2f}%")
                # return False, None, None, None, None

            # 取っ手部分の傾きが急でないか（緩やかであることを確認）
            handle_slope = (handle_data['Close'].iloc[-1] - handle_data['Close'].iloc[0]) / len(handle_data)
            if handle_slope < 0 or handle_slope > abs(data['Close'].values[right_peak] - handle_min_price) / len(handle_data):
                print("取っ手部分の傾きが緩やかではありません。パターンを無効とします。")
                return False, None, None, None, None

            # 購入価格はカップの右側のピーク（取っ手の上限）に設定
            purchase_price = data['Close'].values[right_peak]
            lines = [
                {'y': purchase_price, 'label': 'Purchase Price', 'color': 'purple', 'linestyle': '--'}
            ]
            # 取っ手部分のラインを追加
            lines.append({
                'x': [data.index[handle_start], data.index[handle_end]],
                'y': [data['Close'].iloc[handle_start], data['Close'].iloc[handle_end]],
                'label': 'Handle',
                'color': 'cyan',
                'linestyle': '-'  # 実線で取っ手部分を表示
            })

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

def detect_saucer_with_handle(data, window=5, image_folder=None, depth_check=True):
    # 高値の局所的な極大を検出（ソーサーの両側）
    max_idx = argrelextrema(data['Close'].values, np.greater, order=window)[0]

    # 安値の局所的な極小を検出（ソーサーの底、max_idxの間に限定）
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
                # 深さが5〜15%の範囲にある場合のみ追加（depth_checkがTrueの場合）
                if not depth_check or ((5 <= depth_left <= 15) and (5 <= depth_right <= 15)):
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
    purchase_price = 0
    lines = []
    if len(max_idx) >= 2 and len(min_idx) >= 1:
        # 最新２つのピークを選択
        right_peak = max_idx[-1]
        left_peak = max_idx[-2]

        # ソーサーの底を選択
        saucer_bottom_candidates = min_idx[(min_idx > left_peak) & (min_idx < right_peak)]
        if len(saucer_bottom_candidates) == 0:
            print("ソーサーの底が見つかりません。")
            return False, None, None, None, None
        saucer_bottom = saucer_bottom_candidates[np.argmin(data['Close'].values[saucer_bottom_candidates])]

        points = [
            {'index': left_peak, 'label': 'Left Peak', 'color': 'g'},
            {'index': saucer_bottom, 'label': 'Saucer Bottom', 'color': 'r'},
            {'index': right_peak, 'label': 'Right Peak', 'color': 'g'}
        ]
        
        # 取っ手部分の開始点を設定（ソーサーの右ピークから軽い調整部分）
        handle_start_candidates = min_idx[min_idx > right_peak]
        if len(handle_start_candidates) == 0:
            print("取っ手部分の開始点が見つかりません。")
            return False, None, None, None, None
        
        handle_start = handle_start_candidates[0]
        handle_end = len(data) - 1  # 初期の期間をデータの最終インデックスに設定
        while handle_start < handle_end:
            # 取っ手部分のデータを取得
            handle_data = data.iloc[handle_start:handle_end + 1]

            if len(handle_data) < 2:
                print("取っ手部分が1週間未満です。パターンを無効とします。")
                return False, None, None, None, None

            handle_min_price = handle_data['Close'].min()
            if handle_min_price >= data['Close'].values[right_peak]:
                print("取っ手部分の下降が確認できません。パターンを無効とします。")
                handle_start += 5
                continue

            # 最新ソーサーの右高値からその後の最小値の下降が5%の範囲にあるかを確認
            right_peak_price = data['Close'].values[right_peak]
            handle_depth = (right_peak_price - handle_min_price) / right_peak_price * 100
            if handle_depth > 5:
                print(f"取っ手部分の下降が5%の範囲外です: {handle_depth:.2f}%")
                handle_start += 5
                continue

            # 取っ手部分の傾きが急でないか（緩やかであることを確認）
            handle_slope = (handle_data['Close'].iloc[-1] - handle_data['Close'].iloc[0]) / len(handle_data)
            if handle_slope < 0 or handle_slope > abs(data['Close'].values[right_peak] - handle_min_price) / len(handle_data):
                print("取っ手部分の傾きが緩やかではありません。パターンを無効とします。")
                handle_start += 5
                continue

            # 条件に合致した場合は終了
            break

        # 購入価格はソーサーの右側のピーク（取っ手の上限）に設定
        purchase_price = data['Close'].values[right_peak]
        lines = [
            {'y': purchase_price, 'label': 'Purchase Price', 'color': 'purple', 'linestyle': '--'}
        ]
        # 取っ手部分のラインを追加
        lines.append({
            'x': [data.index[handle_start], data.index[handle_end]],
            'y': [data['Close'].iloc[handle_start], data['Close'].iloc[handle_end]],
            'label': 'Handle',
            'color': 'cyan',
            'linestyle': '-'
        })

        # max_idx の他の局所的極大値も追加（色を区別）
        for idx in max_idx:
            if idx != left_peak and idx != right_peak:
                points.append({'index': idx, 'label': 'Local Max', 'color': 'b'})

        # min_idx の他の局所的極小値も追加（色を区別）
        for idx in min_idx:
            if idx != saucer_bottom:
                points.append({'index': idx, 'label': 'Local Min', 'color': 'orange'})

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

# OK!!
def detect_double_bottom(data, window=2, image_folder=None):
    # 局所的な極小値（ボトム）を検出
    min_idx = argrelextrema(data['Close'].values, np.less, order=window)[0]

    if len(min_idx) >= 2:
        first_bottom = min_idx[-2]
        second_bottom = min_idx[-1]

        # 2つのボトムが近い価格帯であることを確認（差が10%以内）
        if abs(data['Close'].iloc[first_bottom] - data['Close'].iloc[second_bottom]) / data['Close'].iloc[first_bottom] > 0.1:
            print("2つのボトムの価格差が大きいため、ダブルボトムとは認められません。")
            return False, None, None, None

        # ネックライン（2つのボトム間の高値）を取得
        neckline = data['Close'][first_bottom:second_bottom+1].max()

        # ネックラインがボトムから十分な差があるか確認（最低15%の差）
        if (neckline - min(data['Close'].iloc[first_bottom], data['Close'].iloc[second_bottom])) / min(data['Close'].iloc[first_bottom], data['Close'].iloc[second_bottom]) < 0.15:
            print("ネックラインがボトムから十分に離れていないため、ダブルボトムとは認められません。")
            return False, None, None, None
        purchase_price = neckline * 1.02  # ネックラインの2%上

        # プロット用のポイントとラインを準備
        points = [
            {'index': first_bottom, 'label': 'First Bottom', 'color': 'r'},
            {'index': second_bottom, 'label': 'Second Bottom', 'color': 'r'}
        ]
        lines = [
            {'y': neckline, 'label': 'Neckline', 'color': 'orange', 'linestyle': '--'},
            {'y': purchase_price, 'label': 'Purchase Price(Neckline x 1.02)', 'color': 'purple', 'linestyle': '--'}
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


def detect_flat_base(data, period=7, tolerance=0.05, image_folder=None):
    recent_data = data[-period:]
    max_price = recent_data['Close'].max()
    min_price = recent_data['Close'].min()

    # 変動幅が許容範囲内か確認
    if (max_price - min_price) / max_price < tolerance:
        purchase_price = max_price * 1.03  # ベースの高値の3%上

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
        print('範囲外です')
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

# OK!window_sizesは修正する可能性あり。
def detect_vcp(data, window_sizes=[24, 16, 8, 2], image_folder=None):
    volatilities = []
    for window in window_sizes:
        recent_data = data[-window:]
        volatility = recent_data['Close'].std() / recent_data['Close'].mean()
        volatilities.append(volatility)
        # print(f"Window: {window}, Volatility: {volatility:.4f}")

    # ボラティリティが縮小しているか確認
    if all(round(x, 4) >= round(y, 4) for x, y in zip(volatilities, volatilities[1:])):
        print("ボラティリティが縮小しています。VCPの条件を満たしています。")
        recent_high = data['Close'][-10:].max()
        purchase_price = recent_high * 1.02  # 高値の2%上
        print(f"Recent High: {recent_high:.2f}, Purchase Price: {purchase_price:.2f}")

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
        print("ボラティリティが縮小していません。VCPの条件を満たしていません。")
        return False, None

def calculate_buy_price(data, right_peak, buffer_percent=1.02):
    # 右ピークの価格を取得し、購入価格を計算（例: 右ピークの価格を2%上回る）
    right_peak_price = data['Close'][right_peak]
    buy_price = right_peak_price * buffer_percent
    return buy_price

def is_trend(index_data, ma_period=25, threshold=0.03, trend_type='downtrend'):
    # 移動平均を計算
    moving_average = index_data['Close'].rolling(window=ma_period).mean()
    # 最新の価格と移動平均を取得
    latest_price = index_data['Close'].iloc[-1]
    ma_value = moving_average.iloc[-1]
    # 移動平均線の傾きを計算（過去5期間）
    ma_slope = moving_average.diff().iloc[-5:]
    
    if trend_type == 'downtrend':
        # 下降トレンドの確認
        return latest_price < ma_value * (1 - threshold) and (ma_slope < 0).all()
    elif trend_type == 'uptrend':
        # 上昇トレンドの確認
        return latest_price > ma_value * (1 + threshold) and (ma_slope > 0).all()
    else:
        raise ValueError("trend_type must be either 'downtrend' or 'uptrend'")

def detect_market_downtrend(data, dow_data, sp500_data, ma_period=25, threshold=0.03, image_folder=None):
    # ダウ平均とS&P500の下降トレンドを確認
    dow_downtrend = is_trend(dow_data, ma_period, threshold)
    sp500_downtrend = is_trend(sp500_data, ma_period, threshold)

    # どちらかが下降トレンドに入っていれば売りシグナルを出す
    if dow_downtrend or sp500_downtrend:
        # プロットを作成
        plot_pattern(data=data, title='Market Downtrend', image_name='market_downtrend.png', image_folder=image_folder)
        return True, data['Close'].iloc[-1]
    else:
        print(f"市場全体は下降トレンドではありませんので、売りは保留です。")
        return False, None

def detect_moving_average_break(data, ma_period=50, threshold=0.05, image_folder=None):
    moving_average = data['Close'].rolling(window=ma_period).mean()
    latest_price = data['Close'].iloc[-1]
    ma_value = moving_average.iloc[-1]
    previous_price = data['Close'].iloc[-2]
    previous_ma_value = moving_average.iloc[-2]

    # 元々移動平均線より上にあり、その後急激に5%以上下回ったかを確認
    if previous_price > previous_ma_value and latest_price < ma_value * (1 - threshold):
        plot_pattern(data=data, title='Moving Average Break', image_name='moving_average_break.png', image_folder=image_folder)
        return True, latest_price
    else:
        print(f"移動平均線を下回っていませんので、売りは保留です。")
        return False, None
# OK!
def detect_climax_top(data, window=4, threshold=0.3, image_folder=None):
    latest_high = data['Close'].iloc[-1]  # 最新の高値
    latest_low = data['Close'].iloc[-2]  # 最新の1個前の安値
    recent_volume = data['Volume'].iloc[-1]  # 最新の出来高
    max_volume = data['Volume'].max()  # 全範囲の最大出来高
    
    # プロット用のポイントとラインを準備
    points = [
        {'index': data.index.get_loc(data['Close'][-window:].idxmin()), 'label': 'Latest Low', 'color': 'blue'},
        {'index': data.index.get_loc(data['Close'][-window:].idxmax()), 'label': 'Latest High', 'color': 'red'},
        {'index': data.index.get_loc(data['Volume'].idxmax()), 'label': 'Max Volume', 'color': 'orange'}
    ]
    lines = [
        {'y': latest_high, 'label': 'Latest High', 'color': 'red', 'linestyle': '--'},
        {'y': latest_low, 'label': 'Latest Low', 'color': 'blue', 'linestyle': '--'}
    ]
    
    # 最近で上昇を続けていることを確認
    is_recently_rising = data['Close'].iloc[-window:].is_monotonic_increasing
    
    # 30%以上の急上昇を確認し、最大の陽線（出来高の急増）を確認
    if is_recently_rising and (latest_high - latest_low) / latest_low > threshold and recent_volume >= max_volume * 0.9:  # 出来高が最大出来高
        sell_price = latest_high * 0.95  # 高値からの5%下で売り
        plot_pattern(data=data, points=points, lines=lines, title='Climax Top', image_name='climax_top.png', image_folder=image_folder)
        return True, sell_price
    else:
        print(f"クライマックストップ：最大の陽線と出来高は確認できませんでした。")
        return False, None

def detect_exhaustion_gap(data, image_folder=None):
    if len(data) < 2:
        return False, None

    gap_up = data['Low'].iloc[-2] < data['High'].iloc[-1]  # ギャップアップを確認
    high_formed = data['Close'].iloc[-1] < data['High'].iloc[-1]  # 高値更新後に失速
    
    if gap_up and high_formed:
        sell_price = data['Close'].iloc[-1]
        plot_pattern(data=data, title='Exhaustion Gap', image_name='exhaustion_gap.png', image_folder=image_folder)
        return True, sell_price
    else:
        return False, None

# JNJは2, 
def detect_upper_channel_line(data, window=2, channel_multiplier=1.05, image_folder=None):
    # 最高値のインデックスを取得
    high_indices = argrelextrema(data['Close'].values, np.greater, order=window)[0]
    high_values = data['Close'].iloc[high_indices]
    
    def find_high_points(high_indices, high_values, latest_high_indices, latest_high_values):
        # ベースケース: 3つの高値が見つかった場合
        if len(latest_high_indices) == 3:
            return latest_high_indices, latest_high_values

        # 最初の点はここ1ヶ月の高めな点
        if len(latest_high_indices) == 0:
            latest_high_indices = [high_indices[-1]]
            latest_high_values = [high_values.iloc[-1]]
        
        # 次の点を探す
        for i in range(len(high_indices) - len(latest_high_indices) - 1, -1, -1):
            potential_index = high_indices[i]
            potential_value = high_values.iloc[i]

            # 既存の最新の点と新しい点を結んだとき、傾きが負ならスキップ
            slope, intercept, _, _, _ = linregress([potential_index, latest_high_indices[0]], [potential_value, latest_high_values[0]])
            if slope <= 0:
                continue

            # 線上により高い値があるか確認
            if all(data['Close'].iloc[potential_index + 1:latest_high_indices[0]] <= (slope * np.arange(potential_index + 1, latest_high_indices[0]) + intercept)):
                # 新しい点を追加
                latest_high_indices.insert(0, potential_index)
                latest_high_values.insert(0, potential_value)
                return find_high_points(high_indices, high_values, latest_high_indices, latest_high_values)
        
        # 再帰の深さを制限するため、再帰を中断して終了する
        return latest_high_indices, latest_high_values

    # 最高値が3つ以上ある場合に再帰的に3つの点を探す
    latest_high_indices, latest_high_values = find_high_points(high_indices, high_values, [], [])
    
    # 傾きが負でないか確認
    if len(latest_high_indices) >= 2:
        # 右から1個目と2個目の点を結ぶ直線を計算
        slope, intercept, _, _, _ = linregress(latest_high_indices[-2:], latest_high_values[-2:])
        if slope <= 0:
            print(f"上方チャネルライン：下降向きです")
            return False, None
    else:
        print(f"上方チャネルラインの点が不足しています")
        return False, None
    
    # 数値インデックスを使用して上方チャネルラインを計算
    numeric_indices = np.arange(len(data))
    upper_channel_line = slope * numeric_indices + intercept

    # 最新の価格と上方チャネルラインの価格を比較
    latest_price = data['Close'].iloc[-1]
    upper_line_price = upper_channel_line[-1]

    # プロット用のライン
    lines = [
        {'y': upper_line_price, 'label': 'Upper Channel Line', 'color': 'red', 'linestyle': '--'},
        {'x': data.index, 'y': upper_channel_line, 'label': 'Upper Trend Line', 'color': 'orange', 'linestyle': '-'}
    ]
    # プロット用のポイント（最新の3つの高値）
    points = [
        {'index': idx, 'label': 'High Point', 'color': 'green'} for idx in latest_high_indices
    ]
    plot_pattern(data=data, lines=lines, points=points, title='Upper Channel Line', image_name='upper_channel_line.png', image_folder=image_folder)

    if latest_price >= upper_line_price:
        return True, latest_price
    else:
        return False, latest_price

def detect_double_top(data, window=10, image_folder=None):
    """
    ダブルトップを使用して売りシグナルを検出する関数。
    
    Parameters:
    - data: DataFrame（週足の株価データ）
    - window: int（ピーク間の期間）
    - image_folder: str（プロットを保存するフォルダパス）
    
    Returns:
    - sell_signal: bool（売りシグナルの有無）
    - sell_price: float（売り価格）
    """
    max_idx = argrelextrema(data['Close'].values, np.greater, order=window)[0]

    if len(max_idx) >= 2:
        first_peak = max_idx[-2]
        second_peak = max_idx[-1]
        
        if abs(data['Close'].iloc[first_peak] - data['Close'].iloc[second_peak]) / data['Close'].iloc[first_peak] < 0.05:
            neckline = data['Close'][first_peak:second_peak+1].min()
            sell_price = neckline * 0.98  # ネックラインを割り込むことで売り
            plot_pattern(data=data, title='Double Top', image_name='double_top.png', image_folder=image_folder)
            return True, sell_price
    return False, None

def detect_railroad_tracks(data, image_folder=None):
    """
    レールロードトラックを使用して売りシグナルを検出する関数。
    
    Parameters:
    - data: DataFrame（週足の株価データ）
    - image_folder: str（プロットを保存するフォルダパス）
    
    Returns:
    - sell_signal: bool（売りシグナルの有無）
    - sell_price: float（売り価格）
    """
    if len(data) < 2:
        return False, None
    
    prev_candle = data.iloc[-2]
    latest_candle = data.iloc[-1]

    # 前日が陽線、当日が陰線でかつ同程度の長さか確認
    if (prev_candle['Close'] > prev_candle['Open'] and
        latest_candle['Close'] < latest_candle['Open'] and
        abs(prev_candle['Close'] - prev_candle['Open']) * 0.9 < abs(latest_candle['Close'] - latest_candle['Open']) < abs(prev_candle['Close'] - prev_candle['Open']) * 1.1):
        sell_price = latest_candle['Close']
        plot_pattern(data=data, title='Railroad Tracks', image_name='railroad_tracks.png', image_folder=image_folder)
        return True, sell_price
    else:
        return False, None
   
# 20〜25%の利益確定ルール: 一定の利益が得られたら部分的に売却。
# 7〜8%の損切りルール: 早期の損失確定で大きな損失を避ける。
# クライマックス・ランでの売却: 短期間の急激な上昇後に売却。
# トレンドラインの下抜け: 長期トレンドの終わりの兆し。
# 上方チャネルラインの到達: 過熱感を示し、反転のリスクが高い。
# 逆カップウィズハンドル型: 下落トレンドへの反転を示すパターン。
# 出来高減少での新高値形成失敗: 買い手の勢いの減少を示す。
# コンソリデーションブレイク: 安定が崩れて下方に動いた場合。
# 急上昇後の調整時に一部売却: 調整を見越して部分的に利益を確定。
# def detect_upper_channel_line(data, date = 360):

#     # 高値と日付を抽出
#     highs = data['High']
#     dates = pd.to_datetime(data['Date'])
    
#     # 最高値の抽出（過去数ヶ月分、最低3つの高値を検出）
#     high_points = []
#     num_highs_needed = 3
#     for i in range(1, len(highs) - 1):
#         # 高値の条件: 前後の日よりも高い
#         if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
#             high_points.append((dates[i], highs[i]))
        
#         if len(high_points) >= num_highs_needed:
#             break
    
#     # 高値が3つ以上あるかチェック
#     if len(high_points) < num_highs_needed:
#         print(f"高値が十分にありません")
#         return None, None
    
#     # 高値の座標を取得
#     high_dates = [point[0].toordinal() for point in high_points]  # 日付を数値化
#     high_values = [point[1] for point in high_points]  # 高値の価格
    
#     # 高値の直線回帰を行い、チャネルラインを作成
#     slope, intercept, _, _, _ = linregress(high_dates, high_values)
    
#     # 直線（上方チャネルライン）の式: y = slope * x + intercept
#     upper_channel_line = slope * dates.apply(lambda date: date.toordinal()) + intercept
    
#     # 現在の価格
#     current_price = data['Close'].iloc[-1]
    
#     # 売りサイン: 現在の価格が上方チャネルラインを超えた場合
#     sell_signal = current_price > upper_channel_line.iloc[-1]
    
#     # 次の売り価格を予測 (例えば、1週間後の価格を予測)
#     future_date = dates.iloc[-1] + pd.Timedelta(days=7)
#     future_date_ordinal = future_date.toordinal()
#     future_sell_price = slope * future_date_ordinal + intercept

#     # 結果を出力
#     print(f'現在価格: {current_price}')
#     print(f'売りサイン: {sell_signal} (上方チャネルライン: {upper_channel_line.iloc[-1]})')
#     print(f'次の売り価格（1週間後予測）: {future_sell_price}（予測日: {future_date.strftime("%Y-%m-%d")}）')
    
#     # グラフの描画（オプション）
#     plt.figure(figsize=(10, 6))
#     plt.plot(dates, highs, label="高値", color="blue")
#     plt.plot(dates, upper_channel_line, label="上方チャネルライン", color="red")
#     plt.scatter([point[0] for point in high_points], [point[1] for point in high_points], color="green", label="高値のポイント")
#     plt.axhline(y=future_sell_price, color='orange', linestyle='--', label="次の売り価格予測")
#     plt.title(f'高値と上方チャネルライン')
#     plt.xlabel("日付")
#     plt.ylabel("価格")
#     plt.legend()
#     plt.show()
    
#     return upper_channel_line.iloc[-1], sell_signal