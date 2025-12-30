import talib
import numpy as np

def get_strategy_feature_filtered_feature_and_labels(df, strategy_feature_name):
    if strategy_feature_name == 'low_bb_du':
        return low_bb_du(df)
    return None

def label_df(df, label_name, upper, lower):
    labels = np.full(len(df), np.nan)

    buy_indices = df.index[df['strategy_feature'] == True]

    for idx in buy_indices:
        buy_close = df.at[idx, 'close']

        future = df.loc[idx+1:]

        # 손절 먼저 체크
        stop_loss = future[future['low'] < buy_close * lower]
        take_profit = future[future['high'] > buy_close * upper]

        if not stop_loss.empty and not take_profit.empty:
            # 둘 다 발생하면 더 먼저 발생한 것
            if stop_loss.index[0] <= take_profit.index[0]:
                labels[idx] = 0
            else:
                labels[idx] = 1

        elif not stop_loss.empty:
            labels[idx] = 0

        elif not take_profit.empty:
            labels[idx] = 1
        # else:
        #     # 끝까지 갔을 때
        #     last_close = future.iloc[-1]['close']
        #     labels[idx] = 1 if last_close > buy_close else 0

    df[label_name] = labels
    return df

def low_bb_du(df):
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['strategy_feature'] = (df['close'] > df['bb_lower']) & (df['open'] < df['bb_lower'])

    df['stoch_k'], df['stoch_d'] = talib.STOCH(
        df['high'],
        df['low'],
        df['close'],
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0
    )

    df['f1'] = (df['close'] / df['open'])
    df['f2'] = df['low'] / df['open']
    df['f3'] = df['high'] / df['open']

    # gc 제대로.
    df['f4'] = (df['stoch_d'] < df['stoch_k'])
    df['f5'] = (df['stoch_d'].shift(1) > df['stoch_k'].shift(1))
    df['f6'] = (df['stoch_d'].shift(2) > df['stoch_k'].shift(2))
    df['f7'] = df['stoch_k']
    df['f8'] = (df['close'] / df['bb_lower']) 
    df['f9'] = df['close'] / (df['close'].shift(1).rolling(5).min())
    
    bb_range = df["bb_upper"] - df["bb_lower"]
    df['f10'] = ((df['close'] - df['bb_lower']) / (bb_range))
    
    df['f11'] = False
    for i in range(1, 6):
        df['f11'] |= df['close'].shift(i) < df['bb_lower'].shift(i)
    
    df['f12'] = df['bb_mid'] / df['bb_mid'].shift(1)
    df['f13'] = talib.RSI(df['close'], timeperiod=14)

    sma40 = talib.SMA(df['close'], timeperiod=40)
    df['f14'] = sma40 / sma40.shift(1)
    sma60 = talib.SMA(df['close'], timeperiod=60)
    df['f15'] = sma60 / sma60.shift(1)
    df['f16'] = df['bb_mid'] / sma40
    df['f17'] = sma40 / sma60

    window = 20
    donchain_min = (df["close"].shift(1).rolling(window).min())
    donchain_max = (df["close"].shift(1).rolling(window).max())
    donchain_min_w1 = donchain_min.shift(window)
    donchain_max_w1 = donchain_max.shift(window)
    donchain_min_w2 = donchain_min.shift(window*2)
    donchain_max_w2 = donchain_max.shift(window*2)

    df['f18'] = donchain_min_w1 / donchain_min_w2
    df['f19'] = donchain_min / donchain_min_w1
    df['f20'] = donchain_max_w1 / donchain_max_w2
    df['f21'] = donchain_max / donchain_max_w1
    
    bb_range_std = (bb_range.shift(1).rolling(window).std())
    df['f22'] = bb_range_std / bb_range.std()

    df = label_df(df, 'label1', 1.085, 0.925)
    df = label_df(df, 'label2', 1.12, 0.9)
    df = label_df(df, 'label3', 1.15, 0.875)
    df = label_df(df, 'label4', 1.18, 0.85)

    feat_cols = [f"f{i}" for i in range(1, 23)]
    label_cols = ["label1", "label2", "label3", "label4"]
    need_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']

    df = df[df['strategy_feature'] == True]
    df = (
        df[feat_cols + label_cols + need_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df
    