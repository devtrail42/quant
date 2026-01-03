import numpy as np
import pandas as pd
import talib

def get_vbt_indicators(df: pd.DataFrame, ema_period: int = 15, rsi_period: int = 8):
    """지표(EMA, RSI)만 미리 계산하여 캐싱하기 위한 함수"""
    indicators = {}
    indicators['ema'] = talib.EMA(df['close'], timeperiod=ema_period).values
    indicators['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period).values
    return indicators

def numpy_rolling_mean(a, window):
    """Numpy 기반 고속 이동 평균"""
    ret = np.cumsum(a, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return np.concatenate(([np.nan]*(window-1), ret[window-1:] / window))

def numpy_rolling_std(a, window):
    """Numpy 기반 고속 이동 표준편차"""
    # std = sqrt(mean(x^2) - mean(x)^2)
    a2 = a**2
    m1 = numpy_rolling_mean(a, window)
    m2 = numpy_rolling_mean(a2, window)
    return np.sqrt(np.maximum(0, m2 - m1**2))

def vbt_with_filters(
    df: pd.DataFrame,
    window: int = 5,
    k_long: float = 0.5,
    k_short: float = 0.5,
    ema_period: int = 15,
    rsi_period: int = 8,
    rsi_upper: float = 70,
    rsi_lower: float = 30,
    use_std: bool = False,
    std_mult: float = 1.0,
    volume_window: int = 20, # [추가] 거래량 평균 윈도우
    volume_mult: float = 1.0, # [추가] 평균 거래량 대비 배수
    volatility_window: int = 20, # [추가] 변동성(ATR 등) 윈도우
    volatility_threshold: float = 0.0, # [추가] 최소 변동성 수준
    cached_indicators: dict = None,
    cached_ranges: dict = None
):
    """
    변동성 돌파(VBT) 전략 + 필터 (상세 버전)
    """
    open_val = df['open'].values
    close_val = df['close'].values
    high_val = df['high'].values
    low_val = df['low'].values
    vol_val = df['volume'].values
    
    # 1. 가격 돌파 타겟 및 EMA/RSI 지표
    if cached_ranges and window in cached_ranges:
        avg_range = cached_ranges[window]['avg']
        std_range = cached_ranges[window].get('std')
    else:
        range_val = (high_val - low_val)
        shifted_range = pd.Series(range_val).shift(1)
        avg_range = shifted_range.rolling(window=window).mean().values
        std_range = shifted_range.rolling(window=window).std().values if use_std else None
    
    if use_std and std_range is not None:
        target_long = open_val + (avg_range + std_range * std_mult) * k_long
        target_short = open_val - (avg_range + std_range * std_mult) * k_short
    else:
        target_long = open_val + avg_range * k_long
        target_short = open_val - avg_range * k_short
        
    if cached_indicators and 'ema' in cached_indicators:
        ema_val = cached_indicators['ema']
        rsi_val = cached_indicators['rsi']
    else:
        ema_val = talib.EMA(df['close'], timeperiod=ema_period).values
        rsi_val = talib.RSI(df['close'], timeperiod=rsi_period).values

    # 2. 거래량 및 변동성 필터 (상세화)
    # 거래량 필터: 현재 거래량이 과거 n봉 평균보다 높은지
    avg_volume = pd.Series(vol_val).shift(1).rolling(window=volume_window).mean().values
    volume_filter = vol_val > (avg_volume * volume_mult)
    
    # 변동성 필터: 최근 변동성(ATR 대용으로 range 활용)이 일정 수준 이상인지
    curr_range = high_val - low_val
    avg_volatility = pd.Series(curr_range).shift(1).rolling(window=volatility_window).mean().values
    volatility_filter = avg_volatility > (open_val * volatility_threshold)

    # 3. 신호 결합
    signal_long = (high_val >= target_long) & (close_val > ema_val) & volume_filter & volatility_filter
    signal_short = (low_val <= target_short) & (close_val < ema_val) & volume_filter & volatility_filter
    
    reverse_to_short = (rsi_val >= rsi_upper)
    reverse_to_long = (rsi_val <= rsi_lower)
    
    vbt_direction = np.zeros(len(df), dtype=int)
    vbt_direction[signal_long] = 1
    vbt_direction[signal_short] = -1
    
    return {
        'vbt_direction': vbt_direction,
        'reverse_to_short': reverse_to_short,
        'reverse_to_long': reverse_to_long,
        'target_long': target_long,
        'target_short': target_short
    }

# 전략 레지스트리 (고도화 백테스터에서 참조)
VBT_STRATEGY_REGISTRY = {
    "vbt_with_filters": vbt_with_filters
}

def get_vbt_strategy_params_list(strategy_name, config):
    import itertools
    import inspect
    
    if strategy_name not in VBT_STRATEGY_REGISTRY:
        raise KeyError(f"{strategy_name} not in VBT_STRATEGY_REGISTRY")
        
    func = VBT_STRATEGY_REGISTRY[strategy_name]
    sig = inspect.signature(func)
    arg_names = [name for name in sig.parameters.keys() if name != "df"]

    param_values = []
    for name in arg_names:
        list_key = name + '_list'
        if list_key not in config:
             if name in config:
                 param_values.append([config[name]])
             else:
                 # Default if optional
                 default = sig.parameters[name].default
                 if default is not inspect.Parameter.empty:
                     param_values.append([default])
                 else:
                     raise KeyError(f"Config missing {list_key} or {name}")
        else:
            param_values.append(config[list_key])
            
    params_list = []
    for combo in itertools.product(*param_values):
        params = dict(zip(arg_names, combo))
        params_list.append(params)
        
    return params_list
