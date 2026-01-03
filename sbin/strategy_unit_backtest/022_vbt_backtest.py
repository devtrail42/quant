import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import argparse
import json
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import itertools
import inspect

# tqdm 임포트 시도
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# 전략 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from strategy.vbt_strategy_012 import vbt_with_filters, get_vbt_indicators
from strategy.vbt_sell_strategy_013 import bailout_sell_strategy

# 전역 공유 데이터 (모듈 레벨)
GLOBAL_OHLCV = {}
GLOBAL_CACHED_DATA = {}

def init_worker():
    """워커 프로세스 시작 시 호출 (선택 사항)"""
    pass

def load_one_ticker(args_tuple):
    db_path, table_name, ticker = args_tuple
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(f"SELECT * FROM {table_name} WHERE ticker = ? ORDER BY date ASC", conn, params=(ticker,))
        conn.close()
        return ticker, df
    except: return ticker, None

def calc_pre_data(args_tuple):
    ticker, df, ema_rsi_list, window_list, vol_win_list, volatility_win_list = args_tuple
    res = {'indicators': {}, 'ranges': {}, 'volumes': {}, 'volatilities': {}}
    
    range_val = (df['high'] - df['low']).values
    shifted_range = pd.Series(range_val).shift(1)
    
    for ema, rsi in ema_rsi_list:
        res['indicators'][(ema, rsi)] = get_vbt_indicators(df, ema, rsi)
    for w in window_list:
        res['ranges'][w] = {
            'avg': shifted_range.rolling(window=w).mean().values,
            'std': shifted_range.rolling(window=w).std().values
        }
    
    # [추가] 거래량 및 변동성 사전 계산
    vol_val = df['volume'].values
    shifted_vol = pd.Series(vol_val).shift(1)
    for vw in vol_win_list:
        res['volumes'][vw] = shifted_vol.rolling(window=vw).mean().values
        
    for vvw in volatility_win_list:
        res['volatilities'][vvw] = shifted_range.rolling(window=vvw).mean().values
        
    return ticker, res

def get_params_list(func, config):
    sig = inspect.signature(func)
    exclude = ['df', 'entry_price', 'entry_idx', 'current_idx', 'position_type',
               'low_val', 'high_val', 'open_val', 'close_val', 'cached_indicators', 'cached_ranges']
    arg_names = [name for name in sig.parameters.keys() if name not in exclude]
    param_values = []
    for name in arg_names:
        list_key = name + '_list'
        val = config.get(list_key, [config.get(name, sig.parameters[name].default)])
        param_values.append(val)
    return [dict(zip(arg_names, combo)) for combo in itertools.product(*param_values)]

def run_vbt_backtest_core(df, entry_params, sell_params, cached_inds, cached_ranges):
    # vbt_with_filters를 직접 호출하는 대신 
    # 이미 최적화된 vbt_with_filters가 넘파이 연산을 수행함
    from strategy.vbt_strategy_012 import vbt_with_filters
    res = vbt_with_filters(df, **entry_params, cached_indicators=cached_inds, cached_ranges=cached_ranges)
    direc, r_short, r_long = res['vbt_direction'], res['reverse_to_short'], res['reverse_to_long']
    target_longs, target_shorts = res['target_long'], res['target_short']
    
    opens, highs, lows, closes = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    
    position, entry_price, entry_idx = 0, 0.0, 0
    pnls = []
    
    for i in range(1, len(df)):
        if position != 0:
            if (position == 1 and r_short[i]) or (position == -1 and r_long[i]):
                exit_p = opens[i]
                pnls.append((exit_p - entry_price) / entry_price if position == 1 else (entry_price - exit_p) / entry_price)
                position, entry_price, entry_idx = (-1 if position == 1 else 1), opens[i], i
                continue
            exit_sig, reason = bailout_sell_strategy(entry_price, entry_idx, i, 'long' if position == 1 else 'short',
                                                   lows[i], highs[i], opens[i], closes[i], **sell_params)
            if exit_sig:
                exit_p = opens[i] if 'profit' in reason else closes[i]
                pnls.append((exit_p - entry_price) / entry_price if position == 1 else (entry_price - exit_p) / entry_price)
                position = 0
        else:
            if direc[i] == 1:
                entry_p = max(opens[i], target_longs[i])
                position, entry_price, entry_idx = 1, entry_p, i
            elif direc[i] == -1:
                entry_p = min(opens[i], target_shorts[i])
                position, entry_price, entry_idx = -1, entry_p, i
    return pnls

def worker_task(chunk):
    if not GLOBAL_OHLCV: return []
    results = []
    for entry_p, sell_p in chunk:
        ema, rsi, window = entry_p['ema_period'], entry_p['rsi_period'], entry_p['window']
        all_pnls = []
        for ticker, df in GLOBAL_OHLCV.items():
            pre = GLOBAL_CACHED_DATA[ticker]
            # [수정] vbt_with_filters에서 직접 캐싱을 더 활용할 수도 있지만, 
            # 현재는Indicators와 Range만 사전에 넘김.
            all_pnls.extend(run_vbt_backtest_core(df, entry_p, sell_p, pre['indicators'].get((ema, rsi)), pre['ranges']))
        if all_pnls:
            trades_count = len(all_pnls)
            win_rate = len([p for p in all_pnls if p > 0]) / trades_count
            results.append({'entry': entry_p, 'sell': sell_p, 'trades': trades_count, 
                            'win_rate': win_rate, 'total_pnl': sum(all_pnls)})
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=os.getcwd())
    parser.add_argument('--market', type=str, default="coin")
    parser.add_argument('--interval', type=str, default="minute60")
    parser.add_argument('--processes', type=int, default=cpu_count())
    args = parser.parse_args()

    table_name = f'{args.market}_ohlcv_{args.interval}'
    db_path = os.path.join(args.root_dir, f'var/data/{table_name}.db')
    config_path = os.path.join(args.root_dir, 'sbin/strategy/vbt_config.json')

    print("--- [1/3] 데이터 병렬 로드 ---")
    conn = sqlite3.connect(db_path); cur = conn.cursor(); cur.execute(f"SELECT DISTINCT ticker FROM {table_name}"); all_tickers = [r[0] for r in cur.fetchall()]; conn.close()
    
    ohlcv_dict = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        it = ex.map(load_one_ticker, [(db_path, table_name, t) for t in all_tickers])
        for ticker, df in tqdm(it, total=len(all_tickers), desc="Loading") if tqdm else it:
            if df is not None: ohlcv_dict[ticker] = df

    print("\n--- [2/3] 지표 사전 계산 ---")
    with open(config_path, 'r', encoding='utf-8') as f: config = json.load(f)[0]
    ema_list = config['buy_signal_config'].get('ema_period_list', [config['buy_signal_config'].get('ema_period', 15)])
    rsi_list = config['buy_signal_config'].get('rsi_period_list', [config['buy_signal_config'].get('rsi_period', 8)])
    window_list = config['buy_signal_config'].get('window_list', [config['buy_signal_config'].get('window', 5)])
    vol_win_list = config['buy_signal_config'].get('volume_window_list', [20])
    volatility_win_list = config['buy_signal_config'].get('volatility_window_list', [20])
    ema_rsi_combi = list(itertools.product(ema_list, rsi_list))
    
    cached_dict = {}
    with ThreadPoolExecutor(max_workers=args.processes * 2) as ex:
        it = ex.map(calc_pre_data, [(t, df, ema_rsi_combi, window_list, vol_win_list, volatility_win_list) for t, df in ohlcv_dict.items()])
        for t, res in tqdm(it, total=len(ohlcv_dict), desc="Caching") if tqdm else it:
            cached_dict[t] = res

    # 리눅스 Fork를 통한 전역 데이터 공유 (Assign to module-level globals)
    GLOBAL_OHLCV.update(ohlcv_dict)
    GLOBAL_CACHED_DATA.update(cached_dict)

    # 3. 백테스트 실행
    print("\n--- [3/3] 백테스트 시뮬레이션 ---")
    entry_combi = get_params_list(vbt_with_filters, config['buy_signal_config'])
    sell_combi = get_params_list(bailout_sell_strategy, config['sell_signal_config'])
    all_combis = list(itertools.product(entry_combi, sell_combi))
    
    chunk_size = max(50, len(all_combis) // (args.processes * 100))
    chunks = [all_combis[i:i+chunk_size] for i in range(0, len(all_combis), chunk_size)]
    
    print(f"전체 조합: {len(all_combis)}, 프로세스: {args.processes}")
    
    final_results = []
    with Pool(args.processes) as pool:
        it = pool.imap_unordered(worker_task, chunks)
        if tqdm: it = tqdm(it, total=len(chunks), desc="Testing")
        for res_list in it:
            if res_list: final_results.extend(res_list)

    if final_results:
        sorted_res = sorted(final_results, key=lambda x: x['total_pnl'], reverse=True)
        print("\n### TOP 10 RESULTS ###")
        for r in sorted_res[:10]:
            print(f"PnL: {r['total_pnl']:.4f} | Win: {r['win_rate']:.2f} | Trades: {r['trades']} | Entry: {r['entry']} | Sell: {r['sell']}")
    else:
        print("거래 내역이 없습니다. (No trades found)")
