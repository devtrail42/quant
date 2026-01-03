import pandas as pd
import numpy as np

def bailout_sell_strategy(
    entry_price,
    entry_idx,
    current_idx,
    position_type, # 'long' or 'short'
    low_val,
    high_val,
    open_val,
    close_val,
    stop_loss_ratio = 0.02,
    bailout_profit_days = 1,
    bailout_no_profit_days = 4,
    price_flow_sluggish_threshold = 0.005
):
    """
    Bailout 및 Stop Loss 복합 전략 (Numpy 연산 친화적으로 변경)
    """
    bars_held = current_idx - entry_idx
    
    # 1. Stop Loss 체크
    if position_type == 'long':
        if low_val <= entry_price * (1 - stop_loss_ratio):
            return True, 'stop_loss'
    else: # short
        if high_val >= entry_price * (1 + stop_loss_ratio):
            return True, 'stop_loss'
            
    # 2. Bailout Profit-taking
    if bars_held >= bailout_profit_days:
        if (position_type == 'long' and open_val > entry_price) or \
           (position_type == 'short' and open_val < entry_price):
            return True, 'bailout_profit'
                
    # 3. Bailout No Profit
    if bars_held >= bailout_no_profit_days:
        if (position_type == 'long' and close_val <= entry_price) or \
           (position_type == 'short' and close_val >= entry_price):
            return True, 'bailout_timeout'
                
    return False, 'none'
