import FinanceDataReader as fdr
import time
import datetime
import os
import pandas as pd
import pykrx as krx
import argparse
import pyupbit
import traceback
import sys
import sqlite3

pd.options.display.max_columns = None
pd.options.display.max_rows = None

def isNotDataframeOrEmpty(df):
    return not isinstance(df, pd.core.frame.DataFrame) or (isinstance(df, pd.core.frame.DataFrame) and df.empty)

def interval_to_timedelta(interval: str) -> datetime.timedelta:
    """Convert interval string to timedelta."""
    if interval == "minute1":
        return datetime.timedelta(minutes=1)
    elif interval == "minute3":
        return datetime.timedelta(minutes=3)
    elif interval == "minute5":
        return datetime.timedelta(minutes=5)
    elif interval == "minute10":
        return datetime.timedelta(minutes=10)
    elif interval == "minute15":
        return datetime.timedelta(minutes=15)
    elif interval == "minute30":
        return datetime.timedelta(minutes=30)
    elif interval == "minute60":
        return datetime.timedelta(hours=1)
    elif interval == "minute240":
        return datetime.timedelta(hours=4)
    elif interval == "day":
        return datetime.timedelta(days=1)
    elif interval == "week":
        return datetime.timedelta(weeks=1)
    elif interval == "month":
        return datetime.timedelta(days=30)  # approximate
    else:
        raise ValueError(f"Unknown interval: {interval}")

def find_and_fill_gaps(df: pd.DataFrame, code: str, interval: str, expected_delta: datetime.timedelta) -> pd.DataFrame:
    """Find gaps in data and fill them by fetching missing data."""
    if isNotDataframeOrEmpty(df) or len(df) < 2:
        return df
    
    df_sorted = df.sort_index()
    total_gaps_filled = 0
    max_gap_iterations = 2  # prevent infinite loops
    
    for iteration in range(max_gap_iterations):
        print("find_and_fill_gaps iteration : %d" %(iteration))
        gaps_found = []
        
        # Find all gaps using vectorized operations
        if len(df_sorted) < 2:
            break
        
        # Calculate time differences between consecutive rows
        time_diffs = df_sorted.index[1:] - df_sorted.index[:-1]
        
        # Find gaps larger than expected interval (with tolerance)
        gap_mask = time_diffs > expected_delta * 1.5
        
        # Create list of (start, end) tuples for gaps
        gap_starts = df_sorted.index[:-1][gap_mask]
        gap_ends = df_sorted.index[1:][gap_mask]
        gaps_found = list(zip(gap_starts, gap_ends))
        
        if not gaps_found:
            break  # No more gaps
        
        # Fill gaps by fetching data for each gap
        new_dfs = []
        iteration_gaps_filled = 0
        print("gaps_found count : %d" %(len(gaps_found)))
        for gap_start, gap_end in gaps_found:
            # Calculate how many records should be in this gap
            expected_count = int((gap_end - gap_start) / expected_delta) - 1
            if expected_count <= 0:
                continue
            
            # Fetch data for the gap period (walk backward from gap_end)
            gap_dfs = []
            to_cursor = gap_end
            gap_max_iter = min(100, expected_count // 200 + 2)  # reasonable limit
            
            for _ in range(gap_max_iter):
                # 날짜까지만 전달
                gap_to_str = to_cursor.strftime("%Y-%m-%d")
                # 분봉의 경우 하루(1440분)를 커버하도록 count=1440 사용
                if interval.startswith("minute"):
                    gap_count = 1440
                else:
                    gap_count = 200

                gap_df = pyupbit.get_ohlcv(code, interval=interval, count=gap_count, to=gap_to_str)
                if isNotDataframeOrEmpty(gap_df):
                    break
                
                # Filter to only include data within the gap
                gap_df = gap_df[(gap_df.index > gap_start) & (gap_df.index < gap_end)]
                if gap_df.empty:
                    break
                
                gap_dfs.append(gap_df)
                
                # Move cursor backward by one interval from earliest_in_gap
                earliest_in_gap = gap_df.index.min()
                if earliest_in_gap <= gap_start:
                    break

                to_cursor = earliest_in_gap - expected_delta
                time.sleep(0.11)  # throttle
            
            if gap_dfs:
                gap_merged = pd.concat(gap_dfs)
                gap_merged = gap_merged[(gap_merged.index > gap_start) & (gap_merged.index < gap_end)]
                gap_merged = gap_merged.sort_index().drop_duplicates()
                if not gap_merged.empty:
                    new_dfs.append(gap_merged)
                    iteration_gaps_filled += len(gap_merged)
            
            time.sleep(0.11)  # throttle between gaps
        
        # Merge new data with existing data
        if new_dfs:
            new_data = pd.concat(new_dfs)
            new_data = new_data.sort_index().drop_duplicates()
            df_sorted = pd.concat([df_sorted, new_data])
            df_sorted = df_sorted.sort_index().drop_duplicates()
            total_gaps_filled += iteration_gaps_filled
        
        if iteration_gaps_filled == 0:
            break  # No progress made, stop
        
        print(f"  Gap fill iteration {iteration + 1}: Found {len(gaps_found)} gaps, filled {iteration_gaps_filled} records")
    
    if total_gaps_filled > 0:
        print(f"  Total gaps filled: {total_gaps_filled} records")
    
    return df_sorted

def fetch_coin_ohlcv(code: str, interval: str, start_dt=None, end_dt=None):
    """
    Fetch candles for a date window or all history.
    - "all": start_dt is None, end_dt is today; walk backward until no data.
    - specific date: fetch [start_dt, end_dt).
    Cursor moves to the earliest timestamp returned each batch (no fixed step).
    """
    dfs = []
    to_cursor = end_dt if end_dt else datetime.datetime.now()
    max_iter = 100000  # safety guard
    last_earliest = None
    expected_delta = interval_to_timedelta(interval)

    for _ in range(max_iter):
        print("%s\t%s\t%d" %(code, interval, _))
        # pyupbit 'to' 는 일 단위까지만 사용되는 것으로 보이므로 날짜만 전달
        to_str = to_cursor.strftime("%Y-%m-%d")
        # interval 이 분봉인 경우, 하루(1440분)를 커버하도록 count=1440 사용
        if interval.startswith("minute"):
            count = 1440
        else:
            count = 200

        df = pyupbit.get_ohlcv(code, interval=interval, count=count, to=to_str)
        if isNotDataframeOrEmpty(df):
            break
        # print(df.head())
        dfs.append(df)

        earliest = df.index.min()
        if start_dt and earliest <= start_dt:
            break

        if last_earliest is not None and earliest >= last_earliest:
            break
        last_earliest = earliest

        # 다음 페이지 요청용 to 는 현재 배치의 최소 인덱스에서 interval 만큼 뺀 값으로 설정
        to_cursor = earliest - expected_delta
        time.sleep(0.11)  # throttle to avoid rate-limit

    if not dfs:
        return None

    merged = pd.concat(dfs)
    if start_dt and end_dt:
        merged = merged[(merged.index >= start_dt) & (merged.index < end_dt)]
    merged = merged.sort_index().drop_duplicates()
    
    # Fill gaps in the data
    expected_delta = interval_to_timedelta(interval)
    merged = find_and_fill_gaps(merged, code, interval, expected_delta)
    
    return merged

def clean_and_save_to_db(df: pd.DataFrame, conn: sqlite3.Connection, table_name: str, ticker: str, interval: str):
    """Save OHLCV data to SQLite database table with upsert logic."""
    if 'value' in df.columns:
        df = df.drop('value', axis=1)
    
    df['ticker'] = ticker
    df['date'] = df.index
    df['date'] = df['date'].apply(lambda d: d.tz_localize(None) if getattr(d, "tzinfo", None) else d)

    # date formatting depending on interval granularity
    if interval.startswith("minute"):
        fmt = "%Y%m%d%H%M"
    else:
        fmt = "%Y%m%d"
    df['date'] = df['date'].apply(lambda d: d.strftime(fmt))

    df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
    
    # Use INSERT OR REPLACE for upsert
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute(f"""
            INSERT OR REPLACE INTO {table_name} (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (row['ticker'], row['date'], row['open'], row['high'], row['low'], row['close'], row['volume']))
    conn.commit()


def create_table_if_not_exists(conn: sqlite3.Connection, table_name: str):
    """Create OHLCV table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()


#--market : [coin, korea, overseas, ...]
#--interval : [day, minute1, minute3, minute5, minute10, minute15, minute30, minute60, minute240, week, month]
#--date : {YYYYMMDD or "all"}
#--output_dir : {output_dir for db file}
parser = argparse.ArgumentParser(description='get_daily_ohlcv_data')
parser.add_argument('--root_dir', type=str, default="/Users/yongbeom/project/coin_volume_trader")
parser.add_argument('--date', type=str, required=True, help="YYYYMMDD or all")
parser.add_argument('--market', type=str, default="coin")
parser.add_argument('--interval', type=str, default="minute1")
parser.add_argument('--output_dir', type=str, default="var/data")
args = parser.parse_args()

# Create output directory for db file
output_dir = os.path.join(args.root_dir, args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# DB file path: {output_dir}/{market}_ohlcv_{interval}.db
db_path = os.path.join(output_dir, f"{args.market}_ohlcv_{args.interval}.db")
conn = sqlite3.connect(db_path)

ALLOWED_INTERVALS = ["day", "minute1", "minute3", "minute5", "minute10", "minute15", "minute30", "minute60", "minute240", "week", "month"]

if args.interval not in ALLOWED_INTERVALS:
    print(f"Unsupported interval: {args.interval}. Use one of {ALLOWED_INTERVALS}")
    sys.exit(1)

interval = args.interval

# get start and end date (for "all", start_dt=None, end_dt=today)
if args.date.lower() == "all":
    start_dt = None
    end_dt = datetime.datetime.now()
else:
    date = datetime.datetime.strptime(args.date, "%Y%m%d")
    start_dt = datetime.datetime(date.year, date.month, date.day)
    end_dt = start_dt + datetime.timedelta(days=1)


if args.market == 'coin':
    tickers = [t for t in pyupbit.get_tickers() if 'KRW-' in t]
    # tickers = ["KRW-BTC"]
    print("tickers len :", len(tickers))
    
    # Table name: {market}_ohlcv_{interval}
    table_name = f"{args.market}_ohlcv_{args.interval}"
    create_table_if_not_exists(conn, table_name)
    print(f"Using table: {table_name} in DB: {db_path}")

    for ticker_i, ticker in enumerate(tickers):
        print(f"start {ticker_i+1}. {ticker}")
        try:
            df = fetch_coin_ohlcv(ticker, interval, start_dt, end_dt)
            if isNotDataframeOrEmpty(df):
                print(f"{ticker} - no data")
                continue
            
            clean_and_save_to_db(df, conn, table_name, ticker, interval)
            print(f"{ticker} - saved {len(df)} records")
        except Exception:
            print(f"error on {ticker}")
            print(traceback.format_exc())
            time.sleep(1.0)
    
    conn.close()
    print(f"All data saved to {db_path}")