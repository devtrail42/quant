import sqlite3

conn = sqlite3.connect("/Users/yongbeom/cyb/project/2025/quant/var/data/coin_ohlcv_minute1.db")
cur = conn.cursor()

table = "coin_ohlcv_minute1"
column = "ticker"

cur.execute(f"SELECT DISTINCT {column} FROM {table}")
values = [row[0] for row in cur.fetchall()]

print(values)

conn.close()
