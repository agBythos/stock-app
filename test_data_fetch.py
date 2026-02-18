"""測試數據獲取"""
import yfinance as yf
from datetime import datetime, timedelta

# 測試參數
symbol = "2330.TW"
test_start = "2023-07-01"
test_end = "2023-07-30"

# 計算預熱期開始日
train_end = datetime.strptime("2023-06-30", "%Y-%m-%d")
warmup_start = train_end - timedelta(days=60)
warmup_start_str = warmup_start.strftime("%Y-%m-%d")

print(f"預熱期起始: {warmup_start_str}")
print(f"測試期: {test_start} to {test_end}")

# 獲取數據
ticker = yf.Ticker(symbol)
full_data = ticker.history(start=warmup_start_str, end=test_end)
test_data = ticker.history(start=test_start, end=test_end)

print(f"\n完整數據（含預熱期）: {len(full_data)} 行")
print(full_data.head())
print(f"\n測試數據: {len(test_data)} 行")
print(test_data.head())
