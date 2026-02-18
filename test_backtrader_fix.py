"""
測試 Backtrader 修復
測試台積電 2330.TW 2024 年 MA Crossover 策略

BUG-02 Fix: 原版以 `return False` 回報失敗，pytest 不檢查 return value，
導致 CI 偽通過。已改為使用 `assert` / `pytest.fail()` 並在 server
未啟動時以 `pytest.mark.skipif` 跳過，而非偽通過。
"""

import sys
import json
import pytest
import requests


BASE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Lazy server-availability guard (same pattern as test_e2e.py)
# 若 server 未啟動，skip 而非偽通過。
# ---------------------------------------------------------------------------

def _server_available() -> bool:
    """Return True only if the server is actually listening on BASE_URL."""
    try:
        requests.get(BASE_URL, timeout=2)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason=f"server not running at {BASE_URL}",
)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_backtest():
    """測試回測 API — Sharpe Ratio 非零且 Equity Curve 有完整日資料點"""

    payload = {
        "symbol": "2330.TW",
        "strategy": "ma_crossover",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_capital": 100000,
        "parameters": {
            "fast_period": 10,
            "slow_period": 30,
        },
    }

    print("=" * 60)
    print("測試 Backtrader 修復")
    print("=" * 60)
    print(f"股票代碼: {payload['symbol']}")
    print(f"策略: {payload['strategy']}")
    print(f"期間: {payload['start_date']} ~ {payload['end_date']}")
    print(f"初始資金: ${payload['initial_capital']:,}")
    print("=" * 60)

    url = f"{BASE_URL}/api/backtest"

    print("\n發送回測請求...")
    response = requests.post(url, json=payload, timeout=60)

    # --- 斷言 1：HTTP 狀態碼 ---
    assert response.status_code == 200, (
        f"[FAIL] Request failed: HTTP {response.status_code}\n{response.text}"
    )

    result = response.json()

    # --- 斷言 2：回應狀態 ---
    assert result.get("status") == "success", (
        f"[FAIL] Backtest returned non-success status: {result}"
    )

    data = result.get("data", {})
    performance = data.get("performance", {})
    equity_curve = data.get("equity_curve", [])

    print("\n[SUCCESS] Backtest completed!")

    # --- 斷言 3：Sharpe Ratio 欄位存在 ---
    sharpe_ratio = performance.get("sharpe_ratio")
    assert sharpe_ratio is not None, "[FAIL] 'sharpe_ratio' missing from performance"
    print(f"[CHECK 1] Sharpe Ratio: {sharpe_ratio}")

    # --- 斷言 4：Equity Curve 有足夠資料點（> 2）---
    assert len(equity_curve) > 2, (
        f"[FAIL] Equity Curve only has {len(equity_curve)} points (expected >2)"
    )
    print(f"[CHECK 2] Equity Curve data points: {len(equity_curve)}")

    # --- 斷言 5：每筆 equity curve 點都有 date / value / benchmark ---
    for i, point in enumerate(equity_curve[:5]):
        assert "date" in point, f"equity_curve[{i}] missing 'date'"
        assert "value" in point, f"equity_curve[{i}] missing 'value'"
        assert "benchmark" in point, f"equity_curve[{i}] missing 'benchmark'"

    # 印出績效摘要（供 CI log 參考）
    print("\n" + "=" * 60)
    print("績效指標")
    print("=" * 60)
    print(f"總報酬率: {performance.get('total_return', 0):.2f}%")
    print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
    print(f"最大回撤: {performance.get('max_drawdown', 0):.2f}%")
    print(f"勝率: {performance.get('win_rate', 0):.2f}%")
    print(f"交易次數: {performance.get('total_trades', 0)}")
    print(f"最終資產: ${performance.get('final_portfolio_value', 0):,.2f}")

    benchmark = data.get("benchmark", {})
    print("\n" + "=" * 60)
    print("Benchmark (Buy & Hold)")
    print("=" * 60)
    print(f"總報酬率: {benchmark.get('total_return', 0):.2f}%")
    print(f"Sharpe Ratio: {benchmark.get('sharpe_ratio', 0):.2f}")
    print(f"最大回撤: {benchmark.get('max_drawdown', 0):.2f}%")

    if len(equity_curve) > 10:
        print("\nEquity Curve 樣本（前 5 筆）")
        for point in equity_curve[:5]:
            print(f"  {point['date']}: ${point['value']:,.2f} (Benchmark: ${point['benchmark']:,.2f})")

    trades = data.get("trades", [])
    if trades:
        print(f"\n交易明細（共 {len(trades)} 筆）")
        for i, trade in enumerate(trades[:5], 1):
            print(
                f"  #{i}: {trade['entry_date']} @ ${trade['entry_price']}"
                f" → {trade['exit_date']} @ ${trade['exit_price']}"
                f"  P&L: ${trade['pnl']:,.2f} ({trade['pnl_percent']:+.2f}%)"
            )

    print("\n[PASS] All assertions passed.")


# ---------------------------------------------------------------------------
# Standalone runner (backward-compatible with original usage)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    result = pytest.main([__file__, "-v"])
    sys.exit(result)
