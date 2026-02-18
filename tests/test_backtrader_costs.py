"""
tests/test_backtrader_costs.py — 台灣交易成本測試
===============================================

測試項目（Suite 3 — TWSE cost model）：
8. 買入成本 = commission × discount only（無稅）
9. 賣出成本 = commission × discount + 0.3% 證交稅
10. 單趟來回成本 between 0.40%–0.50%
11. TWAECommission._getcommission: 買 vs 賣 side paths
12. End-to-end backtrader fold 執行無 crash

作者：Bythos（sub-agent）
建立：2026-02-18
"""

import numpy as np
import pandas as pd
import pytest

from config.walk_forward import WalkForwardConfig
from validation.backtrader_bridge import TWAECommission, run_backtrader_fold


# ============================================================================
# Test 8: 買入成本（無稅）
# ============================================================================


def test_buy_commission_no_tax():
    """買入成本 = commission × discount（無證交稅）"""
    cfg = WalkForwardConfig(
        commission_rate=0.001425,
        commission_discount=0.6,
        sell_tax_rate=0.003,
    )
    comm = TWAECommission.from_config(cfg)

    price = 100.0
    size = 1000  # 買入 1000 股

    cost = comm._getcommission(size, price, pseudoexec=False)
    expected = 100.0 * 1000 * cfg.effective_buy_rate  # 0.000855

    assert abs(cost - expected) < 0.01, (
        f"買入成本不符：expected={expected:.2f}, actual={cost:.2f}"
    )
    assert cost == pytest.approx(85.5, abs=0.1), f"買入成本={cost:.2f}（預期 85.5）"


# ============================================================================
# Test 9: 賣出成本（含稅）
# ============================================================================


def test_sell_commission_with_tax():
    """賣出成本 = commission × discount + 0.3% 證交稅"""
    cfg = WalkForwardConfig(
        commission_rate=0.001425,
        commission_discount=0.6,
        sell_tax_rate=0.003,
    )
    comm = TWAECommission.from_config(cfg)

    price = 100.0
    size = -1000  # 賣出 1000 股（負數）

    cost = comm._getcommission(size, price, pseudoexec=False)
    expected = 100.0 * 1000 * cfg.effective_sell_rate  # 0.003855

    assert abs(cost - expected) < 0.01, (
        f"賣出成本不符：expected={expected:.2f}, actual={cost:.2f}"
    )
    assert cost == pytest.approx(385.5, abs=0.1), f"賣出成本={cost:.2f}（預期 385.5）"


# ============================================================================
# Test 10: 單趟來回成本 0.40%–0.50%
# ============================================================================


def test_round_trip_cost_range():
    """單趟來回成本（買+賣）between 0.40%–0.50%"""
    cfg = WalkForwardConfig()
    comm = TWAECommission.from_config(cfg)

    price = 100.0
    size = 1000

    buy_cost = comm._getcommission(size, price, False)
    sell_cost = comm._getcommission(-size, price, False)
    total_cost = buy_cost + sell_cost

    # 成交金額
    trade_value = price * size
    cost_pct = (total_cost / trade_value) * 100

    assert 0.40 <= cost_pct <= 0.50, (
        f"單趟來回成本={cost_pct:.4f}% 超出範圍 [0.40%, 0.50%]"
    )
    print(f"✓ 單趟來回成本: {cost_pct:.4f}% (buy={buy_cost:.2f} + sell={sell_cost:.2f})")


# ============================================================================
# Test 11: _getcommission buy vs sell paths
# ============================================================================


def test_getcommission_buy_vs_sell():
    """驗證 _getcommission 買入/賣出路徑不同"""
    cfg = WalkForwardConfig()
    comm = TWAECommission.from_config(cfg)

    price = 100.0
    buy_size = 1000
    sell_size = -1000

    buy_cost = comm._getcommission(buy_size, price, False)
    sell_cost = comm._getcommission(sell_size, price, False)

    # 賣出成本應高於買入成本（因證交稅）
    assert sell_cost > buy_cost, (
        f"賣出成本={sell_cost:.2f} 應 > 買入成本={buy_cost:.2f}"
    )

    # 賣出成本 ≈ 買入成本 × 4.5（0.3855 / 0.0855 ≈ 4.5）
    ratio = sell_cost / buy_cost
    assert 4.0 < ratio < 5.0, f"賣出/買入成本比={ratio:.2f}（預期 4–5）"


# ============================================================================
# Test 12: End-to-end backtrader fold 執行無 crash
# ============================================================================


def test_backtrader_fold_no_crash():
    """End-to-end run_backtrader_fold() 執行完成無 crash"""
    np.random.seed(42)
    n_bars = 60

    # 建立合成 OHLCV 資料
    close = np.cumsum(np.random.randn(n_bars) * 0.5) + 100
    high = close + np.abs(np.random.randn(n_bars) * 0.3)
    low = close - np.abs(np.random.randn(n_bars) * 0.3)
    open_ = close + np.random.randn(n_bars) * 0.2
    volume = np.random.randint(1000, 10000, n_bars)

    price_df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n_bars, freq="D"),
    )

    # 建立信號字典（隨機 0/1）
    signals = {
        date.date().isoformat(): float(np.random.choice([0, 1]))
        for date in price_df.index
    }

    cfg = WalkForwardConfig(initial_capital=100_000)

    # 執行回測
    result = run_backtrader_fold(
        price_df=price_df,
        signals=signals,
        config=cfg,
        fold_id=1,
    )

    # 驗證回傳結果
    assert result is not None
    assert result.fold_id == 1
    assert result.initial_capital == 100_000
    assert result.final_value > 0
    assert not result.skipped or result.error is not None

    print(f"✓ Backtrader fold 執行成功: return={result.total_return_pct:.2f}%")


# ============================================================================
# Test 13: 非對稱成本驗證（買<賣）
# ============================================================================


def test_asymmetric_costs():
    """驗證台灣成本非對稱性：賣出成本 > 買入成本"""
    cfg = WalkForwardConfig()

    buy_rate = cfg.effective_buy_rate
    sell_rate = cfg.effective_sell_rate

    assert sell_rate > buy_rate, (
        f"賣出成本率={sell_rate:.6f} 應 > 買入成本率={buy_rate:.6f}"
    )
    assert sell_rate == pytest.approx(0.003855, abs=1e-6)
    assert buy_rate == pytest.approx(0.000855, abs=1e-6)

    print(f"✓ 非對稱成本: buy={buy_rate:.4%}, sell={sell_rate:.4%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
