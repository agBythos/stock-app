"""
tests/test_cpcv_runner.py — CPCV Runner 測試
=============================================

測試項目（4 tests）：
  T1. paths_structure  : CPCVRunner.run() 回傳正確 path 數與結構
  T2. pbo_range        : PBO（P≤0）值域在 [0, 1]
  T3. sharpe_stats     : _compute_path_stats() 統計正確（mean, std, ci_95）
  T4. full_integration : 端對端執行（小資料，關閉 Backtrader）

注意：完整 ML + Backtrader 測試較慢，T4 使用 run_backtrader=False 快速跑。

作者：Bythos（sub-agent phase4-cpcv-tests）
建立：2026-02-18
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from math import sqrt

from config.cpcv import CPCVConfig
from validation.cpcv_splitter import CPCVSplitter
from validation.cpcv_runner import CPCVRunner, CPCVReport, CPCVPathReport


# ============================================================================
# Helper：建立假 OHLCV DataFrame
# ============================================================================

def make_ohlcv(n_bars: int, seed: int = 0) -> pd.DataFrame:
    """建立有時序索引的假 OHLCV DataFrame（Close 正趨勢）"""
    rng = np.random.default_rng(seed)
    # 加入上升趨勢讓 ML 有東西學
    trend = np.linspace(100, 150, n_bars)
    noise = rng.standard_normal(n_bars) * 2
    close = trend + noise
    close = np.maximum(close, 10)  # 防止負值

    vol = rng.integers(10_000, 500_000, n_bars).astype(float)

    return pd.DataFrame(
        {
            "Open":   close * (1 + rng.uniform(-0.005, 0.005, n_bars)),
            "High":   close * (1 + rng.uniform(0.001, 0.015, n_bars)),
            "Low":    close * (1 - rng.uniform(0.001, 0.015, n_bars)),
            "Close":  close,
            "Volume": vol,
        },
        index=pd.date_range("2018-01-01", periods=n_bars, freq="B"),
    )


def make_path_reports(sharpes: list[float]) -> list[CPCVPathReport]:
    """建立帶指定 Sharpe 值的假 CPCVPathReport 列表"""
    return [
        CPCVPathReport(
            path_id=i + 1,
            group_id=i,
            fold_ids=[i + 1],
            n_bars=100,
            total_return_pct=s * 5.0,   # 假設 Sharpe 與 return 正比
            sharpe_ratio=s,
            max_drawdown_pct=-10.0,
            calmar_ratio=abs(s * 5.0) / 10.0,
        )
        for i, s in enumerate(sharpes)
    ]


# ============================================================================
# T1: paths_structure — run() 回傳正確 path 數與結構
# ============================================================================

def test_paths_count_and_structure():
    """
    CPCVRunner.run() 應回傳 n_backtest_paths 條 paths，
    每條 path 有 path_id, group_id, fold_ids, sharpe_ratio 等欄位。
    """
    cfg = CPCVConfig(
        n_groups=4,
        k_test_groups=1,
        label_horizon=3,
        embargo_bars=3,
        min_train_samples=50,
    )
    runner = CPCVRunner(cfg)
    df = make_ohlcv(800)  # 充足的資料

    report = runner.run(df, symbol="TEST", verbose=False, run_backtrader=False)

    # path 數 = n_backtest_paths = k×C(N,k)/N = 1×4/4 = 1
    expected_phi = cfg.n_backtest_paths
    assert report.n_backtest_paths == expected_phi, (
        f"n_backtest_paths={report.n_backtest_paths}，預期 {expected_phi}"
    )
    assert len(report.paths) == expected_phi, (
        f"paths 數量={len(report.paths)}，預期 {expected_phi}"
    )

    # 結構檢查
    for path in report.paths:
        d = path.to_dict()
        assert "path_id" in d
        assert "group_id" in d
        assert "fold_ids" in d
        assert "sharpe_ratio" in d
        assert isinstance(d["path_id"], int)
        assert isinstance(d["fold_ids"], list)


def test_report_fold_count():
    """CPCVReport 的 n_total_folds 應 = C(N,k)"""
    from math import comb
    cfg = CPCVConfig(
        n_groups=4,
        k_test_groups=1,
        min_train_samples=50,
    )
    runner = CPCVRunner(cfg)
    df = make_ohlcv(800)

    report = runner.run(df, symbol="TEST", verbose=False, run_backtrader=False)

    assert report.n_total_folds == comb(4, 1), (
        f"n_total_folds={report.n_total_folds}，預期 {comb(4, 1)}"
    )
    assert report.n_total_folds == report.n_active_folds + report.n_skipped_folds


# ============================================================================
# T2: pbo_range — PBO 值域在 [0, 1]
# ============================================================================

def test_pbo_range_all_positive():
    """所有 Sharpe > 0 時，PBO 應 = 0.0"""
    runner = CPCVRunner.__new__(CPCVRunner)
    runner.cfg = CPCVConfig()

    paths = make_path_reports([0.5, 1.0, 1.5, 0.8, 2.0])  # 全正
    stats = runner._compute_path_stats(paths)

    assert stats["pbo"] == 0.0, f"PBO 應為 0.0，實際 {stats['pbo']}"


def test_pbo_range_all_negative():
    """所有 Sharpe ≤ 0 時，PBO 應 = 1.0"""
    runner = CPCVRunner.__new__(CPCVRunner)
    runner.cfg = CPCVConfig()

    paths = make_path_reports([-0.5, -1.0, -1.5, -0.1, 0.0])  # 全 ≤ 0
    stats = runner._compute_path_stats(paths)

    assert stats["pbo"] == 1.0, f"PBO 應為 1.0，實際 {stats['pbo']}"


def test_pbo_range_mixed():
    """混合正負 Sharpe 時，PBO 應在 (0, 1) 之間"""
    runner = CPCVRunner.__new__(CPCVRunner)
    runner.cfg = CPCVConfig()

    paths = make_path_reports([1.0, -0.5, 0.8, -1.0, 1.5])  # 2/5 ≤ 0
    stats = runner._compute_path_stats(paths)

    assert 0.0 < stats["pbo"] < 1.0, (
        f"PBO={stats['pbo']} 應在 (0,1)"
    )
    assert abs(stats["pbo"] - 0.4) < 1e-9, (
        f"2/5 path Sharpe ≤ 0 → PBO 應 = 0.4，實際 {stats['pbo']}"
    )


def test_pbo_empty_paths():
    """無有效 paths 時，PBO 應為 1.0（最悲觀）"""
    runner = CPCVRunner.__new__(CPCVRunner)
    runner.cfg = CPCVConfig()

    # 全部有 error 的 paths
    from validation.cpcv_runner import CPCVPathReport
    error_paths = [
        CPCVPathReport(path_id=1, group_id=0, fold_ids=[1], n_bars=0, error="test error")
    ]

    stats = runner._compute_path_stats(error_paths)
    assert stats["pbo"] == 1.0, f"無有效 paths 時 PBO 應 = 1.0"


# ============================================================================
# T3: sharpe_stats — _compute_path_stats() 統計正確
# ============================================================================

def test_sharpe_stats_mean_std():
    """_compute_path_stats() 計算均值與標準差正確"""
    sharpes = [1.0, 2.0, 3.0, 4.0, 5.0]
    runner = CPCVRunner.__new__(CPCVRunner)
    runner.cfg = CPCVConfig()

    paths = make_path_reports(sharpes)
    stats = runner._compute_path_stats(paths)

    expected_mean = np.mean(sharpes)  # = 3.0
    expected_std  = np.std(sharpes, ddof=1)  # sample std

    assert abs(stats["sharpe_mean"] - expected_mean) < 1e-9, (
        f"均值={stats['sharpe_mean']:.6f}，預期 {expected_mean:.6f}"
    )
    assert abs(stats["sharpe_std"] - expected_std) < 1e-9, (
        f"標準差={stats['sharpe_std']:.6f}，預期 {expected_std:.6f}"
    )


def test_sharpe_stats_ci95():
    """95% CI 計算正確（mean ± 1.96 × std/√n）"""
    sharpes = [0.5, 1.5, 2.0, 1.0, 2.5]
    runner = CPCVRunner.__new__(CPCVRunner)
    runner.cfg = CPCVConfig()

    paths = make_path_reports(sharpes)
    stats = runner._compute_path_stats(paths)

    n = len(sharpes)
    mean_s = np.mean(sharpes)
    std_s  = np.std(sharpes, ddof=1)
    margin = 1.96 * std_s / sqrt(n)

    expected_lower = mean_s - margin
    expected_upper = mean_s + margin

    ci = stats["sharpe_ci_95"]
    assert abs(ci[0] - expected_lower) < 1e-9, (
        f"CI 下界={ci[0]:.6f}，預期 {expected_lower:.6f}"
    )
    assert abs(ci[1] - expected_upper) < 1e-9, (
        f"CI 上界={ci[1]:.6f}，預期 {expected_upper:.6f}"
    )


def test_sharpe_stats_return_mean():
    """return_mean 計算正確"""
    sharpes = [1.0, 2.0, 3.0]
    runner = CPCVRunner.__new__(CPCVRunner)
    runner.cfg = CPCVConfig()

    paths = make_path_reports(sharpes)
    stats = runner._compute_path_stats(paths)

    # make_path_reports: total_return_pct = sharpe * 5.0
    expected_return_mean = np.mean([s * 5.0 for s in sharpes])
    assert abs(stats["return_mean"] - expected_return_mean) < 1e-9, (
        f"return_mean={stats['return_mean']:.4f}，預期 {expected_return_mean:.4f}"
    )


# ============================================================================
# T4: full_integration — 端對端執行（關閉 Backtrader）
# ============================================================================

def test_full_integration_no_backtrader():
    """
    完整端對端測試：CPCVRunner.run() → CPCVReport。
    關閉 Backtrader（run_backtrader=False）以加速。
    驗證 report 結構完整、folds 和 paths 非空。
    """
    cfg = CPCVConfig(
        n_groups=4,
        k_test_groups=1,
        label_horizon=3,
        embargo_bars=3,
        min_train_samples=30,
    )
    runner = CPCVRunner(cfg)
    df = make_ohlcv(600)  # 約 2.4 年的週資料

    report = runner.run(df, symbol="INTEGRATION_TEST", verbose=False, run_backtrader=False)

    # ── 結構完整性 ─────────────────────────────────────────────────
    assert isinstance(report, CPCVReport)
    assert report.symbol == "INTEGRATION_TEST"
    assert report.run_timestamp != ""

    # ── folds ──────────────────────────────────────────────────────
    assert report.n_total_folds == 4, f"C(4,1)=4 folds，實際 {report.n_total_folds}"
    assert len(report.folds) == 4

    # ── paths ──────────────────────────────────────────────────────
    assert len(report.paths) == cfg.n_backtest_paths, (
        f"paths={len(report.paths)}，預期 {cfg.n_backtest_paths}"
    )

    # ── to_dict() 序列化 ────────────────────────────────────────────
    d = report.to_dict()
    assert "summary" in d
    assert "folds" in d
    assert "paths" in d
    summary = d["summary"]
    assert "sharpe_mean" in summary
    assert "sharpe_pbo" in summary
    assert "sharpe_ci_95" in summary
    assert isinstance(summary["sharpe_ci_95"], list)
    assert len(summary["sharpe_ci_95"]) == 2

    # ── PBO 值域 ────────────────────────────────────────────────────
    assert 0.0 <= summary["sharpe_pbo"] <= 1.0, (
        f"PBO={summary['sharpe_pbo']} 不在 [0,1]"
    )


def test_report_to_dict_completeness():
    """CPCVReport.to_dict() 包含所有必要欄位"""
    cfg = CPCVConfig(
        n_groups=4,
        k_test_groups=1,
        min_train_samples=30,
    )
    runner = CPCVRunner(cfg)
    df = make_ohlcv(600)

    report = runner.run(df, symbol="DICT_TEST", verbose=False, run_backtrader=False)
    d = report.to_dict()

    top_level_keys = [
        "symbol", "run_timestamp", "config",
        "n_total_folds", "n_active_folds", "n_skipped_folds",
        "n_backtest_paths", "folds", "paths", "summary",
    ]
    for key in top_level_keys:
        assert key in d, f"to_dict() 缺少欄位：{key}"

    # config 欄位
    cfg_dict = d["config"]
    for key in ["n_groups", "k_test_groups", "n_combinations", "n_backtest_paths"]:
        assert key in cfg_dict, f"config 缺少欄位：{key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
