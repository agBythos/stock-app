"""
validation package — Walk-Forward + CPCV Validation Framework
=============================================================

模組說明：
  purged_walk_forward  : SplitEngine + PurgeEngine，生成無洩漏的 fold（Phase 3.5）
  ensemble_trainer     : 多模型 ensemble 訓練/評估 adapter（Phase 3.5）
  backtrader_bridge    : TWAECommission + Backtrader 整合介面（Phase 3.5）
  run_walk_forward     : PurgedWalkForwardRunner orchestrator（Phase 3.5）
  cpcv_splitter        : GroupEngine + CPCVPurgeEngine + CPCVSplitter（Phase 4）
  cpcv_runner          : CPCVRunner + CPCVReport orchestrator（Phase 4）

Phase 3.5 快速使用::

    from validation.run_walk_forward import PurgedWalkForwardRunner
    from config.walk_forward import WalkForwardConfig

    cfg = WalkForwardConfig()
    runner = PurgedWalkForwardRunner(cfg)
    result = runner.run(df, symbol="2330.TW")

Phase 4 CPCV 快速使用::

    from validation.cpcv_runner import CPCVRunner
    from config.cpcv import CPCVConfig

    cfg = CPCVConfig(n_groups=6, k_test_groups=2)
    runner = CPCVRunner(cfg)
    result = runner.run(df, symbol="2330.TW")
    # result.paths: 5 條獨立 backtest paths
    # result.summary_sharpe_mean / std / ci_95: Sharpe 分佈統計
"""

from validation.purged_walk_forward import (
    WalkForwardFold,
    PurgedWalkForwardSplitter,
)
from validation.run_walk_forward import PurgedWalkForwardRunner
from validation.cpcv_splitter import (
    CPCVFold,
    CPCVPath,
    CPCVSplitter,
)
from validation.permutation import permute_bars
from validation.permutation_test import insample_perm_test, walkforward_perm_test
from validation.cpcv_runner import (
    CPCVFoldReport,
    CPCVPathReport,
    CPCVReport,
    CPCVRunner,
)

__all__ = [
    # Phase 3.5
    "WalkForwardFold",
    "PurgedWalkForwardSplitter",
    "PurgedWalkForwardRunner",
    # Phase 4 CPCV
    "CPCVFold",
    "CPCVPath",
    "CPCVSplitter",
    "CPCVFoldReport",
    "CPCVPathReport",
    "CPCVReport",
    "CPCVRunner",
    # Phase 4.5 MCPT
    "permute_bars",
    "insample_perm_test",
    "walkforward_perm_test",
]
