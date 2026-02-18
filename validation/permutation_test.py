"""
permutation_test — Monte Carlo Permutation Test for Strategy Validation
=======================================================================

Two modes:
  - insample_perm_test:  Run strategy on real vs N permuted datasets.
  - walkforward_perm_test: Walk-forward variant with rolling train windows.

A low p-value (< 0.05) means the strategy likely exploits real market structure,
not just noise patterns.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional, Dict, Any

from validation.permutation import permute_bars


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _extract_metric(result: Any, metric: str) -> float:
    """Extract a numeric metric from strategy result.

    Supports: dict key, object attribute, or raw float.
    """
    if isinstance(result, (int, float, np.floating)):
        return float(result)
    if isinstance(result, dict):
        return float(result[metric])
    return float(getattr(result, metric))


# ---------------------------------------------------------------------------
# In-sample permutation test
# ---------------------------------------------------------------------------

def insample_perm_test(
    strategy_fn: Callable[[pd.DataFrame], Any],
    data: pd.DataFrame,
    n_perms: int = 1000,
    metric: str = 'profit_factor',
    seed: Optional[int] = None,
    start_index: int = 0,
) -> Dict[str, Any]:
    """Run in-sample Monte Carlo permutation test.

    Parameters
    ----------
    strategy_fn : callable
        Takes a DataFrame (OHLC) and returns a result from which
        ``metric`` can be extracted (dict, object attr, or float).
    data : DataFrame
        Real OHLC data with columns ['open','high','low','close'].
    n_perms : int
        Number of permutations.
    metric : str
        Which metric to compare (e.g. 'profit_factor', 'sharpe', 'total_return').
    seed : int or None
        Base random seed.
    start_index : int
        Passed to permute_bars.

    Returns
    -------
    dict with keys:
        p_value : float in [0, 1]
        real_metric : float
        perm_metrics : np.ndarray of length n_perms
        n_perms : int
    """
    # Real performance
    real_result = strategy_fn(data)
    real_metric = _extract_metric(real_result, metric)

    # Permuted performances
    perm_metrics = np.empty(n_perms)
    rng = np.random.RandomState(seed)

    for i in range(n_perms):
        perm_seed = rng.randint(0, 2**31)
        perm_data = permute_bars(data, start_index=start_index, seed=perm_seed)
        perm_result = strategy_fn(perm_data)
        perm_metrics[i] = _extract_metric(perm_result, metric)

    # p-value: fraction of permutations >= real metric
    p_value = float(np.mean(perm_metrics >= real_metric))

    return {
        'p_value': p_value,
        'real_metric': real_metric,
        'perm_metrics': perm_metrics,
        'n_perms': n_perms,
    }


# ---------------------------------------------------------------------------
# Walk-forward permutation test
# ---------------------------------------------------------------------------

def walkforward_perm_test(
    strategy_fn: Callable[[pd.DataFrame, pd.DataFrame], Any],
    data: pd.DataFrame,
    train_window: int,
    test_window: Optional[int] = None,
    n_perms: int = 200,
    metric: str = 'profit_factor',
    seed: Optional[int] = None,
    start_index: int = 0,
) -> Dict[str, Any]:
    """Walk-forward Monte Carlo permutation test.

    Splits data into rolling train/test windows. For each fold, trains on
    train data and evaluates on test data. Compares aggregated real metric
    against permuted versions.

    Parameters
    ----------
    strategy_fn : callable
        Takes (train_df, test_df) and returns a result from which
        ``metric`` can be extracted.
    data : DataFrame
        Full OHLC data.
    train_window : int
        Number of bars in each training window.
    test_window : int or None
        Number of bars in each test window. Defaults to train_window // 4.
    n_perms : int
        Number of permutations per fold.
    metric : str
        Metric name.
    seed : int or None
        Base random seed.
    start_index : int
        Passed to permute_bars.

    Returns
    -------
    dict with keys:
        p_value : float in [0, 1]
        real_metric : float (mean across folds)
        perm_metrics : np.ndarray of length n_perms (mean across folds)
        n_folds : int
    """
    if test_window is None:
        test_window = max(1, train_window // 4)

    n = len(data)
    folds = []
    pos = 0
    while pos + train_window + test_window <= n:
        train_end = pos + train_window
        test_end = train_end + test_window
        folds.append((pos, train_end, test_end))
        pos = train_end  # non-overlapping test windows

    if not folds:
        raise ValueError(
            f"Data length {n} too short for train_window={train_window} + test_window={test_window}"
        )

    rng = np.random.RandomState(seed)

    # Real fold metrics
    real_fold_metrics = []
    for start, train_end, test_end in folds:
        train_df = data.iloc[start:train_end]
        test_df = data.iloc[train_end:test_end]
        result = strategy_fn(train_df, test_df)
        real_fold_metrics.append(_extract_metric(result, metric))

    real_mean = float(np.mean(real_fold_metrics))

    # Permuted fold metrics
    perm_means = np.empty(n_perms)
    for p in range(n_perms):
        perm_seed = rng.randint(0, 2**31)
        perm_data = permute_bars(data, start_index=start_index, seed=perm_seed)
        fold_metrics = []
        for start, train_end, test_end in folds:
            train_df = perm_data.iloc[start:train_end]
            test_df = perm_data.iloc[train_end:test_end]
            result = strategy_fn(train_df, test_df)
            fold_metrics.append(_extract_metric(result, metric))
        perm_means[p] = np.mean(fold_metrics)

    p_value = float(np.mean(perm_means >= real_mean))

    return {
        'p_value': p_value,
        'real_metric': real_mean,
        'perm_metrics': perm_means,
        'n_folds': len(folds),
    }
