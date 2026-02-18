"""
permutation — Bar Permutation Algorithm for Monte Carlo Validation
==================================================================

Based on neurotrader888/mcpt (MIT License).

Key idea: decompose OHLC bars into relative log-space components,
shuffle intrabar movements and gaps separately, then reconstruct.

Preserves: mean, std, skew, kurtosis of returns
Destroys: temporal structure (trends, momentum, mean-reversion)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union


def permute_bars(
    ohlc_df: Union[pd.DataFrame, List[pd.DataFrame]],
    start_index: int = 0,
    seed: Optional[int] = None,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """Generate a permuted version of OHLC data.

    Decomposes bars into log-space relative values:
      - gaps: open[i] - close[i-1]  (shuffled independently)
      - intrabar: high-open, low-open, close-open (shuffled together)

    Parameters
    ----------
    ohlc_df : DataFrame or list of DataFrames
        Must have columns ['open', 'high', 'low', 'close'].
        If list, all must share the same index (multi-market).
    start_index : int
        Bars before start_index are copied unchanged.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    DataFrame or list of DataFrames (matches input type).
    """
    assert start_index >= 0

    rng = np.random.RandomState(seed)

    multi = isinstance(ohlc_df, list)
    if not multi:
        ohlc_df = [ohlc_df]

    n_markets = len(ohlc_df)
    n_bars = len(ohlc_df[0])
    time_index = ohlc_df[0].index

    perm_start = start_index + 1
    perm_n = n_bars - perm_start

    if perm_n <= 0:
        return ohlc_df if multi else ohlc_df[0]

    # Extract relative components in log space
    start_bar = np.empty((n_markets, 4))
    rel_open = np.empty((n_markets, perm_n))
    rel_high = np.empty((n_markets, perm_n))
    rel_low = np.empty((n_markets, perm_n))
    rel_close = np.empty((n_markets, perm_n))

    for mi, df in enumerate(ohlc_df):
        log_bars = np.log(df[['open', 'high', 'low', 'close']].values.astype(np.float64))
        start_bar[mi] = log_bars[start_index]

        # Gaps: open relative to previous close
        r_o = log_bars[1:, 0] - log_bars[:-1, 3]
        # Intrabar: H/L/C relative to O
        r_h = log_bars[:, 1] - log_bars[:, 0]
        r_l = log_bars[:, 2] - log_bars[:, 0]
        r_c = log_bars[:, 3] - log_bars[:, 0]

        rel_open[mi] = r_o[start_index:]
        rel_high[mi] = r_h[perm_start:]
        rel_low[mi] = r_l[perm_start:]
        rel_close[mi] = r_c[perm_start:]

    # Shuffle intrabar together (preserves H/L/C relationship within bar)
    idx = np.arange(perm_n)
    perm_intra = rng.permutation(idx)
    rel_high = rel_high[:, perm_intra]
    rel_low = rel_low[:, perm_intra]
    rel_close = rel_close[:, perm_intra]

    # Shuffle gaps independently
    perm_gap = rng.permutation(idx)
    rel_open = rel_open[:, perm_gap]

    # Reconstruct
    results = []
    for mi, df in enumerate(ohlc_df):
        log_orig = np.log(df[['open', 'high', 'low', 'close']].values.astype(np.float64))
        perm = np.zeros((n_bars, 4))
        perm[:start_index] = log_orig[:start_index]
        perm[start_index] = start_bar[mi]

        for i in range(perm_start, n_bars):
            k = i - perm_start
            perm[i, 0] = perm[i - 1, 3] + rel_open[mi, k]
            perm[i, 1] = perm[i, 0] + rel_high[mi, k]
            perm[i, 2] = perm[i, 0] + rel_low[mi, k]
            perm[i, 3] = perm[i, 0] + rel_close[mi, k]

        perm = np.exp(perm)
        results.append(pd.DataFrame(perm, index=time_index, columns=['open', 'high', 'low', 'close']))

    return results if multi else results[0]
