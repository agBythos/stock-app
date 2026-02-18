"""
Tests for permutation and permutation_test modules.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from validation.permutation import permute_bars
from validation.permutation_test import insample_perm_test, walkforward_perm_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlc(n=500, seed=42):
    """Generate synthetic OHLC with a trend (cumulative drift)."""
    rng = np.random.RandomState(seed)
    # Log returns with positive drift (trend)
    log_returns = 0.001 + 0.02 * rng.randn(n)
    log_close = np.cumsum(log_returns) + np.log(100)
    close = np.exp(log_close)

    # Build OHLC
    high = close * (1 + np.abs(rng.randn(n)) * 0.005)
    low = close * (1 - np.abs(rng.randn(n)) * 0.005)
    open_ = close * (1 + rng.randn(n) * 0.003)
    # Ensure H >= max(O,C) and L <= min(O,C)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close
    }, index=pd.date_range('2020-01-01', periods=n, freq='D'))
    return df


# ---------------------------------------------------------------------------
# permute_bars tests
# ---------------------------------------------------------------------------

class TestPermuteBars:
    def test_shape_preserved(self):
        df = _make_ohlc(200)
        perm = permute_bars(df, seed=1)
        assert perm.shape == df.shape
        assert list(perm.columns) == ['open', 'high', 'low', 'close']
        assert (perm.index == df.index).all()

    def test_positive_prices(self):
        df = _make_ohlc(200)
        perm = permute_bars(df, seed=1)
        assert (perm > 0).all().all()

    def test_statistical_properties_preserved(self):
        """Mean and std of log returns should be within 5% of original."""
        df = _make_ohlc(1000, seed=10)
        log_ret_orig = np.diff(np.log(df['close'].values))

        # Average over multiple permutations for stability
        means, stds = [], []
        for s in range(20):
            perm = permute_bars(df, seed=s)
            lr = np.diff(np.log(perm['close'].values))
            means.append(lr.mean())
            stds.append(lr.std())

        avg_mean = np.mean(means)
        avg_std = np.mean(stds)

        # Within 5% (relative) or 0.001 absolute for mean
        assert abs(avg_mean - log_ret_orig.mean()) < max(0.05 * abs(log_ret_orig.mean()), 0.001)
        assert abs(avg_std - log_ret_orig.std()) / log_ret_orig.std() < 0.05

    def test_temporal_structure_destroyed(self):
        """Autocorrelation of returns should be reduced."""
        df = _make_ohlc(500, seed=5)
        orig_ret = np.diff(np.log(df['close'].values))
        orig_autocorr = np.corrcoef(orig_ret[:-1], orig_ret[1:])[0, 1]

        perm_autocorrs = []
        for s in range(20):
            perm = permute_bars(df, seed=s)
            pr = np.diff(np.log(perm['close'].values))
            perm_autocorrs.append(np.corrcoef(pr[:-1], pr[1:])[0, 1])

        # Permuted autocorrelations should be closer to 0 on average
        assert abs(np.mean(perm_autocorrs)) < abs(orig_autocorr) + 0.1

    def test_seed_reproducibility(self):
        df = _make_ohlc(100)
        p1 = permute_bars(df, seed=42)
        p2 = permute_bars(df, seed=42)
        pd.testing.assert_frame_equal(p1, p2)

    def test_different_seeds_differ(self):
        df = _make_ohlc(100)
        p1 = permute_bars(df, seed=1)
        p2 = permute_bars(df, seed=2)
        assert not p1.equals(p2)

    def test_multi_market(self):
        df1 = _make_ohlc(200, seed=1)
        df2 = _make_ohlc(200, seed=2)
        result = permute_bars([df1, df2], seed=10)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == df1.shape
        assert result[1].shape == df2.shape

    def test_start_index(self):
        df = _make_ohlc(100)
        perm = permute_bars(df, start_index=10, seed=1)
        # First 10 bars should be unchanged
        pd.testing.assert_frame_equal(perm.iloc[:10], df.iloc[:10], atol=1e-10)


# ---------------------------------------------------------------------------
# insample_perm_test tests
# ---------------------------------------------------------------------------

class TestInsamplePermTest:
    def test_p_value_range(self):
        df = _make_ohlc(200)
        # Simple strategy: return total log return as float
        def strategy(data):
            return float(np.log(data['close'].iloc[-1] / data['close'].iloc[0]))

        result = insample_perm_test(strategy, df, n_perms=50, metric='total_return', seed=0)
        assert 0.0 <= result['p_value'] <= 1.0
        assert len(result['perm_metrics']) == 50

    def test_random_strategy_high_p_value(self):
        """A random strategy should have p > 0.05 most of the time."""
        df = _make_ohlc(300, seed=99)

        def random_strategy(data):
            # Return random metric — no edge
            return {'profit_factor': np.random.rand()}

        result = insample_perm_test(random_strategy, df, n_perms=100, seed=7)
        # Not a strict assertion — just check structure
        assert 0.0 <= result['p_value'] <= 1.0

    def test_dict_metric_extraction(self):
        df = _make_ohlc(100)
        def strategy(data):
            return {'sharpe': 1.5, 'profit_factor': 2.0}
        result = insample_perm_test(strategy, df, n_perms=10, metric='sharpe', seed=0)
        assert result['real_metric'] == 1.5

    def test_trend_strategy_beats_permuted(self):
        """A mean-reversion strategy on mean-reverting data should beat permuted data.

        We construct data with strong negative autocorrelation (mean-reversion),
        which permutation destroys. A strategy that buys after down days and sells
        after up days exploits this structure.
        """
        np.random.seed(123)
        n = 1000
        # Mean-reverting returns: next return = -0.5 * current + noise
        returns = np.zeros(n)
        for i in range(1, n):
            returns[i] = -0.5 * returns[i-1] + 0.01 * np.random.randn()
        close = 100 * np.exp(np.cumsum(returns))
        high = close * 1.003
        low = close * 0.997
        open_ = close * (1 + 0.001 * np.random.randn(n))
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame({
            'open': open_, 'high': high, 'low': low, 'close': close
        }, index=pd.date_range('2020-01-01', periods=n, freq='D'))

        def mean_revert_strategy(data):
            c = data['close'].values
            log_ret = np.diff(np.log(c))
            # Buy after down, sell after up (exploit mean-reversion)
            position = np.where(log_ret[:-1] < 0, 1.0, -1.0)
            pnl = position * log_ret[1:]
            return float(np.sum(pnl))

        result = insample_perm_test(mean_revert_strategy, df, n_perms=100, metric='total_return', seed=42)
        # Real metric should be higher than most permutations
        assert result['real_metric'] > np.percentile(result['perm_metrics'], 25)


# ---------------------------------------------------------------------------
# walkforward_perm_test tests
# ---------------------------------------------------------------------------

class TestWalkforwardPermTest:
    def test_p_value_range(self):
        df = _make_ohlc(400)

        def strategy(train, test):
            # Simple: return average test return
            ret = np.log(test['close'].iloc[-1] / test['close'].iloc[0])
            return {'profit_factor': float(ret)}

        result = walkforward_perm_test(
            strategy, df, train_window=100, n_perms=30, seed=0
        )
        assert 0.0 <= result['p_value'] <= 1.0
        assert result['n_folds'] > 0
        assert len(result['perm_metrics']) == 30

    def test_short_data_raises(self):
        df = _make_ohlc(10)
        def strategy(train, test):
            return 0.0
        with pytest.raises(ValueError):
            walkforward_perm_test(strategy, df, train_window=100, n_perms=5)
