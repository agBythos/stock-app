"""
Tests for Garman-Klass volatility feature integration.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.rf_strategy import _calc_garman_klass_vol, RandomForestPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.randn(n) * 0.3
    volume = rng.randint(1000, 10000, n).astype(float)
    dates = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Tests for _calc_garman_klass_vol
# ---------------------------------------------------------------------------

class TestCalcGarmanKlassVol:
    def test_output_length(self):
        df = _make_ohlcv(50)
        result = _calc_garman_klass_vol(df, window=20)
        assert len(result) == len(df)

    def test_nan_count(self):
        """First (window-1) values should be NaN."""
        df = _make_ohlcv(50)
        result = _calc_garman_klass_vol(df, window=20)
        assert result.isna().sum() == 19

    def test_positive_values(self):
        """Non-NaN GK volatility should be > 0."""
        df = _make_ohlcv(50)
        result = _calc_garman_klass_vol(df, window=20)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_manual_computation(self):
        """Verify against manual calculation for a small window."""
        df = _make_ohlcv(10)
        window = 5
        result = _calc_garman_klass_vol(df, window=window)

        # Manual calc for the last value (index 9, window covers 5..9)
        chunk = df.iloc[5:10]
        log_hl = np.log(chunk["High"] / chunk["Low"])
        log_co = np.log(chunk["Close"] / chunk["Open"])
        expected = np.sqrt(
            (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2).mean()
        )
        np.testing.assert_allclose(result.iloc[-1], expected, rtol=1e-10)

    def test_constant_price(self):
        """When H=L=O=C, GK vol should be 0."""
        n = 25
        df = pd.DataFrame({
            "Open": [100.0] * n,
            "High": [100.0] * n,
            "Low": [100.0] * n,
            "Close": [100.0] * n,
            "Volume": [1000.0] * n,
        })
        result = _calc_garman_klass_vol(df, window=20)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Tests for integration in RandomForestPredictor
# ---------------------------------------------------------------------------

class TestGKInPredictor:
    def test_feature_list_includes_gk(self):
        assert "garman_klass_vol" in RandomForestPredictor.FEATURE_NAMES

    def test_calculate_features_has_gk_column(self):
        df = _make_ohlcv(200)
        predictor = RandomForestPredictor()
        features = predictor._calculate_features(df)
        assert "garman_klass_vol" in features.columns

    def test_train_with_gk(self):
        """Training should succeed with the new feature."""
        df = _make_ohlcv(250)
        predictor = RandomForestPredictor(forward_days=5)
        ok = predictor.train(df)
        assert ok is True
        assert predictor.is_trained
        assert "garman_klass_vol" in predictor.feature_importance()

    def test_predict_with_gk(self):
        """Prediction should work end-to-end with the new feature."""
        df = _make_ohlcv(250)
        predictor = RandomForestPredictor(forward_days=5)
        predictor.train(df)
        result = predictor.predict("TEST", df)
        assert result["signal"] in ("BUY", "SELL", "HOLD")
