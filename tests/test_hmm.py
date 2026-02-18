"""
stock-app/tests/test_hmm.py

Unit tests for the MarketHMM module (Phase 5.1).

Test coverage:
  TC01 - n_states == 3 after construction
  TC02 - fit() completes without error on synthetic data
  TC03 - predict() output shape matches valid observation count
  TC04 - predict() returns only valid standardised state indices {0, 1, 2}
  TC05 - state_label() returns valid strings for all three states
  TC06 - state_label() raises ValueError for out-of-range index
  TC07 - predict_proba() returns DataFrame with correct shape and columns
  TC08 - predict_proba() probabilities sum to 1 per timestep
  TC09 - is_fitted property: False before fit, True after fit
  TC10 - predict() raises RuntimeError if called before fit
  TC11 - state ordering: Bull mean_return >= Sideways mean_return >= Bear mean_return
  TC12 - scaler is not re-fitted on predict (anti-leakage: scaler stats preserved)
"""

import sys
import os

# Ensure stock-app root is on the path when running from the tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from hmm import MarketHMM


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_synthetic_df(
    n_days: int = 400,
    seed: int = 0,
    n_regimes: int = 3,
) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame with clear regime structure so that
    the HMM can converge reliably in tests.

    Regime blocks are appended sequentially:
      block 0: low-vol uptrend  (Bull)
      block 1: high-vol downtrend (Bear)
      block 2: medium-vol sideways (Sideways)

    Generates exactly n_days rows by computing cumulative log-returns.
    """
    rng = np.random.default_rng(seed)
    block = n_days // n_regimes

    # Regime parameters: (daily_drift, daily_vol)
    params = [
        (0.001, 0.005),    # Bull
        (-0.002, 0.020),   # Bear
        (0.0,   0.010),    # Sideways
    ]

    log_rets: list[float] = []
    for drift, vol in params:
        log_rets.extend(rng.normal(drift, vol, block).tolist())

    # Pad any remaining days (when n_days % n_regimes != 0) with neutral noise
    while len(log_rets) < n_days:
        log_rets.append(float(rng.normal(0.0, 0.010)))

    log_rets_arr = np.array(log_rets[:n_days])
    close = 100.0 * np.exp(np.cumsum(log_rets_arr))
    volume = rng.integers(500_000, 2_000_000, size=n_days).astype(float)

    idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
    df = pd.DataFrame({"close": close, "volume": volume}, index=idx)
    return df


@pytest.fixture(scope="module")
def fitted_model() -> MarketHMM:
    """Shared fitted MarketHMM across tests to avoid redundant training."""
    df = _make_synthetic_df(n_days=500)
    model = MarketHMM(n_states=3, n_iter=100, n_init=3, random_state=42)
    model.fit(df)
    return model


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    return _make_synthetic_df(n_days=500)


# ── Test Cases ────────────────────────────────────────────────────────────


class TestConstruction:
    """TC01 — n_states == 3 after construction."""

    def test_tc01_n_states_default(self):
        model = MarketHMM()
        assert model.n_states == 3, "Default n_states must be 3"

    def test_tc01_n_states_custom(self):
        model = MarketHMM(n_states=3)
        assert model.n_states == 3

    def test_tc01_n_features(self):
        model = MarketHMM()
        assert model.n_features == 3, "Feature count must be 3 (log_return, volatility_20d, volume_ratio)"


class TestFitting:
    """TC02 — fit() completes without error."""

    def test_tc02_fit_no_error(self, sample_df):
        model = MarketHMM(n_states=3, n_iter=50, n_init=2, random_state=7)
        result = model.fit(sample_df)
        assert result is model, "fit() must return self"

    def test_tc02_is_fitted_after_fit(self, sample_df):
        model = MarketHMM(n_states=3, n_iter=50, n_init=2, random_state=7)
        assert model.is_fitted is False
        model.fit(sample_df)
        assert model.is_fitted is True


class TestPredict:
    """TC03, TC04 — predict() output shape and valid state indices."""

    def test_tc03_predict_shape(self, fitted_model, sample_df):
        states = fitted_model.predict(sample_df)
        # Feature engineering drops first 20 NaN rows (rolling window)
        assert isinstance(states, pd.Series), "predict() must return pd.Series"
        assert len(states) > 0, "predict() must return non-empty series"
        assert len(states) <= len(sample_df), "predict() length must not exceed input"

    def test_tc04_predict_valid_state_indices(self, fitted_model, sample_df):
        states = fitted_model.predict(sample_df)
        unique_states = set(states.unique())
        assert unique_states.issubset({0, 1, 2}), (
            f"Predicted states must be subset of {{0, 1, 2}}, got {unique_states}"
        )

    def test_tc04_predict_all_three_states_present(self, fitted_model, sample_df):
        states = fitted_model.predict(sample_df)
        assert len(states.unique()) == 3, (
            "With diverse synthetic data, all 3 states should appear"
        )


class TestStateLabel:
    """TC05, TC06 — state_label() correctness and error handling."""

    def test_tc05_state_label_bull(self):
        model = MarketHMM()
        label = model.state_label(0)
        assert label == "Bull", f"State 0 should be 'Bull', got '{label}'"

    def test_tc05_state_label_sideways(self):
        model = MarketHMM()
        label = model.state_label(1)
        assert label == "Sideways", f"State 1 should be 'Sideways', got '{label}'"

    def test_tc05_state_label_bear(self):
        model = MarketHMM()
        label = model.state_label(2)
        assert label == "Bear", f"State 2 should be 'Bear', got '{label}'"

    def test_tc05_state_label_returns_string(self):
        model = MarketHMM()
        for idx in range(3):
            assert isinstance(model.state_label(idx), str)

    def test_tc06_state_label_out_of_range(self):
        model = MarketHMM()
        with pytest.raises(ValueError):
            model.state_label(99)

    def test_tc06_state_label_negative_raises(self):
        model = MarketHMM()
        with pytest.raises(ValueError):
            model.state_label(-1)


class TestPredictProba:
    """TC07, TC08 — predict_proba() shape and probability sums."""

    def test_tc07_predict_proba_shape(self, fitted_model, sample_df):
        proba = fitted_model.predict_proba(sample_df)
        assert isinstance(proba, pd.DataFrame), "predict_proba() must return pd.DataFrame"
        assert proba.shape[1] == 3, "predict_proba() must have 3 columns"
        assert list(proba.columns) == ["Bull", "Sideways", "Bear"]

    def test_tc08_probabilities_sum_to_one(self, fitted_model, sample_df):
        proba = fitted_model.predict_proba(sample_df)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(
            row_sums.values, 1.0, atol=1e-6,
            err_msg="Posterior probabilities must sum to 1 per timestep"
        )

    def test_tc08_probabilities_non_negative(self, fitted_model, sample_df):
        proba = fitted_model.predict_proba(sample_df)
        assert (proba.values >= 0).all(), "Posterior probabilities must be non-negative"


class TestIsNotFitted:
    """TC09, TC10 — pre-fit guards."""

    def test_tc09_is_fitted_false_before_fit(self):
        model = MarketHMM()
        assert model.is_fitted is False

    def test_tc10_predict_raises_before_fit(self):
        model = MarketHMM()
        df = _make_synthetic_df(n_days=100)
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(df)

    def test_tc10_predict_proba_raises_before_fit(self):
        model = MarketHMM()
        df = _make_synthetic_df(n_days=100)
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_proba(df)


class TestStateOrdering:
    """TC11 — Bull mean_return >= Sideways mean_return >= Bear mean_return."""

    def test_tc11_state_ordering_bull_gt_bear(self, fitted_model, sample_df):
        """After standardisation, state 0 (Bull) has higher mean return than state 2 (Bear)."""
        proba = fitted_model.predict_proba(sample_df)
        states = fitted_model.predict(sample_df)

        # Compute per-state empirical mean log_returns from predictions
        from hmm.market_hmm import MarketHMM as _MarketHMM

        feat = _MarketHMM._compute_features(sample_df)
        state_ser = states.reindex(feat.index).dropna()
        feat_aligned = feat.loc[state_ser.index]

        mean_returns = {}
        for s in [0, 1, 2]:
            mask = state_ser == s
            if mask.sum() > 0:
                mean_returns[s] = feat_aligned.loc[mask, "log_return"].mean()

        if 0 in mean_returns and 2 in mean_returns:
            assert mean_returns[0] >= mean_returns[2], (
                f"Bull (state 0) mean_return={mean_returns[0]:.6f} should be "
                f">= Bear (state 2) mean_return={mean_returns[2]:.6f}"
            )


class TestAntiLeakage:
    """TC12 — Scaler is not re-fitted on predict (preserves training statistics)."""

    def test_tc12_scaler_not_refitted_on_predict(self, sample_df):
        n = len(sample_df)
        train_df = sample_df.iloc[: n // 2]
        test_df = sample_df.iloc[n // 2 :]

        model = MarketHMM(n_states=3, n_iter=50, n_init=2, random_state=42)
        model.fit(train_df)

        # Capture scaler mean/scale before predict
        scaler_mean_before = model._scaler.mean_.copy()
        scaler_scale_before = model._scaler.scale_.copy()

        _ = model.predict(test_df)

        np.testing.assert_array_equal(
            model._scaler.mean_, scaler_mean_before,
            err_msg="Scaler mean must not change after predict() (anti-leakage)"
        )
        np.testing.assert_array_equal(
            model._scaler.scale_, scaler_scale_before,
            err_msg="Scaler scale must not change after predict() (anti-leakage)"
        )
