"""
stock-app/hmm/market_hmm.py

MarketHMM: GaussianHMM-based market regime detector.

States (after standardisation):
  0 → Bull    (highest mean log_return)
  1 → Sideways
  2 → Bear    (lowest mean log_return)

Features (3-dim observation vector):
  - log_return      : log(close_t / close_{t-1})
  - volatility_20d  : 20-day rolling std of log_return * sqrt(252)
  - volume_ratio    : volume / 20-day MA of volume

Anti-leakage contract:
  fit()  only accepts training-period data (caller's responsibility to slice).
  The StandardScaler is fitted exclusively on the training observations.
  predict() / predict_proba() only transform — never re-fit the scaler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# ── Public labels (after state reordering by mean log_return) ──────────────
_STATE_LABELS = {0: "Bull", 1: "Sideways", 2: "Bear"}

# ── Required columns in the input DataFrame ───────────────────────────────
_REQUIRED_COLS = {"close", "volume"}


class MarketHMM:
    """
    Unsupervised HMM market-regime detector wrapping hmmlearn GaussianHMM.

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 3: Bull / Sideways / Bear).
    n_iter : int
        Maximum EM iterations for Baum-Welch (default 100).
    n_init : int
        Number of random initialisations; best log-likelihood is kept (default 5).
    covariance_type : str
        HMM covariance type (default "diag" — more stable than "full").
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        n_init: int = 5,
        covariance_type: str = "diag",
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.random_state = random_state

        self._model: GaussianHMM | None = None
        self._scaler: StandardScaler | None = None
        # Mapping from raw HMM state index → standardised label index
        self._state_map: dict[int, int] | None = None
        self._is_fitted: bool = False

    # ── Feature Engineering ───────────────────────────────────────────────

    @staticmethod
    def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the 3-dim HMM observation features from OHLCV data.

        The *caller* ensures df contains only the desired (training or
        inference) period — no lookahead is introduced here.

        Returns a DataFrame with columns:
            log_return, volatility_20d, volume_ratio
        indexed to the valid (non-NaN) rows.
        """
        df = df.copy()
        required = {"close", "volume"}
        missing = required - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        # Normalise column names to lower-case for robustness
        df.columns = df.columns.str.lower()

        log_ret = np.log(df["close"] / df["close"].shift(1))
        vol_20d = log_ret.rolling(20).std() * np.sqrt(252)
        vol_ratio = df["volume"] / df["volume"].rolling(20).mean()

        feat = pd.DataFrame(
            {
                "log_return": log_ret,
                "volatility_20d": vol_20d,
                "volume_ratio": vol_ratio,
            },
            index=df.index,
        )
        return feat.dropna()

    # ── Fitting ───────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "MarketHMM":
        """
        Train the HMM on training-period data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data for the *training period only*.  Must contain
            columns 'close' and 'volume' (case-insensitive).
            Minimum recommended length: 252 rows (1 trading year).

        Returns
        -------
        self
        """
        feat = self._compute_features(df)
        if len(feat) < 30:
            raise ValueError(
                f"Too few observations after feature engineering: {len(feat)}. "
                "Need at least 30 rows."
            )

        # Fit scaler on training data only
        self._scaler = StandardScaler()
        obs = self._scaler.fit_transform(feat.values)

        # Multiple random initialisations; keep best log-likelihood
        best_model = None
        best_score = -np.inf

        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 10_000, size=self.n_init)

        for seed in seeds:
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=int(seed),
            )
            model.fit(obs)
            try:
                score = model.score(obs)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_model = model

        if best_model is None:
            raise RuntimeError("HMM fitting failed for all initialisations.")

        self._model = best_model

        # ── State standardisation ─────────────────────────────────────────
        # Sort raw states by mean log_return (index 0 in feature matrix),
        # descending: highest return → Bull (label 0),
        #             middle        → Sideways (label 1),
        #             lowest        → Bear (label 2).
        mean_returns = self._model.means_[:, 0]  # log_return mean per state
        sorted_raw = np.argsort(mean_returns)[::-1]  # high → low

        # sorted_raw[i] = raw state index that should map to standardised label i
        self._state_map = {int(raw): i for i, raw in enumerate(sorted_raw)}

        self._is_fitted = True
        return self

    # ── Internal helpers ──────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("MarketHMM must be fitted before prediction. Call fit() first.")

    def _transform_obs(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
        """
        Compute and scale features for df; return (scaled_obs, index).
        """
        feat = self._compute_features(df)
        obs = self._scaler.transform(feat.values)
        return obs, feat.index

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Return standardised state sequence (0=Bull, 1=Sideways, 2=Bear)
        using Viterbi decoding.

        Intended for offline backtesting label generation.
        For live/forward-only inference, use predict_proba().

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data (can be test period; scaler is NOT re-fitted).

        Returns
        -------
        pd.Series[int]  — index aligned to df after NaN drop
        """
        self._check_fitted()
        obs, idx = self._transform_obs(df)
        raw_states = self._model.predict(obs)
        std_states = np.array([self._state_map[s] for s in raw_states])
        return pd.Series(std_states, index=idx, name="hmm_state", dtype=int)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return per-timestep posterior probabilities for each standardised state
        using the forward-backward algorithm (score_samples).

        This is the *forward-safe* method recommended for live inference to
        avoid lookahead leakage (Viterbi uses the full sequence).

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data.

        Returns
        -------
        pd.DataFrame  — columns: ['Bull', 'Sideways', 'Bear'],
                        index aligned to df after NaN drop.
        """
        self._check_fitted()
        obs, idx = self._transform_obs(df)
        _, posteriors = self._model.score_samples(obs)

        # Re-order columns from raw state indices to standardised labels
        n = self.n_states
        reordered = np.zeros_like(posteriors)
        for raw, std in self._state_map.items():
            reordered[:, std] = posteriors[:, raw]

        cols = [_STATE_LABELS[i] for i in range(n)]
        return pd.DataFrame(reordered, index=idx, columns=cols)

    # ── Label Helper ─────────────────────────────────────────────────────

    def state_label(self, state_idx: int) -> str:
        """
        Return human-readable label for a standardised state index.

        Parameters
        ----------
        state_idx : int
            Standardised state index (0=Bull, 1=Sideways, 2=Bear).

        Returns
        -------
        str  e.g. "Bull", "Sideways", "Bear"

        Raises
        ------
        ValueError if state_idx is out of range.
        """
        if state_idx not in _STATE_LABELS:
            raise ValueError(
                f"state_idx={state_idx} is out of range [0, {self.n_states - 1}]."
            )
        return _STATE_LABELS[state_idx]

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_features(self) -> int:
        return 3  # log_return, volatility_20d, volume_ratio

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"MarketHMM(n_states={self.n_states}, "
            f"covariance_type='{self.covariance_type}', "
            f"n_iter={self.n_iter}, status={status})"
        )
