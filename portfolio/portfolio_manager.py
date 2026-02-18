"""
PortfolioManager — Phase 7 Step 6
===================================

Multi-symbol portfolio manager that supports:
- Position management (add / remove)
- Weight rebalancing (equal-weight method, extensible)
- Current regime status per symbol (via MarketHMM)
- Portfolio-level backtesting (per-symbol BacktraderEngine runs, aggregated)
- Correlation matrix calculation

Strategy types supported:
  "rf"     — Random Forest ML Strategy
  "hmm_rf" — HMM-Filtered Random Forest Strategy

Author: Bythos (sub-agent)
Created: 2026-02-18
Phase: 7 Step 6
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional imports — graceful degradation if unavailable
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    yf = None  # type: ignore
    _YF_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── Strategy type constants ──────────────────────────────────────────────────
STRATEGY_RF = "rf"
STRATEGY_HMM_RF = "hmm_rf"
VALID_STRATEGY_TYPES = {STRATEGY_RF, STRATEGY_HMM_RF}

# ── Regime label map (MarketHMM convention) ──────────────────────────────────
REGIME_LABELS = {0: "Bull", 1: "Sideways", 2: "Bear", -1: "Unknown"}

TRADING_DAYS = 252


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Position:
    """Represents a single portfolio position."""
    symbol: str
    weight: float           # target weight in [0, 1]
    strategy_type: str      # "rf" | "hmm_rf"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "weight": round(self.weight, 6),
            "strategy_type": self.strategy_type,
        }


@dataclass
class PortfolioSummary:
    """Full portfolio summary including regime status and KPIs."""
    positions: List[Dict[str, Any]]
    total_weight: float
    n_positions: int
    regime_status: Dict[str, Any]       # symbol → regime info
    portfolio_kpi: Dict[str, Any]       # combined KPI estimates
    as_of: str                          # ISO datetime of summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of": self.as_of,
            "n_positions": self.n_positions,
            "total_weight": round(self.total_weight, 6),
            "positions": self.positions,
            "regime_status": self.regime_status,
            "portfolio_kpi": self.portfolio_kpi,
        }


@dataclass
class PortfolioBacktestResult:
    """Aggregated portfolio backtest result."""
    symbols: List[str]
    weights: List[float]
    start_date: str
    end_date: str
    initial_capital: float
    per_symbol_results: List[Dict[str, Any]]
    portfolio_performance: Dict[str, Any]   # weighted aggregate KPIs
    combined_equity_curve: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": self.symbols,
            "weights": [round(w, 6) for w in self.weights],
            "period": {
                "start": self.start_date,
                "end": self.end_date,
            },
            "initial_capital": self.initial_capital,
            "per_symbol_results": self.per_symbol_results,
            "portfolio_performance": self.portfolio_performance,
            "combined_equity_curve": self.combined_equity_curve,
        }


# ── Core class ───────────────────────────────────────────────────────────────

class PortfolioManager:
    """
    Multi-symbol portfolio manager.

    Usage::

        pm = PortfolioManager(initial_capital=1_000_000)
        pm.add_position("2330.TW", weight=0.5, strategy_type="hmm_rf")
        pm.add_position("0050.TW", weight=0.3, strategy_type="rf")
        pm.add_position("2317.TW", weight=0.2, strategy_type="rf")

        # Rebalance to equal weight
        pm.rebalance(method="equal_weight")

        # Get regime summary
        summary = pm.get_portfolio_summary()

        # Run backtest
        result = pm.run_portfolio_backtest("2023-01-01", "2024-01-01")

        # Correlation matrix
        corr = pm.calculate_correlation_matrix(lookback_days=252)
    """

    def __init__(self, initial_capital: float = 1_000_000.0):
        """
        Initialise PortfolioManager.

        Args:
            initial_capital: Total capital to allocate (default NTD 1,000,000)
        """
        self.initial_capital = initial_capital
        self._positions: Dict[str, Position] = {}   # symbol (upper) → Position

    # ── Position management ───────────────────────────────────────────────

    def add_position(
        self,
        symbol: str,
        weight: float,
        strategy_type: str = STRATEGY_RF,
    ) -> None:
        """
        Add or update a portfolio position.

        Args:
            symbol:        Ticker symbol (e.g. "2330.TW")
            weight:        Target portfolio weight (≥ 0). Need not sum to 1 —
                           call rebalance() to normalise.
            strategy_type: "rf" (Random Forest) | "hmm_rf" (HMM-Filtered RF)

        Raises:
            ValueError: Invalid strategy_type or negative weight
        """
        symbol = symbol.upper().strip()
        if strategy_type not in VALID_STRATEGY_TYPES:
            raise ValueError(
                f"Invalid strategy_type '{strategy_type}'. "
                f"Must be one of: {sorted(VALID_STRATEGY_TYPES)}"
            )
        if weight < 0:
            raise ValueError(f"Weight must be ≥ 0, got {weight}")

        self._positions[symbol] = Position(
            symbol=symbol,
            weight=float(weight),
            strategy_type=strategy_type,
        )

    def remove_position(self, symbol: str) -> None:
        """
        Remove a position from the portfolio.

        Args:
            symbol: Ticker symbol to remove

        Raises:
            KeyError: Symbol not in portfolio
        """
        symbol = symbol.upper().strip()
        if symbol not in self._positions:
            raise KeyError(f"Symbol '{symbol}' not in portfolio.")
        del self._positions[symbol]

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Return current positions as plain dicts."""
        return {sym: pos.to_dict() for sym, pos in self._positions.items()}

    # ── Rebalancing ───────────────────────────────────────────────────────

    def rebalance(self, method: str = "equal_weight", lookback_days: int = 252) -> Dict[str, float]:
        """
        Rebalance portfolio weights in-place.

        Args:
            method: Rebalancing method.
                    "equal_weight" — equal allocation to all positions (1/N each)
                    "risk_parity"  — inverse-volatility weighting; requires yfinance
            lookback_days: Historical window for volatility estimation (risk_parity only)

        Returns:
            Dict mapping symbol → new weight

        Raises:
            ValueError: No positions, unsupported method, or volatility data unavailable
        """
        if not self._positions:
            raise ValueError("No positions in portfolio. Add positions before rebalancing.")

        if method == "equal_weight":
            n = len(self._positions)
            equal_w = round(1.0 / n, 10)
            for pos in self._positions.values():
                pos.weight = equal_w

        elif method == "risk_parity":
            new_weights = self._compute_risk_parity_weights(lookback_days=lookback_days)
            for sym, w in new_weights.items():
                self._positions[sym].weight = w

        else:
            raise ValueError(
                f"Unsupported rebalance method: '{method}'. "
                f"Supported: ['equal_weight', 'risk_parity']"
            )

        return {sym: pos.weight for sym, pos in self._positions.items()}

    def _compute_risk_parity_weights(self, lookback_days: int = 252) -> Dict[str, float]:
        """
        Compute inverse-volatility (Risk Parity) weights.

        Each symbol's weight is proportional to 1/σ where σ is the annualised
        volatility of daily log-returns over the lookback window.

        Args:
            lookback_days: Number of trading days to estimate volatility

        Returns:
            Dict mapping symbol → weight (sum = 1.0)

        Raises:
            ValueError: If volatility data cannot be fetched for any symbol,
                        or all volatilities are zero
        """
        if not _YF_AVAILABLE:
            raise ValueError("yfinance not available; cannot compute risk_parity weights.")

        symbols = list(self._positions.keys())
        end_dt = datetime.now()
        # Add buffer days to account for weekends/holidays
        start_dt = end_dt - timedelta(days=int(lookback_days * 1.5))

        vols: Dict[str, float] = {}
        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(start=start_dt.strftime("%Y-%m-%d"),
                                      end=end_dt.strftime("%Y-%m-%d"))
                if hist.empty or len(hist) < 20:
                    raise ValueError(f"Insufficient data for {sym}: {len(hist)} bars")
                close = hist["Close"].dropna()
                log_returns = np.log(close / close.shift(1)).dropna()
                # Annualised volatility
                sigma = float(log_returns.std() * np.sqrt(TRADING_DAYS))
                if sigma <= 0:
                    raise ValueError(f"Zero/negative volatility for {sym}")
                vols[sym] = sigma
            except Exception as exc:
                raise ValueError(
                    f"Failed to compute volatility for {sym}: {exc}"
                ) from exc

        # Inverse-volatility weights: w_i = (1/σ_i) / Σ(1/σ_j)
        inv_vols = {sym: 1.0 / sigma for sym, sigma in vols.items()}
        total_inv = sum(inv_vols.values())
        if total_inv == 0:
            raise ValueError("Sum of inverse volatilities is zero; cannot normalise.")

        return {sym: inv_v / total_inv for sym, inv_v in inv_vols.items()}

    # ── Portfolio Summary ─────────────────────────────────────────────────

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive portfolio summary including:
        - Per-position details (weight, strategy_type)
        - Latest market regime for each symbol (via MarketHMM)
        - Portfolio-level KPI estimates (based on recent 6-month data)

        Returns:
            Dict with keys: as_of, n_positions, total_weight,
                            positions, regime_status, portfolio_kpi

        Notes:
            - If network or HMM is unavailable, regime defaults to "Unknown"
            - KPI estimates are buy-and-hold (no strategy simulation)
        """
        as_of = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        positions_list = [pos.to_dict() for pos in self._positions.values()]
        total_w = sum(pos.weight for pos in self._positions.values())

        regime_status: Dict[str, Any] = {}
        price_data: Dict[str, pd.Series] = {}

        for sym, pos in self._positions.items():
            regime_info = self._get_regime_for_symbol(sym)
            regime_status[sym] = regime_info

            # Collect price series for KPI calculation
            if regime_info.get("prices") is not None:
                price_data[sym] = regime_info.pop("prices")
            else:
                regime_info.pop("prices", None)

        # Compute portfolio KPI from buy-and-hold returns (last 6 months)
        portfolio_kpi = self._compute_portfolio_kpi(price_data, self._positions)

        summary = PortfolioSummary(
            positions=positions_list,
            total_weight=total_w,
            n_positions=len(self._positions),
            regime_status=regime_status,
            portfolio_kpi=portfolio_kpi,
            as_of=as_of,
        )
        return summary.to_dict()

    def _get_regime_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Fetch latest HMM regime for a single symbol. Gracefully degrades."""
        result: Dict[str, Any] = {
            "symbol": symbol,
            "regime_idx": -1,
            "regime_label": "Unknown",
            "regime_proba": None,
            "data_bars": 0,
            "prices": None,
            "error": None,
        }

        try:
            if not _YF_AVAILABLE or yf is None:
                result["error"] = "yfinance not available"
                return result
            ticker = yf.Ticker(symbol)
            raw = ticker.history(period="6mo")

            if raw.empty or len(raw) < 30:
                result["error"] = f"Insufficient data ({len(raw)} bars)"
                return result

            result["data_bars"] = len(raw)
            result["prices"] = raw["Close"]

            # Try HMM regime prediction
            try:
                from hmm.market_hmm import MarketHMM
                hmm = MarketHMM(n_states=3, n_iter=50)
                hmm.fit(raw)
                regimes = hmm.predict(raw)
                latest_regime = int(regimes[-1])
                result["regime_idx"] = latest_regime
                result["regime_label"] = REGIME_LABELS.get(latest_regime, "Unknown")

                # Try to get probability
                try:
                    proba = hmm.predict_proba(raw)
                    if proba is not None and len(proba) > 0:
                        last_proba = proba[-1].tolist()
                        result["regime_proba"] = [round(p, 4) for p in last_proba]
                except Exception:
                    pass

            except Exception as hmm_err:
                result["error"] = f"HMM failed: {hmm_err}"

        except Exception as e:
            result["error"] = str(e)

        return result

    def _compute_portfolio_kpi(
        self,
        price_data: Dict[str, pd.Series],
        positions: Dict[str, "Position"],
    ) -> Dict[str, Any]:
        """Compute buy-and-hold portfolio KPIs from price series."""
        if not price_data:
            return {
                "estimated_6m_return_pct": None,
                "estimated_volatility_pct": None,
                "estimated_sharpe": None,
                "note": "No price data available",
            }

        try:
            # Align all price series
            combined = pd.DataFrame(price_data).dropna()
            if len(combined) < 5:
                return {"note": "Insufficient overlapping data"}

            # Compute returns
            returns = combined.pct_change().dropna()

            # Build weight array
            symbols_with_data = list(combined.columns)
            weights = np.array([
                positions[sym].weight if sym in positions else 0.0
                for sym in symbols_with_data
            ])
            total_w = weights.sum()
            if total_w > 0:
                weights = weights / total_w  # normalise

            # Portfolio return series
            port_ret = (returns * weights).sum(axis=1)

            # KPIs
            n = len(port_ret)
            total_return = float((combined.iloc[-1] / combined.iloc[0] - 1).dot(weights) * 100)
            ann_vol = float(port_ret.std() * np.sqrt(TRADING_DAYS) * 100)
            rf_daily = (1 + 0.045) ** (1 / TRADING_DAYS) - 1
            excess = port_ret - rf_daily
            sharpe = float(excess.mean() / port_ret.std() * np.sqrt(TRADING_DAYS)) if port_ret.std() > 0 else 0.0

            return {
                "estimated_6m_return_pct": round(total_return, 4),
                "estimated_annualized_volatility_pct": round(ann_vol, 4),
                "estimated_sharpe": round(sharpe, 4),
                "data_bars": n,
                "note": "Buy-and-hold estimate (6mo, no strategy simulation)",
            }
        except Exception as e:
            return {"note": f"KPI calculation failed: {e}"}

    # ── Portfolio Backtest ────────────────────────────────────────────────

    def run_portfolio_backtest(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """
        Run individual backtests for each position and aggregate results.

        Each symbol is backtested with its assigned strategy_type:
        - "rf"     → RFStrategy
        - "hmm_rf" → HMMFilterStrategy

        The portfolio equity curve is computed as a weighted combination
        of per-symbol equity curves (resampled to common dates).

        Args:
            start_date: ISO date string, e.g. "2023-01-01"
            end_date:   ISO date string, e.g. "2024-01-01"

        Returns:
            PortfolioBacktestResult.to_dict()

        Raises:
            ValueError: No positions or invalid dates
        """
        if not self._positions:
            raise ValueError("No positions in portfolio.")

        try:
            from backtest.backtrader_engine import BacktraderEngine
            from backtest.rf_strategy import RFStrategy
            from backtest.hmm_filter_strategy import HMMFilterStrategy
        except ImportError as e:
            raise RuntimeError(f"Backtest modules unavailable: {e}")

        if not _YF_AVAILABLE or yf is None:
            raise RuntimeError("yfinance not available for portfolio backtest")

        strategy_map = {
            STRATEGY_RF: (RFStrategy, "Random Forest"),
            STRATEGY_HMM_RF: (HMMFilterStrategy, "HMM-Filtered RF"),
        }

        symbols = list(self._positions.keys())
        weights = np.array([self._positions[sym].weight for sym in symbols])
        total_w = weights.sum()
        if total_w > 0:
            weights = weights / total_w

        per_symbol_results: List[Dict[str, Any]] = []
        equity_curves: Dict[str, pd.Series] = {}

        for sym, w in zip(symbols, weights):
            pos = self._positions[sym]
            strategy_class, strategy_name = strategy_map.get(
                pos.strategy_type, (RFStrategy, "Random Forest")
            )

            try:
                ticker = yf.Ticker(sym)
                raw_df = ticker.history(start=start_date, end=end_date)

                if raw_df.empty or len(raw_df) < 60:
                    per_symbol_results.append({
                        "symbol": sym,
                        "strategy_type": pos.strategy_type,
                        "weight": round(float(w), 6),
                        "status": "insufficient_data",
                        "bars": len(raw_df),
                        "error": f"Only {len(raw_df)} bars (need ≥ 60)",
                    })
                    continue

                df = raw_df.copy()
                df.columns = [c.lower() for c in df.columns]

                capital_alloc = self.initial_capital * float(w)
                engine = BacktraderEngine(
                    symbol=sym,
                    initial_capital=max(capital_alloc, 10_000),
                )

                result = engine.run(
                    strategy_class=strategy_class,
                    data=df,
                    strategy_params={},
                    strategy_name=strategy_name,
                )

                # Extract equity curve as Series
                if result.equity_curve:
                    eq = pd.Series(
                        [e["value"] for e in result.equity_curve],
                        index=pd.to_datetime([e["date"] for e in result.equity_curve]),
                    )
                    # Normalise to initial allocation
                    eq_norm = eq / eq.iloc[0]
                    equity_curves[sym] = eq_norm

                per_symbol_results.append({
                    "symbol": sym,
                    "strategy_type": pos.strategy_type,
                    "weight": round(float(w), 6),
                    "status": "success",
                    "performance": result.to_dict()["performance"],
                })

            except Exception as e:
                per_symbol_results.append({
                    "symbol": sym,
                    "strategy_type": pos.strategy_type,
                    "weight": round(float(w), 6),
                    "status": "error",
                    "error": str(e),
                })

        # ── Build combined equity curve ─────────────────────────────────
        combined_curve = self._build_combined_equity_curve(
            equity_curves, symbols, weights
        )

        # ── Portfolio-level performance ─────────────────────────────────
        portfolio_perf = self._aggregate_portfolio_performance(
            per_symbol_results, combined_curve
        )

        result_obj = PortfolioBacktestResult(
            symbols=symbols,
            weights=weights.tolist(),
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            per_symbol_results=per_symbol_results,
            portfolio_performance=portfolio_perf,
            combined_equity_curve=combined_curve,
        )
        return result_obj.to_dict()

    def _build_combined_equity_curve(
        self,
        equity_curves: Dict[str, pd.Series],
        symbols: List[str],
        weights: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Build weighted combined equity curve from per-symbol normalised curves."""
        if not equity_curves:
            return []

        # Find symbols with data
        available = [s for s in symbols if s in equity_curves]
        if not available:
            return []

        # Get corresponding weights (renormalised)
        avail_idx = [list(symbols).index(s) for s in available]
        avail_w = weights[avail_idx]
        if avail_w.sum() > 0:
            avail_w = avail_w / avail_w.sum()

        # Resample all curves to common date range
        df = pd.DataFrame({s: equity_curves[s] for s in available})
        df = df.fillna(method="ffill").fillna(method="bfill").dropna()

        if df.empty:
            return []

        # Weighted sum
        combined = (df * avail_w).sum(axis=1)
        base = combined.iloc[0]
        combined = combined / base * self.initial_capital  # scale to capital

        return [
            {"date": d.strftime("%Y-%m-%d"), "value": round(v, 2)}
            for d, v in combined.items()
        ]

    def _aggregate_portfolio_performance(
        self,
        per_symbol_results: List[Dict[str, Any]],
        combined_curve: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute portfolio-level KPIs from per-symbol results and combined curve."""
        successful = [r for r in per_symbol_results if r.get("status") == "success"]

        if not successful:
            return {
                "status": "no_successful_backtests",
                "total_return_pct": None,
                "sharpe_ratio": None,
                "max_drawdown_pct": None,
                "total_trades": 0,
            }

        # Weighted average metrics
        total_w = sum(r["weight"] for r in successful)
        if total_w == 0:
            total_w = 1.0

        def wavg(key: str) -> float:
            return sum(
                r["performance"][key] * r["weight"] for r in successful
                if key in r.get("performance", {})
            ) / total_w

        # From combined equity curve
        eq_values = [e["value"] for e in combined_curve]
        if len(eq_values) >= 2:
            total_return = (eq_values[-1] - eq_values[0]) / eq_values[0] * 100
            # Sharpe from daily returns
            arr = np.array(eq_values)
            rets = np.diff(arr) / arr[:-1]
            sharpe = float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else 0.0
            # Max drawdown
            peak = np.maximum.accumulate(arr)
            dd = (arr - peak) / peak * 100
            max_dd = float(np.min(dd))
        else:
            total_return = wavg("total_return_pct")
            sharpe = wavg("sharpe_ratio")
            max_dd = wavg("max_drawdown_pct")

        return {
            "status": "success",
            "total_return_pct": round(float(total_return), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "max_drawdown_pct": round(float(max_dd), 4),
            "weighted_win_rate_pct": round(wavg("win_rate_pct"), 2),
            "total_trades": sum(r["performance"].get("total_trades", 0) for r in successful),
            "successful_symbols": len(successful),
            "total_symbols": len(per_symbol_results),
        }

    # ── Correlation Matrix ────────────────────────────────────────────────

    def calculate_correlation_matrix(
        self, lookback_days: int = 252
    ) -> Dict[str, Any]:
        """
        Calculate Pearson correlation matrix for all portfolio positions.

        Args:
            lookback_days: Number of trading days to look back (default 252 = 1 year)

        Returns:
            {
                "symbols": [...],
                "matrix": [[...], ...],
                "lookback_days": 252,
                "data_bars": 245,
                "error": None,
            }
        """
        if not self._positions:
            return {
                "symbols": [],
                "matrix": [],
                "lookback_days": lookback_days,
                "data_bars": 0,
                "error": "No positions in portfolio",
            }

        symbols = list(self._positions.keys())

        try:
            if not _YF_AVAILABLE or yf is None:
                return {
                    "symbols": symbols,
                    "matrix": [],
                    "lookback_days": lookback_days,
                    "data_bars": 0,
                    "error": "yfinance not available",
                }

            # Download price data
            price_frames = {}
            for sym in symbols:
                try:
                    ticker = yf.Ticker(sym)
                    # Approximate lookback_days as calendar days (×1.4 buffer for weekends)
                    cal_days = int(lookback_days * 1.4)
                    end = datetime.now()
                    start = end - timedelta(days=cal_days)
                    raw = ticker.history(start=start.strftime("%Y-%m-%d"),
                                         end=end.strftime("%Y-%m-%d"))
                    if not raw.empty and len(raw) > 5:
                        price_frames[sym] = raw["Close"]
                except Exception:
                    pass

            if len(price_frames) < 2:
                return {
                    "symbols": symbols,
                    "matrix": [],
                    "lookback_days": lookback_days,
                    "data_bars": 0,
                    "error": "Insufficient price data for correlation (need ≥ 2 symbols with data)",
                }

            # Align and compute correlation
            prices_df = pd.DataFrame(price_frames).dropna()
            returns_df = prices_df.pct_change().dropna()

            # Keep only last lookback_days trading rows
            if len(returns_df) > lookback_days:
                returns_df = returns_df.iloc[-lookback_days:]

            valid_symbols = list(returns_df.columns)
            corr = returns_df.corr()
            matrix = [[round(float(corr.loc[r, c]), 4) for c in valid_symbols] for r in valid_symbols]

            return {
                "symbols": valid_symbols,
                "matrix": matrix,
                "lookback_days": lookback_days,
                "data_bars": len(returns_df),
                "error": None,
            }

        except Exception as e:
            return {
                "symbols": symbols,
                "matrix": [],
                "lookback_days": lookback_days,
                "data_bars": 0,
                "error": str(e),
            }

    # ── Utility ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        pos_strs = [
            f"{sym}({pos.strategy_type}, w={pos.weight:.2f})"
            for sym, pos in self._positions.items()
        ]
        return f"PortfolioManager(capital={self.initial_capital:,.0f}, positions=[{', '.join(pos_strs)}])"

    def __len__(self) -> int:
        return len(self._positions)

    def __contains__(self, symbol: str) -> bool:
        return symbol.upper().strip() in self._positions
