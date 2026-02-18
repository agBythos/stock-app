"""
Portfolio Analyzer — Phase 7
Analyzes multiple stocks simultaneously: individual metrics, correlation matrix,
and combined portfolio-level performance metrics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

TRADING_DAYS = 252          # annualisation factor
RISK_FREE_RATE = 0.045      # 4.5% annual (US 3-month T-bill proxy)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AssetMetrics:
    """Per-asset statistics."""
    symbol: str
    total_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    sortino_ratio: float
    win_rate_pct: float           # % of days with positive return
    beta_vs_portfolio: float      # beta vs equal-weight portfolio (set after all assets computed)


@dataclass
class PortfolioResult:
    """Full portfolio analysis result."""
    symbols: List[str]
    weights: List[float]          # normalised to sum=1
    period: str
    start_date: str
    end_date: str

    asset_metrics: List[AssetMetrics]

    # Correlation matrix — stored as list-of-lists for JSON serialisation
    correlation_matrix: List[List[float]]
    correlation_labels: List[str]           # = symbols

    # Portfolio-level (weighted combination)
    portfolio_metrics: Dict

    # Normalised price series (base=100 at start) for charting
    normalized_prices: Dict[str, List[float]]
    portfolio_curve: List[float]            # weighted normalised equity curve
    dates: List[str]                        # ISO date strings

    def to_dict(self) -> Dict:
        return {
            "symbols": self.symbols,
            "weights": [round(w, 6) for w in self.weights],
            "period": self.period,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "asset_metrics": [
                {
                    "symbol": m.symbol,
                    "total_return_pct": round(m.total_return_pct, 4),
                    "annualized_return_pct": round(m.annualized_return_pct, 4),
                    "annualized_volatility_pct": round(m.annualized_volatility_pct, 4),
                    "sharpe_ratio": round(m.sharpe_ratio, 4),
                    "max_drawdown_pct": round(m.max_drawdown_pct, 4),
                    "calmar_ratio": round(m.calmar_ratio, 4),
                    "sortino_ratio": round(m.sortino_ratio, 4),
                    "win_rate_pct": round(m.win_rate_pct, 4),
                    "beta_vs_portfolio": round(m.beta_vs_portfolio, 4),
                }
                for m in self.asset_metrics
            ],
            "correlation_matrix": [
                [round(v, 4) for v in row] for row in self.correlation_matrix
            ],
            "correlation_labels": self.correlation_labels,
            "portfolio_metrics": {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in self.portfolio_metrics.items()},
            "normalized_prices": {
                sym: [round(v, 4) for v in vals]
                for sym, vals in self.normalized_prices.items()
            },
            "portfolio_curve": [round(v, 4) for v in self.portfolio_curve],
            "dates": self.dates,
        }


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------

class PortfolioAnalyzer:
    """
    Multi-stock portfolio analyser.

    Usage::

        analyzer = PortfolioAnalyzer()
        result = analyzer.analyze(
            symbols=["AAPL", "MSFT", "GOOG"],
            period="1y",
            weights=[0.4, 0.3, 0.3],   # optional; defaults to equal weight
        )
        data = result.to_dict()
    """

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        symbols: List[str],
        period: str = "1y",
        weights: Optional[List[float]] = None,
    ) -> PortfolioResult:
        """
        Fetch price data for *symbols* and compute full portfolio analytics.

        Parameters
        ----------
        symbols  : list of ticker strings (max 10)
        period   : yfinance period string, e.g. "1y", "2y", "6mo"
        weights  : portfolio weights (must sum to ≈ 1.0).
                   If None → equal weight.

        Returns
        -------
        PortfolioResult (call .to_dict() for JSON-serialisable dict)
        """
        if len(symbols) < 2:
            raise ValueError("Portfolio analysis requires at least 2 symbols.")
        if len(symbols) > 10:
            raise ValueError("Maximum 10 symbols per portfolio analysis.")

        symbols = [s.upper().strip() for s in symbols]

        # Normalise weights
        if weights is None:
            w = np.ones(len(symbols)) / len(symbols)
        else:
            w = np.array(weights, dtype=float)
            if len(w) != len(symbols):
                raise ValueError("len(weights) must equal len(symbols).")
            total = w.sum()
            if abs(total) < 1e-9:
                raise ValueError("Weights must not all be zero.")
            w = w / total   # normalise to sum=1

        # Fetch data
        prices_df = self._fetch_prices(symbols, period)

        # Drop symbols with no data
        valid_symbols = [s for s in symbols if s in prices_df.columns and prices_df[s].notna().sum() > 10]
        if len(valid_symbols) < 2:
            raise ValueError(f"Not enough valid price data for symbols: {symbols}")

        # Re-align weights to valid symbols
        valid_idx = [symbols.index(s) for s in valid_symbols]
        w_valid = w[valid_idx]
        w_valid = w_valid / w_valid.sum()   # renormalise

        prices_df = prices_df[valid_symbols].dropna()

        if len(prices_df) < 5:
            raise ValueError("Not enough overlapping trading days for portfolio analysis.")

        returns_df = prices_df.pct_change().dropna()

        # Asset-level metrics
        asset_metrics = self._compute_asset_metrics(valid_symbols, prices_df, returns_df, w_valid)

        # Correlation matrix
        corr_matrix, corr_labels = self._compute_correlation(returns_df, valid_symbols)

        # Portfolio-level metrics
        portfolio_ret = (returns_df * w_valid).sum(axis=1)
        portfolio_metrics = self._compute_portfolio_metrics(portfolio_ret, returns_df, w_valid)

        # Normalised price curves
        norm_prices = self._normalise_prices(prices_df, valid_symbols)
        portfolio_curve = (norm_prices * w_valid).sum(axis=1).tolist()

        dates = [d.strftime("%Y-%m-%d") for d in prices_df.index]

        return PortfolioResult(
            symbols=valid_symbols,
            weights=w_valid.tolist(),
            period=period,
            start_date=dates[0] if dates else "",
            end_date=dates[-1] if dates else "",
            asset_metrics=asset_metrics,
            correlation_matrix=corr_matrix,
            correlation_labels=corr_labels,
            portfolio_metrics=portfolio_metrics,
            normalized_prices={s: norm_prices[s].tolist() for s in valid_symbols},
            portfolio_curve=portfolio_curve,
            dates=dates,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_prices(self, symbols: List[str], period: str) -> pd.DataFrame:
        """Download adjusted close prices for all symbols."""
        try:
            raw = yf.download(
                symbols,
                period=period,
                progress=False,
                auto_adjust=True,
                group_by="ticker" if len(symbols) > 1 else "column",
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to download price data: {exc}") from exc

        if raw.empty:
            raise RuntimeError("yfinance returned empty data.")

        # Extract Close prices
        if len(symbols) == 1:
            df = raw[["Close"]].copy()
            df.columns = symbols
        else:
            # Multi-level columns: (field, symbol)
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw["Close"].copy()
            else:
                df = raw[["Close"]].copy()
                df.columns = symbols

        df.index = pd.to_datetime(df.index).normalize()
        return df

    def _compute_asset_metrics(
        self,
        symbols: List[str],
        prices_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
    ) -> List[AssetMetrics]:
        """Compute per-asset statistics."""
        # Weighted portfolio returns for beta calculation
        port_ret = (returns_df * weights).sum(axis=1)
        port_var = port_ret.var()

        metrics = []
        for sym in symbols:
            r = returns_df[sym].dropna()
            p = prices_df[sym].dropna()

            n_days = len(r)
            if n_days < 2:
                continue

            total_ret = (p.iloc[-1] / p.iloc[0] - 1) * 100
            ann_ret = ((1 + total_ret / 100) ** (TRADING_DAYS / n_days) - 1) * 100
            ann_vol = r.std() * np.sqrt(TRADING_DAYS) * 100
            rf_daily = (1 + self.risk_free_rate) ** (1 / TRADING_DAYS) - 1
            excess = r - rf_daily
            sharpe = (excess.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0

            # Sortino (downside deviation)
            downside = r[r < rf_daily]
            dd_std = downside.std() if len(downside) > 1 else 1e-9
            sortino = (excess.mean() / dd_std * np.sqrt(TRADING_DAYS)) if dd_std > 0 else 0.0

            # Max drawdown
            cum = (1 + r).cumprod()
            roll_max = cum.cummax()
            drawdown = (cum - roll_max) / roll_max * 100
            max_dd = drawdown.min()

            calmar = (ann_ret / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0
            win_rate = (r > 0).sum() / n_days * 100

            # Beta vs portfolio
            cov_with_port = r.cov(port_ret.reindex(r.index).fillna(0))
            beta = (cov_with_port / port_var) if port_var > 1e-12 else 0.0

            metrics.append(AssetMetrics(
                symbol=sym,
                total_return_pct=float(total_ret),
                annualized_return_pct=float(ann_ret),
                annualized_volatility_pct=float(ann_vol),
                sharpe_ratio=float(sharpe),
                max_drawdown_pct=float(max_dd),
                calmar_ratio=float(calmar),
                sortino_ratio=float(sortino),
                win_rate_pct=float(win_rate),
                beta_vs_portfolio=float(beta),
            ))

        return metrics

    def _compute_correlation(
        self, returns_df: pd.DataFrame, symbols: List[str]
    ) -> tuple[List[List[float]], List[str]]:
        """Compute Pearson correlation matrix."""
        corr = returns_df[symbols].corr()
        matrix = [[float(corr.loc[r, c]) for c in symbols] for r in symbols]
        return matrix, symbols

    def _compute_portfolio_metrics(
        self,
        port_ret: pd.Series,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
    ) -> Dict:
        """Portfolio-level aggregated metrics."""
        n = len(port_ret)
        rf_daily = (1 + self.risk_free_rate) ** (1 / TRADING_DAYS) - 1
        excess = port_ret - rf_daily

        ann_ret = ((1 + port_ret.mean()) ** TRADING_DAYS - 1) * 100
        ann_vol = port_ret.std() * np.sqrt(TRADING_DAYS) * 100
        sharpe = (excess.mean() / port_ret.std() * np.sqrt(TRADING_DAYS)) if port_ret.std() > 0 else 0.0

        downside = port_ret[port_ret < rf_daily]
        dd_std = downside.std() if len(downside) > 1 else 1e-9
        sortino = (excess.mean() / dd_std * np.sqrt(TRADING_DAYS)) if dd_std > 0 else 0.0

        cum = (1 + port_ret).cumprod()
        roll_max = cum.cummax()
        drawdown = (cum - roll_max) / roll_max * 100
        max_dd = float(drawdown.min())

        total_ret = float((cum.iloc[-1] - 1) * 100)
        calmar = (ann_ret / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

        # Portfolio variance from covariance matrix
        cov_matrix = returns_df.cov()
        port_var_annual = float(np.dot(weights, np.dot(cov_matrix.values, weights)) * TRADING_DAYS)
        port_vol_from_cov = float(np.sqrt(port_var_annual) * 100)

        # Diversification ratio: weighted avg vol / portfolio vol
        individual_vols = np.array([returns_df[s].std() * np.sqrt(TRADING_DAYS) for s in returns_df.columns])
        weighted_avg_vol = float(np.dot(weights, individual_vols) * 100)
        diversification_ratio = float(weighted_avg_vol / port_vol_from_cov) if port_vol_from_cov > 0 else 1.0

        return {
            "total_return_pct": round(total_ret, 4),
            "annualized_return_pct": round(float(ann_ret), 4),
            "annualized_volatility_pct": round(float(ann_vol), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "sortino_ratio": round(float(sortino), 4),
            "max_drawdown_pct": round(max_dd, 4),
            "calmar_ratio": round(float(calmar), 4),
            "diversification_ratio": round(diversification_ratio, 4),
            "portfolio_volatility_cov_pct": round(port_vol_from_cov, 4),
            "trading_days": n,
        }

    def _normalise_prices(
        self, prices_df: pd.DataFrame, symbols: List[str]
    ) -> pd.DataFrame:
        """Normalise all prices to 100 at start date."""
        first_valid = prices_df[symbols].apply(lambda col: col.first_valid_index())
        norm = pd.DataFrame(index=prices_df.index)
        for sym in symbols:
            fv = first_valid[sym]
            base = prices_df.loc[fv, sym]
            norm[sym] = prices_df[sym] / base * 100
        return norm
