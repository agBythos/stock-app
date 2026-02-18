# 📈 Stock Analysis Pro

> A full-stack Taiwan stock analysis platform with interactive candlestick charts, real-time technical indicators, ML ensemble predictions, Hidden Markov Model market-regime detection, and a rigorous quantitative backtesting framework — built with FastAPI, TradingView Lightweight Charts, and advanced time-series cross-validation methods (Purged Walk-Forward + CPCV) from Marcos Lopez de Prado's *Advances in Financial Machine Learning*.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![hmmlearn](https://img.shields.io/badge/hmmlearn-0.3-blueviolet)](https://hmmlearn.readthedocs.io)
[![Backtrader](https://img.shields.io/badge/Backtrader-1.9-orange)](https://www.backtrader.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![PWA](https://img.shields.io/badge/PWA-Ready-5A0FC8?logo=pwa)](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps)

---

## 📖 Overview

Stock Analysis Pro is a personal portfolio project developed by a statistics undergraduate student. It combines a responsive single-page web app with a Python backend that supports:

- **Multiple ML classifiers** — Random Forest, CatBoost, XGBoost, LightGBM, and majority-vote ensemble
- **Hidden Markov Model (HMM)** — unsupervised market regime detection (Bull / Sideways / Bear)
- **Backtrader simulation engine** — calibrated for Taiwan Stock Exchange (TWSE) trading costs
- **Two-tier anti-leakage validation pipeline** — rolling **Purged Walk-Forward** (Phase 3.5) and **Combinatorially Purged Cross-Validation / CPCV** (Phase 4)
- **Model persistence + Discord alert system** — regime change monitoring with automatic notifications (Phase 7)

> ⚠️ **Educational use only.** Backtest results do not guarantee future performance. This application does not constitute financial advice.  
> 投資有風險，決策需謹慎。本系統僅供學習參考，不構成任何投資建議。

---

## ✨ Features

- **Interactive K-line charts** — TradingView Lightweight Charts with MA5/MA20/MA60 overlays, RSI-14 panel, MACD histogram; time range: 1M · 3M · 6M · 1Y · 2Y
- **Multi-model ML prediction** — Random Forest, CatBoost, XGBoost, LightGBM, and a majority-vote ensemble; 10-feature engineering pipeline
- **HMM market regime detection** — 3-state Gaussian HMM (Bull / Sideways / Bear) trained on log-return, 20-day volatility, and volume ratio
- **HMM-filtered backtesting** — `HMMFilterStrategy` uses regime awareness to avoid trading in Bear markets, reducing drawdown
- **4-strategy Backtrader engine** — MA Crossover, RF classifier, HMM Filter, and custom strategies with TWSE asymmetric cost model
- **Purged Walk-Forward Validation** — rolling train/test windows with purge + embargo to eliminate label-leakage
- **CPCV Validation** — C(N,k) combinatorial splits → φ independent backtest paths; Sharpe distribution statistics (mean, std, 95% CI)
- **Model persistence cache** — trained RF/HMM models are pickled and reused across API calls (from ~60s → <1s)
- **Regime-change alert system** — `MonitorScheduler` polls daily at 15:30 TWN, detects regime transitions, and writes to `alert-log.json`
- **Discord notifications** — webhook-based Embed messages on Bull→Bear or Bear→Bull transitions
- **Simulated trading** — virtual portfolio with SQLite persistence, multi-account isolation, realistic commission calculation
- **Progressive Web App** — installable on iOS/Android, service worker offline cache, responsive dark theme

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | [TradingView Lightweight Charts](https://tradingview.github.io/lightweight-charts/) v4 · [Tailwind CSS](https://tailwindcss.com/) v3 · PWA (Service Worker + Manifest) |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) 0.115 · [Uvicorn](https://www.uvicorn.org/) · [Pydantic](https://docs.pydantic.dev/) v2 |
| **Market Data** | [yfinance](https://github.com/ranaroussi/yfinance) 0.2 (TWSE + US markets) |
| **Data Processing** | [pandas](https://pandas.pydata.org/) 2.2 · [NumPy](https://numpy.org/) 2.2 |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/) 1.6 (RandomForest) · [CatBoost](https://catboost.ai/) · [XGBoost](https://xgboost.readthedocs.io/) · [LightGBM](https://lightgbm.readthedocs.io/) |
| **Market Regime** | [hmmlearn](https://hmmlearn.readthedocs.io/) 0.3.3 (Gaussian HMM, 3-state) |
| **Backtesting Engine** | [Backtrader](https://www.backtrader.com/) — standalone `backtest/` module |
| **Validation Framework** | `validation/` — Purged Walk-Forward (Phase 3.5) + CPCV (Phase 4) |
| **Model Persistence** | `cache/model_cache.py` — pickle-based RF/HMM cache |
| **Alert System** | `alerts/` — regime monitor · Discord webhook notifier |
| **Scheduler** | `scheduler.py` — `MonitorScheduler` (asyncio-based, daily 15:30 TWN) |
| **Database** | SQLite (simulated trading persistence) |
| **Testing** | [pytest](https://pytest.org/) 9 — 100+ test cases across all phases |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Browser / PWA                                │
│   TradingView Charts · Tailwind CSS · Service Worker                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ HTTP / REST
┌──────────────────────────▼──────────────────────────────────────────┐
│                    FastAPI  (server.py)  v2.0.0                     │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ /api/stock   │  │ /api/predict │  │ /api/backtest            │  │
│  │ OHLCV +      │  │ RF / CB /    │  │ /api/validate/           │  │
│  │ indicators   │  │ XGB / LGBM   │  │ walk-forward + cpcv      │  │
│  │              │  │ ensemble     │  │ /api/validate/hmm        │  │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘  │
│         │                 │                        │                │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌────────────▼─────────────┐  │
│  │  yfinance    │  │  ML Engine   │  │  Backtest + Validation    │  │
│  │  (TWSE/US)   │  │  + Model     │  │  ┌──────────────────────┐ │  │
│  └──────────────┘  │  Cache       │  │  │ backtest/            │ │  │
│                    │  (pickle)    │  │  │ BacktraderEngine     │ │  │
│                    └──────────────┘  │  │ RFStrategy           │ │  │
│                                      │  │ HMMFilterStrategy    │ │  │
│  ┌───────────────────────────────┐   │  └──────────────────────┘ │  │
│  │ /api/monitor/*                │   │  ┌──────────────────────┐ │  │
│  │ check · alert-log             │   │  │ validation/          │ │  │
│  │ scheduler-status              │   │  │ PurgedWalkForward    │ │  │
│  └───────────────┬───────────────┘   │  │ CPCVSplitter         │ │  │
│                  │                   │  │ CPCVRunner           │ │  │
│  ┌───────────────▼───────────────┐   │  └──────────────────────┘ │  │
│  │ MonitorScheduler (15:30 TWN)  │   └──────────────────────────┘  │
│  │ alerts/regime_monitor.py      │                                  │
│  │ alerts/discord_notifier.py    │─────────────► Discord Webhook   │
│  └───────────────────────────────┘                                  │
│                                                                     │
│  ┌───────────────────────────────┐  ┌──────────────────────────┐   │
│  │ /api/sim/*  (SimTrading)      │  │ hmm/market_hmm.py        │   │
│  │ SQLite  multi-account DB      │  │ MarketHMM (3-state HMM)  │   │
│  └───────────────────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
stock-app/
├── server.py                     # FastAPI app: all endpoints, ML models, strategies
│
├── backtest/                     # Standalone Backtrader module
│   ├── backtrader_engine.py      #   BacktraderEngine, TaiwanCommission, WalkForwardResult
│   ├── rf_strategy.py            #   RFStrategy (Backtrader + Random Forest)
│   ├── hmm_filter_strategy.py    #   HMMFilterStrategy (Backtrader + HMM regime)
│   └── __init__.py
│
├── hmm/                          # Hidden Markov Model module
│   ├── market_hmm.py             #   MarketHMM class (GaussianHMM, 3-state, auto-label)
│   └── __init__.py
│
├── validation/                   # Anti-leakage validation framework
│   ├── purged_walk_forward.py    #   SplitEngine + PurgeEngine (Phase 3.5)
│   ├── ensemble_trainer.py       #   Multi-model ensemble adapter
│   ├── backtrader_bridge.py      #   TWAECommission + Backtrader integration
│   ├── run_walk_forward.py       #   PurgedWalkForwardRunner orchestrator
│   ├── cpcv_splitter.py          #   CPCVSplitter (Phase 4)
│   ├── cpcv_runner.py            #   CPCVRunner + CPCVReport
│   └── __init__.py
│
├── alerts/                       # Regime-change alert system (Phase 7)
│   ├── regime_monitor.py         #   RegimeMonitor: compare last vs current regime
│   ├── discord_notifier.py       #   DiscordNotifier: webhook Embed messages
│   └── state.json                #   Persistent regime state (auto-generated)
│
├── cache/                        # Model persistence (Phase 7)
│   └── model_cache.py            #   ModelCache: pickle-based RF/HMM cache
│
├── models/                       # Saved model files (auto-generated)
│   ├── rf_{ticker}_{date}.pkl
│   └── hmm_{ticker}_{date}.pkl
│
├── scheduler.py                  # MonitorScheduler (asyncio, daily 15:30 TWN)
│
├── config/
│   ├── walk_forward.py           # WalkForwardConfig (train/test windows, costs)
│   └── cpcv.py                   # CPCVConfig (n_groups, k_test_groups)
│
├── static/
│   ├── index.html                # Single-page application
│   ├── backtest.html             # Backtest strategy UI (Phase 6)
│   ├── manifest.json             # PWA manifest
│   ├── service-worker.js         # Cache-first service worker
│   └── icons/                   # App icons (192×192, 512×512)
│
├── tests/                        # pytest test suite (100+ cases)
│   ├── test_hmm.py               #   HMM: 22 tests
│   ├── test_cpcv_splitter.py     #   CPCV splitter: 8 tests
│   ├── test_cpcv_runner.py       #   CPCV runner: 11 tests
│   ├── test_purged_splitter.py   #   Walk-Forward purge: 8 tests
│   ├── test_purge_embargo.py     #   Embargo boundary: 6 tests
│   ├── test_backtrader_costs.py  #   TaiwanCommission: 8 tests
│   ├── test_phase6_api.py        #   Phase 6 API: 10 tests
│   └── test_scheduler.py         #   Scheduler: 22 tests
│
├── docs/
│   └── survivorship-bias.md
├── requirements.txt
├── scheduler.py
├── API.md                        # Full API reference
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.11+**
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/agBythos/stock-app.git
cd stock-app

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install optional ML boosting libraries (for ensemble predictions)
pip install catboost xgboost lightgbm

# 5. Install HMM support
pip install hmmlearn>=0.3.0

# 6. Install Backtrader (required for backtest / validation endpoints)
pip install backtrader
```

### Configuration (Optional)

To enable Discord regime-change notifications, create a `.env` file:

```ini
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your-webhook-url
```

### Run

```bash
python server.py
```

| URL | Description |
|---|---|
| `http://localhost:8000` | Frontend SPA |
| `http://localhost:8000/static/index.html` | Main application UI |
| `http://localhost:8000/static/backtest.html` | Backtest strategy UI |
| `http://localhost:8000/docs` | Interactive API docs (Swagger UI) |
| `http://localhost:8000/redoc` | API docs (ReDoc) |
| `http://localhost:8000/api/health` | Health check |

---

## 📡 API Endpoint Summary

> Full reference → [API.md](./API.md) · Interactive docs → `/docs`

### Core Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Server health check |
| `GET` | `/api/stock/{symbol}` | OHLCV data; `?period=1mo\|3mo\|6mo\|1y\|2y` |
| `GET` | `/api/stock/{symbol}/indicators` | MA5/MA20/MA60 · RSI-14 · MACD |
| `GET` | `/api/search?q={query}` | Search stocks by symbol or name |

### ML Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/stock/{symbol}/predict` | Technical rule-based predictor |
| `GET` | `/api/predict/catboost/{symbol}` | CatBoost prediction |
| `GET` | `/api/predict/xgboost/{symbol}` | XGBoost prediction |
| `GET` | `/api/predict/lightgbm/{symbol}` | LightGBM prediction |
| `GET` | `/api/predict/ensemble/{symbol}` | Majority-vote ensemble (RF + CB + XGB + LGBM) |

### Backtesting & Validation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/strategies` | List available strategies (ma_crossover, rf, hmm_filter) |
| `POST` | `/api/backtest/run` | Run Backtrader backtest (single window) |
| `POST` | `/api/backtest` | Run backtest (legacy endpoint) |
| `POST` | `/api/validate/walk-forward` | **Purged Walk-Forward** with purge+embargo |
| `POST` | `/api/validate/cpcv` | **CPCV** — combinatorial backtest paths + Sharpe CI |
| `POST` | `/api/validate/hmm` | **HMM regime analysis** + HMMFilterStrategy backtest |

### Monitor & Alerts (Phase 7)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/monitor/check` | Manually trigger regime check + alert write |
| `GET` | `/api/monitor/alert-log` | Recent alert log; `?limit=50` |
| `GET` | `/api/monitor/scheduler-status` | MonitorScheduler next-run time + status |

### Simulated Trading

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/sim/accounts` | Create virtual account |
| `GET` | `/api/sim/accounts` | List all accounts |
| `POST` | `/api/sim/accounts/{id}/trade` | Execute simulated trade |
| `GET` | `/api/sim/accounts/{id}/history` | Trade history |

**Backtest request example:**

```bash
curl -X POST http://localhost:8000/api/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "2330.TW",
    "strategy": "rf",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 1000000
  }'
```

**HMM regime analysis example:**

```bash
curl -X POST http://localhost:8000/api/validate/hmm \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "2330.TW",
    "period": "2y",
    "hmm_n_states": 3,
    "hmm_window": 252
  }'
```

**CPCV request example:**

```bash
curl -X POST http://localhost:8000/api/validate/cpcv \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "2330.TW",
    "n_groups": 6,
    "k_test_groups": 2,
    "period": "5y"
  }'
```

---

## 🤖 Models & Validation Methods

### Machine Learning Models

| Model | Library | Notes |
|---|---|---|
| Random Forest | scikit-learn 1.6 | 100 trees; baseline classifier |
| CatBoost | catboost | GPU-optional; categorical-friendly |
| XGBoost | xgboost | Gradient boosting; high speed |
| LightGBM | lightgbm | Leaf-wise growth; low memory |
| **Ensemble** | custom | Majority vote across all 4 models |

**10-Feature Engineering Pipeline** (common to all models):

```
RSI(14)  ·  MACD histogram  ·  MA crossover ratio (10/30, 20/60)
Bollinger %B  ·  Volume ratio  ·  Momentum (5d / 10d / 20d)
Volatility (20d rolling std)
```

Binary classification: `1` (price up in N days) / `0` (price down or flat)

---

### Phase 5 — HMM Market Regime Detection

Unsupervised 3-state Gaussian HMM trained on market microstructure features:

```
Observation features:
  log_return     — log(close_t / close_{t-1})
  volatility_20d — rolling(20).std() × √252
  volume_ratio   — volume / volume.rolling(20).mean()

States (auto-labelled by mean log_return):
  Bull     — highest mean log_return
  Sideways — intermediate log_return
  Bear     — lowest mean log_return (most negative)
```

**MarketHMM API:**

```python
from hmm.market_hmm import MarketHMM

hmm = MarketHMM(n_states=3)
hmm.fit(df)                       # train on OHLCV DataFrame
regime = hmm.predict(df)          # returns ['Bull', 'Sideways', 'Bear', ...]
current = hmm.current_regime(df)  # most recent state
proba = hmm.predict_proba(df)     # state probabilities per bar
```

**HMMFilterStrategy** (Backtrader integration):
- Trains HMM on historical data
- Enters long positions only when regime ≠ Bear
- Exits positions when Bear regime is detected
- Reduces max drawdown vs. raw RF strategy

---

### Phase 3.5 — Purged Walk-Forward Validation

Rolling train/test windows with two leakage-prevention layers:

```
Full timeline ──────────────────────────────────────────────────────►
         │◄────── train (252 bars) ──────►│◄purge►│◄embargo►│◄ test (21 bars) ►│
                                           ▲5 bars  ▲5 bars
                                    Label horizon  Edge buffer
```

| Parameter | Default | Meaning |
|---|---|---|
| `train_window` | 252 bars | ~1 TWSE trading year |
| `test_window` | 21 bars | ~1 month |
| `label_horizon` | 5 bars | Purge: removes train-end samples whose labels overlap with test set |
| `embargo_bars` | 5 bars | Additional gap to block autocorrelation leakage |
| `min_train_samples` | 200 | Skip fold if purge leaves insufficient training data |

---

### Phase 4 — CPCV (Combinatorially Purged Cross-Validation)

Based on Lopez de Prado, *Advances in Financial Machine Learning*, Chapter 12.

```
N groups (default 6), k test groups (default 2)
  → C(6,2) = 15 train/test combinations
  → φ = k×C(N,k)/N = 5 independent backtest paths

Each path is a non-overlapping time slice; Sharpe statistics
are computed across all 5 paths to yield a distribution.
```

**CPCVReport output:**

```python
result.paths                 # 5 independent BacktestResult objects
result.summary_sharpe_mean   # Mean Sharpe across paths
result.summary_sharpe_std    # Std dev of Sharpe
result.summary_sharpe_ci95   # 95% confidence interval
result.pbo                   # Probability of Back-test Overfitting
```

---

### Phase 7 — Model Persistence + Alert System

**Model Cache** (`cache/model_cache.py`):
- RF and HMM models are pickled after first training
- Subsequent API calls load from cache: **~60s → <1s** per request
- Cache keyed by `{ticker}_{date}`, auto-invalidated on new trading day

**Regime Monitor** (`alerts/regime_monitor.py`):
- Compares current regime vs. last-known regime from `alerts/state.json`
- Detects Bull→Bear and Bear→Bull transitions
- Atomic JSON write to `alert-log.json` (max 200 entries, auto-truncated)

**MonitorScheduler** (`scheduler.py`):
- asyncio-based background loop, no external cron dependency
- Fires daily at **15:30 TWN (UTC+8)** — after TWSE market close
- Writes regime-change events to alert log; triggers Discord webhook if configured
- Idempotent start/stop; survives server restart (state persists via JSON)

---

## 🇹🇼 Taiwan Market Adaptation

### Asymmetric Trading Costs (TWSE)

Taiwan Stock Exchange uses **non-symmetric** transaction costs:

| Direction | Component | Rate |
|---|---|---|
| Buy | Brokerage fee (online discount 40% off) | 0.1425% × 0.6 = **0.0855%** |
| Sell | Brokerage fee | 0.0855% |
| Sell | Securities Transaction Tax (STT) | **0.3%** (0.1% for ETFs) |
| **Buy total** | | **0.0855%** |
| **Sell total** | | **0.3855%** |
| **Round-trip** | | **≈ 0.47%** |

This asymmetry is modelled in `TaiwanCommission(bt.CommInfoBase)` and `WalkForwardConfig` / `CPCVConfig`, ensuring backtest performance reflects real execution costs.

### Supported Markets

| Market | Symbol Format | Examples |
|---|---|---|
| Taiwan (TWSE) | `{code}.TW` | `2330.TW`, `2317.TW`, `0050.TW` |
| US (NASDAQ / NYSE) | Ticker | `AAPL`, `MSFT`, `NVDA`, `TSLA` |

---

## 📊 Performance Results

### TSMC (2330.TW) — Full Year 2024

| Metric | RF Strategy | Buy & Hold |
|---|---|---|
| **Total Return** | **+10.99%** | +6.87% |
| **Win Rate** | **75.00%** | — |
| **Sharpe Ratio** | **1.34** | 0.82 |
| **Max Drawdown** | -8.21% | -15.43% |
| **Total Trades** | 12 | 1 |
| **Final Portfolio** | NT$1,109,900 | NT$1,068,700 |

> Initial capital: NT$1,000,000 · Period: 2024-01-01 → 2024-12-31  
> Trading costs: 0.0855% buy + 0.3855% sell (TWSE standard)

### Walk-Forward Validation Summary (Phase 3.5)

| Metric | Value |
|---|---|
| Rolling windows | 18 |
| Avg. monthly return | +0.67% |
| Best window | +7.85% |
| Worst window | -4.14% |
| Return std dev | ±2.90% |
| Avg. max drawdown | 1.50% |

---

## 🗺️ Roadmap

### ✅ Phase 0 — Core Infrastructure
- [x] FastAPI backend (CORS, static file serving, NumpyEncoder)
- [x] yfinance data pipeline (TWSE + US markets)
- [x] Technical indicators: MA5/20/60, RSI-14, MACD(12/26/9)
- [x] TradingView Lightweight Charts frontend (candlestick + overlays)
- [x] Tailwind CSS dark-theme responsive UI
- [x] Time range selector (1M · 3M · 6M · 1Y · 2Y)
- [x] Technical Predictor (rule-based buy/sell/hold + confidence)
- [x] Stock search endpoint

### ✅ Phase 1 — ML & Backtesting
- [x] `RandomForestPredictor` with 10-feature engineering pipeline
- [x] Backtrader integration with `TaiwanCommission` (asymmetric cost model)
- [x] 4 strategies: MA Crossover · RSI Reversal · MACD Signal · Random Forest
- [x] Buy & Hold benchmark comparison
- [x] Equity curve visualization + trade history table
- [x] Walk-Forward Validation (18-window, anti-lookahead)
- [x] **RF strategy: 75% win rate, +10.99% return on TSMC 2024**

### ✅ Phase 2 — Platform Expansion
- [x] Simulated trading system (virtual portfolio, multi-account, SQLite)
- [x] PWA support (Service Worker, offline cache, Web App Manifest)
- [x] Mobile-responsive layout
- [x] Standalone `backtest/` module (decoupled from `server.py`)
- [x] CatBoost / XGBoost / LightGBM model endpoints
- [x] Majority-vote ensemble endpoint

### ✅ Phase 3.5 — Purged Walk-Forward
- [x] `SplitEngine` + `PurgeEngine` — purge + embargo fold generation
- [x] `PurgedWalkForwardSplitter` — leak-free time-series CV splits
- [x] `PurgedWalkForwardRunner` — orchestrator (train → backtest → aggregate)
- [x] `TWAECommission` Backtrader bridge with asymmetric TWSE costs
- [x] `EnsembleTrainer` multi-model adapter
- [x] `/api/validate/walk-forward` endpoint integration
- [x] Full pytest suite (purge logic, embargo boundary, fold count)

### ✅ Phase 4 — CPCV
- [x] `CPCVSplitter` — C(N,k) combinatorial group split with purge
- [x] `CPCVRunner` — φ-path orchestrator + `CPCVReport`
- [x] Sharpe distribution statistics: mean · std · 95% CI
- [x] Probability of Back-test Overfitting (PBO) metric
- [x] `/api/validate/cpcv` endpoint integration
- [x] 19/19 pytest cases pass (splitter, runner, path count, leakage checks)

### ✅ Phase 5 — HMM Market Regime
- [x] `MarketHMM` — 3-state Gaussian HMM (hmmlearn) with auto-labelling
- [x] 3-dimensional observation: log_return, volatility_20d, volume_ratio
- [x] `HMMFilterStrategy` — Backtrader strategy with regime-based entry/exit
- [x] `hmm_filter` strategy integrated into `BacktraderEngine`
- [x] 22/22 HMM unit tests pass

### ✅ Phase 6 — Full API + UI Integration
- [x] `GET /api/strategies` — lists ma_crossover, rf, hmm_filter with param specs
- [x] `POST /api/backtest/run` — unified Backtrader endpoint using `BacktraderEngine`
- [x] `POST /api/validate/hmm` — HMM regime analysis + `HMMFilterStrategy` backtest
- [x] `static/backtest.html` — strategy comparison UI with equity curve canvas
- [x] 10/10 Phase 6 API integration tests pass

### ✅ Phase 7 — Model Persistence + Alert System
- [x] `ModelCache` — pickle-based RF/HMM cache (~60s → <1s inference)
- [x] `RegimeMonitor` + atomic `alert-log.json` write
- [x] `DiscordNotifier` — webhook Embed on regime transition
- [x] `MonitorScheduler` — asyncio daily scheduler at 15:30 TWN
- [x] `GET /api/monitor/alert-log` + `GET /api/monitor/scheduler-status`
- [x] `POST /api/monitor/check` — manual trigger + log write
- [x] 22/22 scheduler unit tests pass

### ✅ Phase 8 — Ensemble Model Comparison
- [x] Multi-model comparison dashboard (`static/compare.html`)
- [x] Side-by-side RF / CatBoost / XGBoost / LightGBM accuracy display
- [x] Per-model prediction confidence with feature importance charts
- [x] Ensemble majority-vote result with agreement meter
- [x] `/api/compare/*` endpoints for batch model evaluation
- [x] `tests/test_compare_api.py` integration tests

### ✅ Phase 9 — UX & Help System
- [x] In-app Help Modal (`#helpModal`) with 6 topic sections
- [x] Section tooltips on AI Prediction, Simulated Trading, Backtest System panels
- [x] Toast notification system (`showToast`) — success / error / info / warn types
- [x] Signal alert integration (`alerts/signal_alert.py` → Discord webhook)
- [x] Backend: Risk Parity Rebalancing endpoint (`/api/portfolio/rebalance`)
- [x] 25/25 tests pass (15 backend + 10 frontend)

### ✅ Phase 8B — WebSocket Real-Time Dashboard
- [x] `WebSocketConnectionManager` — global connection pool (max 10, per-symbol grouping)
- [x] `_fetch_realtime_quote()` — 1m intraday data + 1d fallback via yfinance
- [x] `/ws/price/{symbol}` endpoint — 5-second push interval, graceful disconnect
- [x] `/api/ws/status` — connection diagnostics endpoint
- [x] `static/realtime.html` — standalone real-time dashboard (TradingView Lightweight Charts v4)
- [x] Exponential-backoff reconnect (1s → 30s); code 1008 rejection on max-connection
- [x] 11/11 WebSocket unit tests pass (full mock, no live network calls)

### 🔄 Future Plans
- [ ] Parameter optimisation (grid search for strategy hyperparams)
- [ ] Multi-stock portfolio backtesting
- [ ] Survivorship-bias mitigation (delisted stock handling)
- [ ] Export backtest report as PDF / CSV
- [ ] User authentication & saved watchlists

---

## 🧪 Testing

```bash
# Run full test suite
pytest tests/ -v

# Individual suites
pytest tests/test_hmm.py               # HMM: 22 tests
pytest tests/test_cpcv_splitter.py     # CPCV splitter: 8 tests
pytest tests/test_cpcv_runner.py       # CPCV runner: 11 tests
pytest tests/test_purged_splitter.py   # Walk-Forward purge
pytest tests/test_purge_embargo.py     # Embargo boundary
pytest tests/test_backtrader_costs.py  # TaiwanCommission accuracy
pytest tests/test_phase6_api.py        # Phase 6 API integration
pytest tests/test_scheduler.py         # Phase 7 scheduler: 22 tests

# End-to-end API tests (requires running server)
python test_e2e.py
python test_endpoint_e2e.py
```

---

## 🎓 Academic Background

This project was developed as a **portfolio and learning exercise** by a Statistics undergraduate student. It applies statistical and ML concepts studied in coursework to a real-world financial domain:

- **Hidden Markov Models** — Gaussian HMM for unsupervised regime detection; EM training via hmmlearn; Viterbi decoding for state sequences
- **Time-series cross-validation** — Purged Walk-Forward and CPCV address data leakage unique to financial panel data (as opposed to i.i.d. assumptions in standard k-fold CV)
- **Ensemble learning** — majority-vote aggregation of heterogeneous boosting classifiers
- **Performance attribution** — Sharpe ratio, max drawdown, PBO, and confidence intervals for rigorous model evaluation
- **Cost-aware modelling** — asymmetric transaction costs reflect real TWSE market microstructure
- **Alert system design** — event-driven architecture with idempotent state persistence

> ⚠️ **Educational use only.** Backtest results do not guarantee future performance. This application does not constitute financial advice.  
> 投資有風險，決策需謹慎。本系統僅供學習參考，不構成任何投資建議。

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License — Copyright (c) 2026 Saklas / agBythos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright
notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 👨‍💻 Author

**Saklas** — Statistics undergraduate student  
- GitHub: [@agBythos](https://github.com/agBythos)
- Email: agbythos@gmail.com

---

*Built with FastAPI · scikit-learn · hmmlearn · CatBoost · XGBoost · LightGBM · Backtrader · TradingView Lightweight Charts*
