# 📈 Stock Analysis Pro

> A full-stack stock analysis platform with interactive candlestick charts, real-time technical indicators, ML-powered trading signals, and a quantitative backtesting framework — built with FastAPI and TradingView Lightweight Charts.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![PWA](https://img.shields.io/badge/PWA-Ready-5A0FC8?logo=pwa)](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps)

---

## 📸 Screenshots

```
┌─────────────────────────────────────────────────────────────┐
│  [Screenshot: Main Dashboard]                               │
│  Candlestick chart for 2330.TW (TSMC) with MA5/MA20/MA60   │
│  overlays and a signal badge showing "BUY (75% confidence)" │
└─────────────────────────────────────────────────────────────┘
```
> 🖼️ *Main dashboard — K-line chart with moving averages, RSI panel, MACD histogram, and AI signal card.*

```
┌─────────────────────────────────────────────────────────────┐
│  [Screenshot: Backtest Results]                             │
│  Equity curve: RF Strategy vs. Buy & Hold for 2330.TW 2024  │
│  Metrics: Return 10.99% | Win Rate 75% | Sharpe 1.34        │
└─────────────────────────────────────────────────────────────┘
```
> 🖼️ *Backtest results panel — equity curve comparison, trade history table, and walk-forward validation folds.*

```
┌─────────────────────────────────────────────────────────────┐
│  [Screenshot: Mobile PWA View]                              │
│  Responsive dark theme layout on iPhone — installable as    │
│  a standalone app via "Add to Home Screen"                  │
└─────────────────────────────────────────────────────────────┘
```
> 🖼️ *Mobile / PWA view — responsive dark theme, installable as a native-like app.*

---

## ✨ Features

### 📊 Real-Time Charting
- **Candlestick Charts** powered by [TradingView Lightweight Charts](https://tradingview.github.io/lightweight-charts/)
- Overlay **MA5 / MA20 / MA60** moving averages on the price chart
- Separate **RSI-14** panel with overbought (70) / oversold (30) zones
- **MACD histogram** with signal line crossover visualization
- Time range selector: **1M · 3M · 6M · 1Y · 2Y**

### 🤖 ML Prediction Engine
- **Technical Predictor**: Rule-based buy/sell/hold signals with confidence scores  
  (MA crossover, RSI, MACD signals combined with weighted scoring)
- **Random Forest Predictor**: scikit-learn `RandomForestClassifier` trained on 10 engineered features:
  - RSI(14), MACD histogram, MA crossover ratios (10/30, 20/60)
  - Bollinger Band %B, Volume ratio, 5d/10d/20d momentum, 20d volatility
- Feature importance ranking and probability output for each prediction

### ⚙️ Backtesting Framework
Four fully implemented strategies, all supporting **Taiwan stock trading costs** (0.1425% buy + 0.4425% sell):

| Strategy | Description |
|---|---|
| `ma_crossover` | MA10/MA30 golden/death cross |
| `rsi_reversal` | RSI < 30 buy, RSI > 70 sell |
| `macd_signal` | MACD/signal line crossover |
| `rf` | Random Forest ML classifier |

- Buy & Hold benchmark comparison
- **Walk-Forward Validation** (18-window, anti-look-ahead-bias)
- Equity curve visualization with per-day granularity
- Full trade history: entry/exit price, shares, P&L, P&L %, commission

### 🎮 Simulated Trading
- Virtual portfolio management with SQLite persistence
- Multi-account isolation
- Order execution with realistic commission calculation

### 📱 Progressive Web App (PWA)
- Installable on iOS / Android via "Add to Home Screen"
- Service Worker caching (offline-capable for cached assets)
- Responsive dark theme optimized for mobile

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) 0.115 + Uvicorn |
| **Market Data** | [yfinance](https://github.com/ranaroussi/yfinance) 0.2 |
| **Data Processing** | [pandas](https://pandas.pydata.org/) 2.2 + [NumPy](https://numpy.org/) 2.2 |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/) 1.6 (RandomForestClassifier) |
| **Backtesting** | [Backtrader](https://www.backtrader.com/) |
| **Frontend Charts** | [Lightweight Charts](https://tradingview.github.io/lightweight-charts/) v4 |
| **Frontend Styling** | [Tailwind CSS](https://tailwindcss.com/) v3 (CDN) |
| **Database** | SQLite (simulated trading) |
| **PWA** | Service Worker + Web App Manifest |

---

## 🚀 Quick Start

### Prerequisites

- Python **3.8+**
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/agBythos/stock-analysis-pro.git
cd stock-analysis-pro

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** Backtrader is not in `requirements.txt` by default. Install it separately:
> ```bash
> pip install backtrader
> ```

### Run

```bash
python server.py
```

| URL | Description |
|---|---|
| `http://localhost:8000` | Frontend (redirects to SPA) |
| `http://localhost:8000/static/index.html` | Main application UI |
| `http://localhost:8000/docs` | Interactive API docs (Swagger UI) |
| `http://localhost:8000/redoc` | API docs (ReDoc) |

---

## 📡 API Reference

### Stock Data
```bash
# OHLCV + indicators in one call
GET /api/stock/{symbol}?range=3M

# Supported ranges: 1M, 3M, 6M, 1Y, 2Y
# Examples: AAPL, MSFT, NVDA, 2330.TW (TSMC), 2317.TW (Hon Hai)
```

### ML Prediction
```bash
# Technical indicator prediction
GET /api/stock/{symbol}/predict

# Random Forest prediction
GET /api/stock/{symbol}/predict/rf
```

**Example response:**
```json
{
  "symbol": "2330.TW",
  "signal": "BUY",
  "confidence": 75.0,
  "reason": "ML prediction based on momentum_5d, rsi_14, macd_hist",
  "probabilities": { "up": 0.75, "down": 0.25 }
}
```

### Backtesting
```bash
# List available strategies
GET /api/backtest/strategies

# Run backtest
POST /api/backtest/run
Content-Type: application/json

{
  "symbol": "2330.TW",
  "strategy": "rf",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "initial_capital": 100000,
  "parameters": {
    "forward_days": 5,
    "confidence_threshold": 0.50,
    "retrain_period": 60
  }
}
```

See [API.md](./API.md) for the full endpoint reference.

---

## 📊 Backtest Results

### TSMC (2330.TW) — Full Year 2024

| Metric | RF Strategy | Buy & Hold |
|---|---|---|
| **Total Return** | **+10.99%** | +6.87% |
| **Win Rate** | **75.00%** | — |
| **Sharpe Ratio** | **1.34** | 0.82 |
| **Max Drawdown** | -8.21% | -15.43% |
| **Total Trades** | 12 | 1 |
| **Final Portfolio** | $110,990 | $106,870 |

> Initial capital: NT$100,000 · Period: 2024-01-01 → 2024-12-31  
> Trading costs: 0.1425% (buy) + 0.4425% (sell), Taiwan market standard

### Walk-Forward Validation (18 windows, 2023–2024)

Walk-forward testing prevents look-ahead bias by training on historical data and testing strictly on unseen future data:

| Metric | Value |
|---|---|
| Total Windows | 18 |
| Avg. Monthly Return | +0.67% |
| Best Window | +7.85% |
| Worst Window | -4.14% |
| Return Std Dev | ±2.90% |
| Avg. Max Drawdown | 1.50% |

> Training window: 6 months · Test window: 1 month · Rolling monthly

---

## 🗂️ Project Structure

```
stock-app/
│
├── server.py                  # FastAPI app: API endpoints, strategies, ML models
│   ├── TechnicalPredictor     #   Rule-based predictor (MA/RSI/MACD)
│   ├── RandomForestPredictor  #   ML predictor with feature engineering
│   ├── BacktestEngine         #   Backtrader orchestration layer
│   ├── MACrossoverStrategy    #   MA10/30 golden cross strategy
│   ├── RSIReversalStrategy    #   RSI oversold/overbought strategy
│   ├── MACDSignalStrategy     #   MACD crossover strategy
│   ├── RFStrategy             #   Random Forest ML strategy
│   └── SimulatedTrading       #   Virtual portfolio endpoints
│
├── walk_forward.py            # Walk-Forward Validation engine (anti-lookahead)
│
├── static/
│   ├── index.html             # Single-page frontend application
│   ├── manifest.json          # PWA manifest
│   ├── service-worker.js      # PWA service worker (cache-first)
│   ├── icon-192.png           # App icon (192×192)
│   └── icon-512.png           # App icon (512×512)
│
├── tests/
│   ├── test_e2e.py            # End-to-end API tests (all endpoints)
│   ├── test_sim_trading.py    # Simulated trading system tests
│   ├── test_walk_forward.py   # Walk-forward validation tests
│   ├── test_data_fetch.py     # yfinance data fetching tests
│   └── test_backtrader_fix.py # Backtrader integration smoke tests
│
├── reports/
│   └── 2026-02-17/
│       ├── phase1-rf-dev.md   # Phase 1 RF backtest results
│       └── backtrader-phase3-fix.md  # Phase 3 bug fix report
│
├── walk_forward_result.json   # Latest walk-forward validation output
├── sim_trading.db             # SQLite database for simulated trading
├── requirements.txt           # Python dependencies
├── API.md                     # Full API documentation
└── README.md                  # This file
```

---

## 🗺️ Roadmap

### ✅ Phase 0 — Core Infrastructure
- [x] FastAPI backend with CORS and static file serving
- [x] yfinance data fetching (US + Taiwan markets)
- [x] Technical indicators: MA5/20/60, RSI-14, MACD
- [x] TradingView Lightweight Charts frontend
- [x] Tailwind CSS dark theme responsive UI
- [x] Time range selector (1M / 3M / 6M / 1Y / 2Y)
- [x] Technical Predictor with buy/sell/hold signals
- [x] Feature importance API

### ✅ Phase 1 — ML + Backtesting
- [x] `RandomForestPredictor` with 10-feature engineering pipeline
- [x] Backtrader integration with Taiwan stock commission model
- [x] 4 strategies: MA Crossover, RSI Reversal, MACD Signal, Random Forest
- [x] Buy & Hold benchmark comparison
- [x] Equity curve visualization
- [x] Trade history table (entry/exit/P&L)
- [x] Walk-Forward Validation engine (18-window, anti-lookahead)
- [x] **RF strategy: 75% win rate, +10.99% return on TSMC 2024**

### 🔄 Phase 2 — Platform Expansion *(in progress)*
- [x] Simulated trading system (virtual portfolio, multi-account, SQLite)
- [x] PWA support (installable, service worker, offline cache)
- [x] Mobile-responsive layout optimization
- [ ] Parameter optimization (grid search for strategy params)
- [ ] Multi-stock portfolio backtesting
- [ ] Real-time WebSocket price streaming
- [ ] User authentication & saved watchlists
- [ ] Export backtest report as PDF/CSV

---

## 🧪 Testing

```bash
# End-to-end API tests (requires running server)
python test_e2e.py

# Simulated trading system tests
python test_sim_trading.py

# Walk-forward validation tests
python test_walk_forward.py

# Data fetching tests
python test_data_fetch.py

# Backtrader integration tests
python test_backtrader_fix.py
```

Test coverage includes: all REST endpoints, indicator calculation accuracy, ML prediction pipeline, backtest trade execution, and walk-forward window generation.

---

## 🌐 Supported Markets

| Market | Symbol Format | Examples |
|---|---|---|
| US (NASDAQ / NYSE) | Ticker | `AAPL`, `MSFT`, `NVDA`, `TSLA`, `GOOGL` |
| Taiwan (TWSE) | Code + `.TW` | `2330.TW` (TSMC), `2317.TW` (Hon Hai) |

---

## ⚠️ Disclaimer

This project is built for **educational and portfolio demonstration purposes only**. Predictions and backtest results do not guarantee future performance. Nothing in this application constitutes financial advice. Always conduct your own due diligence before making any investment decisions.

> 投資有風險，決策需謹慎。本系統僅供學習參考，不構成任何投資建議。

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Bythos (agBythos)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 👨‍💻 Author

**Bythos** (agBythos)  
- GitHub: [@agBythos](https://github.com/agBythos)  
- Email: agbythos@gmail.com  
- Fiverr: [@agBythos](https://fiverr.com/agBythos)

---

*Built with ❤️ using FastAPI, scikit-learn, Backtrader, and TradingView Lightweight Charts*
