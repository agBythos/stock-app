# 📡 Stock Analysis API Documentation

Complete API reference for the Stock Analysis backend server.

**Base URL**: `http://localhost:8000`  
**API Version**: 2.0.0  
**Protocol**: HTTP/REST  
**Response Format**: JSON

---

## 📋 Table of Contents

1. [Authentication](#authentication)
2. [Endpoints](#endpoints)
   - [Root](#get-)
   - [Get Stock Data](#get-apistocksymbol)
   - [Get Technical Indicators](#get-apistocksymbolindicators)
   - [Get AI Prediction](#get-apistocksymbolpredict)
   - [Search Stocks](#get-apisearch)
3. [Data Models](#data-models)
4. [Error Handling](#error-handling)
5. [Rate Limiting](#rate-limiting)

---

## 🔐 Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

> **Note**: In production, consider implementing API key authentication for security.

---

## 🌐 Endpoints

### GET `/`

Root endpoint providing API information and available endpoints.

#### Request
```bash
curl http://localhost:8000/
```

#### Response
```json
{
  "message": "Stock Analysis API",
  "version": "2.0.0",
  "endpoints": {
    "stock_data": "/api/stock/{symbol}",
    "indicators": "/api/stock/{symbol}/indicators",
    "prediction": "/api/stock/{symbol}/predict",
    "search": "/api/search"
  }
}
```

**Status Code**: `200 OK`

---

### GET `/api/stock/{symbol}`

Fetch OHLCV (Open, High, Low, Close, Volume) data for a specific stock.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `symbol` | string | Yes | - | Stock ticker symbol (e.g., `AAPL`, `2330.TW`) |
| `period` | string | No | `1mo` | Time period: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max` |

#### Request Examples

**US Stock (Apple)**
```bash
curl "http://localhost:8000/api/stock/AAPL?period=3mo"
```

**Taiwan Stock (TSMC)**
```bash
curl "http://localhost:8000/api/stock/2330.TW?period=1y"
```

**With Default Period**
```bash
curl "http://localhost:8000/api/stock/MSFT"
```

#### Response Example

```json
{
  "symbol": "AAPL",
  "period": "3mo",
  "data": [
    {
      "date": "2026-01-15",
      "open": 150.25,
      "high": 152.80,
      "low": 149.50,
      "close": 151.75,
      "volume": 85234500
    },
    {
      "date": "2026-01-16",
      "open": 151.80,
      "high": 153.20,
      "low": 150.90,
      "close": 152.40,
      "volume": 72445300
    }
  ],
  "info": {
    "name": "Apple Inc.",
    "currency": "USD",
    "exchange": "NASDAQ"
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Stock ticker symbol |
| `period` | string | Requested time period |
| `data` | array | Array of OHLCV data points |
| `data[].date` | string | Date in `YYYY-MM-DD` format |
| `data[].open` | number | Opening price |
| `data[].high` | number | Highest price of the day |
| `data[].low` | number | Lowest price of the day |
| `data[].close` | number | Closing price |
| `data[].volume` | integer | Trading volume |
| `info.name` | string | Company name |
| `info.currency` | string | Currency code (e.g., USD, TWD) |
| `info.exchange` | string | Stock exchange name |

#### Status Codes

- `200 OK` - Success
- `404 Not Found` - Stock symbol not found
- `422 Unprocessable Entity` - Invalid period parameter
- `500 Internal Server Error` - Server error

---

### GET `/api/stock/{symbol}/indicators`

Calculate and return technical indicators for a stock.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `symbol` | string | Yes | - | Stock ticker symbol |
| `period` | string | No | `3mo` | Time period (same options as above) |

#### Request Example

```bash
curl "http://localhost:8000/api/stock/AAPL/indicators?period=3mo"
```

#### Response Example

```json
{
  "symbol": "AAPL",
  "period": "3mo",
  "indicators": [
    {
      "date": "2026-02-16",
      "close": 151.75,
      "ma5": 150.45,
      "ma20": 148.90,
      "ma60": 145.20,
      "rsi": 62.35,
      "macd": 1.2345,
      "signal": 1.1234,
      "histogram": 0.1111
    },
    {
      "date": "2026-02-17",
      "close": 152.40,
      "ma5": 150.88,
      "ma20": 149.15,
      "ma60": 145.50,
      "rsi": 64.20,
      "macd": 1.3456,
      "signal": 1.2123,
      "histogram": 0.1333
    }
  ]
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Stock ticker symbol |
| `period` | string | Requested time period |
| `indicators` | array | Array of indicator data points |
| `indicators[].date` | string | Date in `YYYY-MM-DD` format |
| `indicators[].close` | number | Closing price |
| `indicators[].ma5` | number\|null | 5-day Moving Average |
| `indicators[].ma20` | number\|null | 20-day Moving Average |
| `indicators[].ma60` | number\|null | 60-day Moving Average |
| `indicators[].rsi` | number\|null | Relative Strength Index (0-100) |
| `indicators[].macd` | number\|null | MACD line value |
| `indicators[].signal` | number\|null | MACD signal line value |
| `indicators[].histogram` | number\|null | MACD histogram (MACD - Signal) |

> **Note**: Early data points may have `null` values for indicators that require historical data (e.g., MA60 needs 60 days).

#### Technical Indicator Details

**Moving Averages (MA)**
- **MA5**: 5-day simple moving average
- **MA20**: 20-day simple moving average
- **MA60**: 60-day simple moving average

**RSI (Relative Strength Index)**
- Period: 14 days
- Range: 0-100
- Overbought: > 70
- Oversold: < 30

**MACD (Moving Average Convergence Divergence)**
- Fast EMA: 12 days
- Slow EMA: 26 days
- Signal line: 9-day EMA of MACD
- Histogram: MACD - Signal

#### Status Codes

- `200 OK` - Success
- `400 Bad Request` - Insufficient data (need 60+ days for full indicators)
- `404 Not Found` - Stock symbol not found
- `500 Internal Server Error` - Server error

---

### GET `/api/stock/{symbol}/predict`

Generate AI-powered buy/sell/hold prediction based on technical analysis.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Stock ticker symbol |

> **Note**: This endpoint uses a fixed 3-month period for analysis.

#### Request Example

```bash
curl "http://localhost:8000/api/stock/AAPL/predict"
```

#### Response Example

**Bullish Signal**
```json
{
  "symbol": "AAPL",
  "signal": "BUY",
  "confidence": 75.50,
  "reason": "MA5 > MA20 > MA60 (uptrend) | Price above MA20 | MACD bullish crossover",
  "indicators": {
    "ma5": 150.45,
    "ma20": 148.90,
    "ma60": 145.20,
    "rsi": 62.35,
    "macd": 1.2345,
    "signal_line": 1.1234,
    "current_price": 151.75
  }
}
```

**Bearish Signal**
```json
{
  "symbol": "TSLA",
  "signal": "SELL",
  "confidence": 68.25,
  "reason": "MA5 < MA20 < MA60 (downtrend) | Price below MA20 | RSI overbought (72.50)",
  "indicators": {
    "ma5": 245.30,
    "ma20": 250.80,
    "ma60": 255.40,
    "rsi": 72.50,
    "macd": -2.3456,
    "signal_line": -1.8765,
    "current_price": 243.20
  }
}
```

**Neutral Signal**
```json
{
  "symbol": "MSFT",
  "signal": "HOLD",
  "confidence": 25.00,
  "reason": "Neutral market conditions",
  "indicators": {
    "ma5": 380.50,
    "ma20": 379.80,
    "ma60": 375.20,
    "rsi": 52.30,
    "macd": 0.5432,
    "signal_line": 0.4567,
    "current_price": 380.00
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Stock ticker symbol |
| `signal` | string | Trading signal: `BUY`, `SELL`, or `HOLD` |
| `confidence` | number | Confidence score (0-100) |
| `reason` | string | Explanation of the signal |
| `indicators` | object | Current indicator values |
| `indicators.ma5` | number | 5-day MA |
| `indicators.ma20` | number | 20-day MA |
| `indicators.ma60` | number | 60-day MA |
| `indicators.rsi` | number | Current RSI value |
| `indicators.macd` | number | Current MACD value |
| `indicators.signal_line` | number | Current MACD signal line |
| `indicators.current_price` | number | Latest closing price |

#### Prediction Logic

The AI prediction system analyzes multiple technical signals:

1. **Moving Average Trend**
   - Bullish: MA5 > MA20 > MA60
   - Bearish: MA5 < MA20 < MA60

2. **Price vs MA20**
   - Bullish: Price above MA20
   - Bearish: Price below MA20

3. **RSI Analysis**
   - Oversold (< 30): Potential buy signal
   - Overbought (> 70): Potential sell signal

4. **MACD Crossover**
   - Bullish: MACD > Signal and MACD > 0
   - Bearish: MACD < Signal and MACD < 0

**Final Signal Calculation**:
- Average all signals
- If average > 0.3: **BUY**
- If average < -0.3: **SELL**
- Otherwise: **HOLD**

Confidence score represents the strength of the signal (0-100%).

#### Status Codes

- `200 OK` - Success
- `404 Not Found` - Stock symbol not found
- `500 Internal Server Error` - Server error

---

### GET `/api/search`

Search for stocks by symbol or company name.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | Yes | Search query (minimum 1 character) |

#### Request Examples

**Search by Symbol**
```bash
curl "http://localhost:8000/api/search?q=AAPL"
```

**Search by Company Name**
```bash
curl "http://localhost:8000/api/search?q=apple"
```

**Partial Search**
```bash
curl "http://localhost:8000/api/search?q=tsm"
```

#### Response Example

```json
{
  "query": "apple",
  "count": 1,
  "results": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "market": "US"
    }
  ]
}
```

**Multiple Results**
```json
{
  "query": "tw",
  "count": 5,
  "results": [
    {
      "symbol": "2330.TW",
      "name": "Taiwan Semiconductor Manufacturing",
      "market": "TW"
    },
    {
      "symbol": "2317.TW",
      "name": "Hon Hai Precision Industry",
      "market": "TW"
    }
  ]
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Original search query |
| `count` | integer | Number of results found |
| `results` | array | Array of matching stocks |
| `results[].symbol` | string | Stock ticker symbol |
| `results[].name` | string | Company name |
| `results[].market` | string | Market identifier (`US`, `TW`, etc.) |

#### Status Codes

- `200 OK` - Success (even if no results found)
- `422 Unprocessable Entity` - Invalid query parameter
- `500 Internal Server Error` - Server error

---

## 📦 Data Models

### Stock Data Point
```typescript
{
  date: string;        // YYYY-MM-DD
  open: number;        // Opening price
  high: number;        // High price
  low: number;         // Low price
  close: number;       // Closing price
  volume: integer;     // Trading volume
}
```

### Technical Indicators Point
```typescript
{
  date: string;           // YYYY-MM-DD
  close: number;          // Closing price
  ma5: number | null;     // 5-day MA
  ma20: number | null;    // 20-day MA
  ma60: number | null;    // 60-day MA
  rsi: number | null;     // RSI (0-100)
  macd: number | null;    // MACD line
  signal: number | null;  // Signal line
  histogram: number | null; // MACD histogram
}
```

### Prediction Result
```typescript
{
  symbol: string;              // Stock symbol
  signal: "BUY" | "SELL" | "HOLD";
  confidence: number;          // 0-100
  reason: string;              // Explanation
  indicators: {
    ma5: number;
    ma20: number;
    ma60: number;
    rsi: number;
    macd: number;
    signal_line: number;
    current_price: number;
  }
}
```

### Search Result
```typescript
{
  symbol: string;    // Stock ticker
  name: string;      // Company name
  market: string;    // Market code (US, TW, etc.)
}
```

---

## ❌ Error Handling

### Error Response Format

All errors follow this consistent format:

```json
{
  "detail": "Error message description"
}
```

### HTTP Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| `200` | Success | Request completed successfully |
| `400` | Bad Request | Invalid parameters, insufficient data |
| `404` | Not Found | Stock symbol doesn't exist |
| `422` | Unprocessable Entity | Validation error (invalid period, missing query) |
| `500` | Internal Server Error | yfinance API error, network issues |

### Error Examples

**Stock Not Found (404)**
```json
{
  "detail": "No data found for symbol: INVALID"
}
```

**Insufficient Data (400)**
```json
{
  "detail": "Insufficient data for indicator calculation (need 60+ days)"
}
```

**Invalid Period (422)**
```json
{
  "detail": [
    {
      "loc": ["query", "period"],
      "msg": "string does not match regex \"^(1d|5d|1mo|3mo|6mo|1y|2y|5y|10y|ytd|max)$\"",
      "type": "value_error.str.regex"
    }
  ]
}
```

**Server Error (500)**
```json
{
  "detail": "Connection timeout: Unable to fetch data from yfinance"
}
```

---

## ⚡ Rate Limiting

**Current Status**: No rate limiting implemented

**Recommendations for Production**:
- Implement rate limiting (e.g., 100 requests/minute per IP)
- Cache frequently requested stock data
- Use background tasks for expensive calculations
- Consider Redis for distributed rate limiting

---

## 🔧 Testing with cURL

### Basic Health Check
```bash
curl http://localhost:8000/
```

### Test Stock Data
```bash
curl "http://localhost:8000/api/stock/AAPL?period=1mo"
```

### Test with Invalid Symbol
```bash
curl "http://localhost:8000/api/stock/INVALID"
```

### Test Indicators
```bash
curl "http://localhost:8000/api/stock/AAPL/indicators?period=3mo"
```

### Test Prediction
```bash
curl "http://localhost:8000/api/stock/AAPL/predict"
```

### Test Search
```bash
curl "http://localhost:8000/api/search?q=apple"
```

---

## 🧪 Testing with Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Test stock data
response = requests.get(f"{BASE_URL}/api/stock/AAPL", params={"period": "3mo"})
print(response.json())

# Test indicators
response = requests.get(f"{BASE_URL}/api/stock/AAPL/indicators")
print(response.json())

# Test prediction
response = requests.get(f"{BASE_URL}/api/stock/AAPL/predict")
prediction = response.json()
print(f"Signal: {prediction['signal']}, Confidence: {prediction['confidence']}%")

# Test search
response = requests.get(f"{BASE_URL}/api/search", params={"q": "tesla"})
print(response.json())
```

---

## 📚 Additional Resources

- **Interactive API Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc (ReDoc)
- **yfinance Documentation**: https://github.com/ranaroussi/yfinance
- **FastAPI Documentation**: https://fastapi.tiangolo.com/

---

## 🚀 Advanced Usage

### Batch Processing Example

```python
import requests
import concurrent.futures

BASE_URL = "http://localhost:8000"
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

def get_prediction(symbol):
    response = requests.get(f"{BASE_URL}/api/stock/{symbol}/predict")
    return response.json()

# Fetch predictions concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    predictions = list(executor.map(get_prediction, SYMBOLS))

# Analyze results
for pred in predictions:
    print(f"{pred['symbol']}: {pred['signal']} ({pred['confidence']}%)")
```

### Custom Time Range Analysis

```python
import requests
import pandas as pd

BASE_URL = "http://localhost:8000"

# Get indicators for different time ranges
periods = ["1mo", "3mo", "6mo", "1y"]
results = {}

for period in periods:
    response = requests.get(
        f"{BASE_URL}/api/stock/AAPL/indicators",
        params={"period": period}
    )
    data = response.json()
    
    # Get latest RSI value
    latest = data['indicators'][-1]
    results[period] = latest['rsi']

print("RSI Comparison:")
for period, rsi in results.items():
    print(f"{period}: RSI = {rsi:.2f}")
```

---

## 📞 Support

For issues or questions:
- **GitHub**: [@agBythos](https://github.com/agBythos)
- **Email**: agbythos@gmail.com
- **Fiverr**: [@agBythos](https://fiverr.com/agBythos)

---

**Last Updated**: 2026-02-16  
**API Version**: 2.0.0
