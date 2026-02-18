"""
rf_strategy.py — Random Forest 機器學習策略（獨立模組）
=======================================================

解耦自 server.py，可直接與 BacktraderEngine.run() 搭配使用。
不依賴 server.py 的全局狀態或模組層級函式。

功能：
- RandomForestPredictor：自包含的 RF 預測器（含 feature engineering）
- RFStrategy：繼承 backtrader Strategy，可插入 BacktraderEngine

用法::

    from backtest.rf_strategy import RFStrategy
    from backtest.backtrader_engine import BacktraderEngine

    engine = BacktraderEngine(symbol="2330.TW", initial_capital=100_000)
    result = engine.run(
        strategy_class=RFStrategy,
        data=df,
        strategy_params={
            "forward_days": 5,
            "confidence_threshold": 0.50,
            "retrain_period": 60,
        },
        strategy_name="Random Forest ML",
    )

設計決策：
- RandomForestPredictor 完全自包含，不依賴 server.py 的 calculate_rsi / calculate_macd 等
- RFStrategy 繼承 _BaseStrategy（來自 backtrader_engine），取得 equity curve 追蹤
- 每 retrain_period 天重新訓練（預設 60 天），與 server.py 原版邏輯一致

作者：Bythos（sub-agent）
建立：2026-02-18
"""

from __future__ import annotations

import os
import pickle
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import backtrader as bt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# 預設模型快取目錄（相對 stock-app/）
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


# ============================================================================
# 技術指標計算（自包含，不依賴 server.py）
# ============================================================================


def _calc_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """計算 RSI（Relative Strength Index）"""
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _calc_macd(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """計算 MACD"""
    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return {"macd": macd, "signal": signal, "histogram": macd - signal}


def _calc_ma(data: pd.DataFrame, period: int) -> pd.Series:
    """計算移動平均線"""
    return data["Close"].rolling(window=period).mean()


def _calc_garman_klass_vol(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    計算 Garman-Klass 波動率（滾動窗口）

    公式：GK = sqrt(1/N * sum(0.5*(log(H/L))^2 - (2*ln2-1)*(log(C/O))^2))

    Args:
        data:   DataFrame with 'Open', 'High', 'Low', 'Close' columns
        window: 滾動窗口大小（預設 20）

    Returns:
        pd.Series of Garman-Klass volatility values
    """
    log_hl = np.log(data["High"] / data["Low"])
    log_co = np.log(data["Close"] / data["Open"])
    gk_daily = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    return np.sqrt(gk_daily.rolling(window=window).mean())


# ============================================================================
# RandomForestPredictor（自包含）
# ============================================================================


class RandomForestPredictor:
    """
    Random Forest 股票方向預測器（自包含版本）

    完全獨立於 server.py，可在任何環境中使用。

    特徵集（10 個技術指標）：
    - RSI(14)
    - MACD histogram
    - MA crossover (10/30, 20/60)
    - Bollinger Band %B
    - Volume ratio (vs 20-day avg)
    - Price momentum (5d, 10d, 20d returns)
    - Volatility (20d rolling std)

    目標變數：未來 forward_days 天報酬率 > 0 → 1, else 0
    """

    FEATURE_NAMES: List[str] = [
        "rsi_14",
        "macd_hist",
        "ma_cross_10_30",
        "ma_cross_20_60",
        "bb_pct_b",
        "vol_ratio",
        "momentum_5d",
        "momentum_10d",
        "momentum_20d",
        "volatility_20d",
        "garman_klass_vol",
    ]

    def __init__(
        self,
        forward_days: int = 5,
        confidence_threshold: float = 0.55,
    ):
        """
        初始化 RandomForestPredictor

        Args:
            forward_days:          預測未來 N 天方向（預設 5）
            confidence_threshold:  信心閾值；低於此值回傳 HOLD（預設 0.55）
        """
        self.forward_days = forward_days
        self.confidence_threshold = confidence_threshold

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_trained: bool = False
        self._feature_importances: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        計算技術指標特徵

        Args:
            data: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns

        Returns:
            DataFrame containing only the 10 feature columns
        """
        df = data.copy()

        # 1. RSI(14)
        df["rsi_14"] = _calc_rsi(df, 14)

        # 2. MACD histogram
        macd_result = _calc_macd(df)
        df["macd_hist"] = macd_result["histogram"]

        # 3. MA crossover (10/30, 20/60)
        df["ma_10"] = _calc_ma(df, 10)
        df["ma_30"] = _calc_ma(df, 30)
        df["ma_20"] = _calc_ma(df, 20)
        df["ma_60"] = _calc_ma(df, 60)
        df["ma_cross_10_30"] = (df["ma_10"] - df["ma_30"]) / df["Close"]
        df["ma_cross_20_60"] = (df["ma_20"] - df["ma_60"]) / df["Close"]

        # 4. Bollinger Bands %B
        bb_mid = df["Close"].rolling(20).mean()
        bb_std = df["Close"].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df["bb_pct_b"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower)

        # 5. Volume ratio (vs 20-day avg)
        df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

        # 6. Price momentum (5d, 10d, 20d)
        df["momentum_5d"] = df["Close"].pct_change(5)
        df["momentum_10d"] = df["Close"].pct_change(10)
        df["momentum_20d"] = df["Close"].pct_change(20)

        # 7. Volatility (20d rolling std of daily returns)
        df["volatility_20d"] = df["Close"].pct_change().rolling(20).std()

        # 8. Garman-Klass Volatility (20-day window)
        df["garman_klass_vol"] = _calc_garman_klass_vol(df, window=20)

        return df[self.FEATURE_NAMES]

    def _calculate_target(self, data: pd.DataFrame) -> pd.Series:
        """計算目標：未來 N 天報酬率 > 0 → 1, else 0"""
        future_return = data["Close"].pct_change(self.forward_days).shift(-self.forward_days)
        return (future_return > 0).astype(int)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data: pd.DataFrame) -> bool:
        """
        訓練 Random Forest 模型

        Args:
            data: 歷史 OHLCV DataFrame（至少 100 行）

        Returns:
            True 若訓練成功，False 若資料不足
        """
        if len(data) < 100:
            print(f"[RF] Insufficient data for training: {len(data)} rows (need ≥ 100)")
            return False

        features = self._calculate_features(data)
        target = self._calculate_target(data)

        combined = pd.concat([features, target.rename("target")], axis=1).dropna()

        if len(combined) < 50:
            print(f"[RF] Too few samples after NaN removal: {len(combined)} (need ≥ 50)")
            return False

        X = combined[self.FEATURE_NAMES].values
        y = combined["target"].values

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        self._feature_importances = dict(
            zip(self.FEATURE_NAMES, self.model.feature_importances_)
        )

        top3 = sorted(self._feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"[RF] Trained on {len(combined)} samples. Top features: {top3}")
        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成預測信號

        若模型未訓練，自動嘗試訓練。

        Args:
            symbol: 股票代碼（用於日誌）
            data:   最新 OHLCV DataFrame

        Returns:
            dict with keys: symbol, signal, confidence, reason, probabilities
        """
        if not self.is_trained:
            ok = self.train(data)
            if not ok:
                return {
                    "symbol": symbol,
                    "signal": "HOLD",
                    "confidence": 0,
                    "reason": "Model not trained (insufficient data)",
                    "probabilities": {"up": 0.5, "down": 0.5},
                }

        features = self._calculate_features(data)
        latest = features.iloc[[-1]].dropna()

        if latest.empty or len(latest.columns) < len(self.FEATURE_NAMES):
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "reason": "Insufficient feature data",
                "probabilities": {"up": 0.5, "down": 0.5},
            }

        X = self.scaler.transform(latest[self.FEATURE_NAMES].values)
        proba = self.model.predict_proba(X)[0]
        prob_down, prob_up = float(proba[0]), float(proba[1])

        if prob_up >= self.confidence_threshold:
            signal = "BUY"
            confidence = prob_up * 100
        elif prob_down >= self.confidence_threshold:
            signal = "SELL"
            confidence = prob_down * 100
        else:
            signal = "HOLD"
            confidence = max(prob_up, prob_down) * 100

        top_features = sorted(
            self._feature_importances.items(), key=lambda x: x[1], reverse=True
        )[:3]
        reason = f"RF prediction based on {', '.join(f[0] for f in top_features)}"

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(float(confidence), 2),
            "reason": reason,
            "probabilities": {
                "up": round(float(prob_up), 3),
                "down": round(float(prob_down), 3),
            },
        }

    def feature_importance(self) -> Dict[str, float]:
        """回傳特徵重要性（需先訓練）"""
        return {k: round(float(v), 4) for k, v in self._feature_importances.items()}

    # ------------------------------------------------------------------
    # 模型持久化（pickle 快取）
    # ------------------------------------------------------------------

    @staticmethod
    def _model_path(symbol: str, model_date: Optional[str] = None, model_dir: Optional[Path] = None) -> Path:
        """
        組合模型快取路徑

        路徑規則：``models/rf_{symbol}_{date}.pkl``

        Args:
            symbol:     股票代碼（e.g. "2330.TW"，自動轉換 '.' → '_'）
            model_date: 日期字串 YYYY-MM-DD，預設今天
            model_dir:  快取目錄，預設 stock-app/models/

        Returns:
            Path 物件
        """
        if model_date is None:
            model_date = date.today().isoformat()
        if model_dir is None:
            model_dir = _DEFAULT_MODEL_DIR
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.replace(".", "_").replace("/", "_")
        return model_dir / f"rf_{safe_symbol}_{model_date}.pkl"

    def save(self, symbol: str, model_date: Optional[str] = None, model_dir: Optional[Path] = None) -> Path:
        """
        將訓練好的模型序列化至 pickle 檔案

        Args:
            symbol:     股票代碼
            model_date: 日期字串 YYYY-MM-DD，預設今天
            model_dir:  快取目錄，預設 stock-app/models/

        Returns:
            儲存路徑

        Raises:
            RuntimeError: 若模型尚未訓練
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save: model has not been trained yet.")

        path = self._model_path(symbol, model_date, model_dir)
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "forward_days": self.forward_days,
            "confidence_threshold": self.confidence_threshold,
            "feature_importances": self._feature_importances,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[RF] Model saved → {path}")
        return path

    @classmethod
    def load(cls, symbol: str, model_date: Optional[str] = None, model_dir: Optional[Path] = None) -> "RandomForestPredictor":
        """
        從 pickle 快取載入模型

        Args:
            symbol:     股票代碼
            model_date: 日期字串 YYYY-MM-DD，預設今天
            model_dir:  快取目錄，預設 stock-app/models/

        Returns:
            已恢復狀態的 RandomForestPredictor 實例

        Raises:
            FileNotFoundError: 若快取檔案不存在
        """
        path = cls._model_path(symbol, model_date, model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Model cache not found: {path}")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        instance = cls(
            forward_days=payload["forward_days"],
            confidence_threshold=payload["confidence_threshold"],
        )
        instance.model = payload["model"]
        instance.scaler = payload["scaler"]
        instance._feature_importances = payload["feature_importances"]
        instance.is_trained = True
        print(f"[RF] Model loaded ← {path}")
        return instance

    @classmethod
    def load_or_train(
        cls,
        symbol: str,
        data: pd.DataFrame,
        model_date: Optional[str] = None,
        model_dir: Optional[Path] = None,
        forward_days: int = 5,
        confidence_threshold: float = 0.55,
    ) -> "RandomForestPredictor":
        """
        便利方法：先嘗試載入快取，不存在則訓練後儲存

        Args:
            symbol:               股票代碼
            data:                 OHLCV DataFrame（用於訓練）
            model_date:           日期字串 YYYY-MM-DD，預設今天
            model_dir:            快取目錄
            forward_days:         預測天數（僅在重新訓練時生效）
            confidence_threshold: 信心閾值（僅在重新訓練時生效）

        Returns:
            RandomForestPredictor 實例（已訓練）
        """
        try:
            return cls.load(symbol, model_date, model_dir)
        except FileNotFoundError:
            print(f"[RF] Cache miss for {symbol}, training new model...")
            instance = cls(forward_days=forward_days, confidence_threshold=confidence_threshold)
            instance.train(data)
            if instance.is_trained:
                instance.save(symbol, model_date, model_dir)
            return instance


# ============================================================================
# RFStrategy — Backtrader 策略
# ============================================================================


class RFStrategy(bt.Strategy):
    """
    Random Forest 機器學習策略（解耦版）

    - 繼承 backtrader.Strategy，與 BacktraderEngine.run() 完全相容
    - 每 retrain_period 天重新訓練模型（預設 60 天）
    - 不依賴 server.py 的全局狀態或 BaseBacktestStrategy

    策略參數::

        forward_days         預測未來幾天方向（預設 5）
        confidence_threshold 信心閾值，低於此值不交易（預設 0.50）
        retrain_period       重新訓練週期（天，預設 60）

    使用範例::

        from backtest.rf_strategy import RFStrategy
        from backtest.backtrader_engine import BacktraderEngine

        engine = BacktraderEngine("2330.TW", initial_capital=500_000)
        result = engine.run(
            strategy_class=RFStrategy,
            data=ohlcv_df,
            strategy_params={"retrain_period": 60, "confidence_threshold": 0.50},
            strategy_name="Random Forest ML",
        )
    """

    params = (
        ("forward_days", 5),
        ("confidence_threshold", 0.50),
        ("retrain_period", 60),   # 每 N 天重新訓練
    )

    def __init__(self):
        self.order: Optional[bt.Order] = None
        self.predictor = RandomForestPredictor(
            forward_days=self.p.forward_days,
            confidence_threshold=self.p.confidence_threshold,
        )
        self.days_since_train: int = 0
        self.trained: bool = False

        # Equity curve 追蹤（與 _BaseStrategy 相容）
        self._equity_dates: List[str] = []
        self._equity_values: List[float] = []

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------

    def _get_history_df(self, max_bars: int = 500) -> pd.DataFrame:
        """
        將 backtrader 數據轉換為 pandas DataFrame

        Args:
            max_bars: 最多取最近幾根 K 棒（避免記憶體過大）

        Returns:
            OHLCV DataFrame，index 為 DatetimeIndex
        """
        d = self.datas[0]
        n = min(max_bars, len(d))

        indices = range(-n + 1, 1)  # 最舊 → 最新
        df = pd.DataFrame(
            {
                "Open": [d.open[i] for i in indices],
                "High": [d.high[i] for i in indices],
                "Low": [d.low[i] for i in indices],
                "Close": [d.close[i] for i in indices],
                "Volume": [d.volume[i] for i in indices],
            },
            index=pd.DatetimeIndex([d.datetime.date(i) for i in indices]),
        )
        return df

    def log(self, msg: str):
        """日誌輸出"""
        dt = self.datas[0].datetime.date(0)
        print(f"[RF {dt}] {msg}")

    # ------------------------------------------------------------------
    # Backtrader hooks
    # ------------------------------------------------------------------

    def notify_order(self, order: bt.Order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

    def next(self):
        # Equity curve 記錄
        dt = self.datas[0].datetime.date(0).isoformat()
        val = self.broker.getvalue()
        self._equity_dates.append(dt)
        self._equity_values.append(val)

        if self.order:
            return

        # ── 重新訓練判斷 ──────────────────────────────────────────────
        self.days_since_train += 1
        need_train = (not self.trained) or (self.days_since_train >= self.p.retrain_period)

        if need_train:
            hist_df = self._get_history_df(max_bars=500)
            ok = self.predictor.train(hist_df)
            if ok:
                self.trained = True
                self.days_since_train = 0
                self.log(f"Model retrained on {len(hist_df)} bars")

        if not self.trained:
            return

        # ── 生成預測 ──────────────────────────────────────────────────
        # 只取最近 252 根 K 棒做 feature 計算（提升速度）
        pred_df = self._get_history_df(max_bars=252)
        prediction = self.predictor.predict("STOCK", pred_df)
        signal = prediction.get("signal", "HOLD")
        confidence = prediction.get("confidence", 0)

        # ── 執行交易 ──────────────────────────────────────────────────
        if not self.position:
            if signal == "BUY":
                size = int(self.broker.get_cash() * 0.95 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    self.log(f"BUY {size} @ {self.data.close[0]:.2f} (conf={confidence:.2f}%)")
        else:
            if signal == "SELL":
                self.log(f"SELL {self.position.size} @ {self.data.close[0]:.2f} (conf={confidence:.2f}%)")
                self.order = self.sell(size=self.position.size)
