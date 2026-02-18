"""
tests/test_model_cache.py
=========================

Unit tests for stock-app/cache/model_cache.py

Test coverage:
  1. save_model + load_model round-trip (basic object)
  2. save_model + load_model round-trip (complex nested dict)
  3. save_model + load_model round-trip (sklearn estimator-like object)
  4. Metadata is preserved and accessible via load_model_with_metadata
  5. Multiple symbols don't interfere with each other
  6. max_age_days=0 makes every cache immediately stale
  7. load_model returns None when no file exists
  8. load_model returns None when max_age_days < age of file (mtime-based)
  9. clear_cache(symbol) removes only that symbol's files
 10. clear_cache() removes all files

Author: Bythos (sub-agent)
Created: 2026-02-18
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# ── Make sure the stock-app/ directory is on sys.path ─────────────────────
_STOCK_APP_DIR = Path(__file__).resolve().parent.parent
if str(_STOCK_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_STOCK_APP_DIR))

from cache.model_cache import ModelCache


# ── Module-level helper class (pickle requires top-level pickleable classes) ─

class _FakePredictor:
    """Module-level fake predictor — must be at module level for pickle."""

    def __init__(self):
        self.is_trained = True
        self.feature_importances_ = {"rsi_14": 0.25, "macd_hist": 0.15}
        self.n_estimators = 100


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_cache(tmp_path):
    """Return a ModelCache instance pointing at a throwaway temp directory."""
    return ModelCache(cache_dir=tmp_path)


# ── Helper ────────────────────────────────────────────────────────────────


def _write_stale_pkl(cache: ModelCache, symbol: str, model_type: str, days_old: int):
    """
    Write a pickle file with an mtime set *days_old* days in the past.
    Used to simulate an expired cache entry.
    """
    # Build the path that ModelCache would use for "today"
    today = datetime.utcnow().strftime("%Y-%m-%d")
    dest = cache._model_path(symbol, model_type, today)
    payload = {"model": {"stale": True}, "metadata": {}}
    with open(dest, "wb") as fh:
        pickle.dump(payload, fh)

    # Backdate the mtime
    past_ts = time.time() - (days_old * 86400 + 3600)
    os.utime(dest, (past_ts, past_ts))

    # Also rename the file to have the old date in the stem so the
    # filename-based freshness check also reports it as stale.
    old_date = (datetime.utcnow() - timedelta(days=days_old)).strftime("%Y-%m-%d")
    old_path = cache._model_path(symbol, model_type, old_date)
    dest.rename(old_path)
    return old_path


# ══════════════════════════════════════════════════════════════════════════
# Test Group 1: Save / Load round-trips
# ══════════════════════════════════════════════════════════════════════════


class TestSaveLoadRoundTrip:
    """Round-trip correctness tests (at least 5 test cases)."""

    def test_01_basic_string_object(self, tmp_cache):
        """TC-01: Save a plain string, load it back unchanged."""
        obj = "hello model cache"
        tmp_cache.save_model("AAPL", "rf", obj)
        loaded = tmp_cache.load_model("AAPL", "rf", max_age_days=1)
        assert loaded == obj, f"Expected {obj!r}, got {loaded!r}"

    def test_02_complex_dict_result(self, tmp_cache):
        """TC-02: Save a complex nested dict (simulating backtest result)."""
        obj = {
            "symbol": "2330.TW",
            "performance": {
                "total_return_pct": 23.5,
                "max_drawdown_pct": -8.2,
                "sharpe_ratio": 1.42,
                "total_trades": 17,
            },
            "equity_curve": [100000, 102000, 101500, 104000],
            "trades": [{"date": "2024-01-15", "type": "BUY", "price": 520.0}],
        }
        tmp_cache.save_model("2330.TW", "rf", obj, metadata={"strategy": "rf"})
        loaded = tmp_cache.load_model("2330.TW", "rf", max_age_days=1)
        assert loaded == obj
        assert loaded["performance"]["sharpe_ratio"] == 1.42

    def test_03_sklearn_like_object(self, tmp_cache):
        """TC-03: Save an object with sklearn-like attributes (simulated)."""
        predictor = _FakePredictor()
        tmp_cache.save_model("MSFT", "rf", predictor, metadata={"version": "v1"})
        loaded = tmp_cache.load_model("MSFT", "rf", max_age_days=1)
        assert isinstance(loaded, _FakePredictor)
        assert loaded.is_trained is True
        assert loaded.n_estimators == 100
        assert loaded.feature_importances_["rsi_14"] == pytest.approx(0.25)

    def test_04_metadata_preserved(self, tmp_cache):
        """TC-04: Metadata is stored and retrievable via load_model_with_metadata."""
        features = ["rsi_14", "macd_hist", "bb_pct_b"]
        obj = {"model_type": "hmm", "n_states": 3}
        meta = {"features": features, "n_samples": 500, "version": "v2"}

        tmp_cache.save_model("AAPL", "hmm", obj, metadata=meta)
        result = tmp_cache.load_model_with_metadata("AAPL", "hmm", max_age_days=1)

        assert result is not None
        assert result["model"] == obj
        assert result["metadata"]["features"] == features
        assert result["metadata"]["n_samples"] == 500
        assert result["metadata"]["version"] == "v2"
        # Standard metadata fields
        assert result["metadata"]["symbol"] == "AAPL"
        assert result["metadata"]["model_type"] == "hmm"
        assert "saved_at" in result["metadata"]

    def test_05_list_object(self, tmp_cache):
        """TC-05: Save a list (equity curve), load it back."""
        obj = [100000.0, 102500.0, 98750.0, 105000.0, 110250.0]
        tmp_cache.save_model("GOOG", "rf", obj)
        loaded = tmp_cache.load_model("GOOG", "rf", max_age_days=1)
        assert loaded == obj
        assert len(loaded) == 5

    def test_06_overwrite_existing_cache(self, tmp_cache):
        """TC-06: Saving a second time should overwrite (most-recent wins)."""
        tmp_cache.save_model("AAPL", "rf", {"v": 1})
        tmp_cache.save_model("AAPL", "rf", {"v": 2})
        loaded = tmp_cache.load_model("AAPL", "rf", max_age_days=1)
        assert loaded == {"v": 2}, f"Expected v=2, got {loaded}"

    def test_07_none_values_in_dict(self, tmp_cache):
        """TC-07: Model obj containing None values survives round-trip."""
        obj = {"signal": None, "confidence": 0, "reason": None}
        tmp_cache.save_model("TSLA", "rf", obj)
        loaded = tmp_cache.load_model("TSLA", "rf", max_age_days=1)
        assert loaded == obj
        assert loaded["signal"] is None


# ══════════════════════════════════════════════════════════════════════════
# Test Group 2: max_age_days expiry logic
# ══════════════════════════════════════════════════════════════════════════


class TestMaxAgeDays:
    """Cache freshness / TTL tests."""

    def test_fresh_cache_returns_model(self, tmp_cache):
        """A cache saved today is returned with max_age_days=1."""
        tmp_cache.save_model("AAPL", "rf", {"ok": True})
        loaded = tmp_cache.load_model("AAPL", "rf", max_age_days=1)
        assert loaded is not None
        assert loaded["ok"] is True

    def test_zero_max_age_is_always_stale(self, tmp_cache):
        """max_age_days=0 means the cache is immediately stale."""
        tmp_cache.save_model("AAPL", "rf", {"ok": True})
        loaded = tmp_cache.load_model("AAPL", "rf", max_age_days=0)
        assert loaded is None

    def test_stale_file_by_filename_date(self, tmp_cache):
        """A file with a 2-day-old date in its filename is stale (max_age_days=1)."""
        _write_stale_pkl(tmp_cache, "AAPL", "rf", days_old=2)
        loaded = tmp_cache.load_model("AAPL", "rf", max_age_days=1)
        assert loaded is None

    def test_is_fresh_true_for_today(self, tmp_cache):
        """is_fresh returns True when cache was saved today."""
        tmp_cache.save_model("AAPL", "rf", {"ok": True})
        assert tmp_cache.is_fresh("AAPL", "rf", max_age_days=1) is True

    def test_is_fresh_false_when_no_cache(self, tmp_cache):
        """is_fresh returns False when no cache exists."""
        assert tmp_cache.is_fresh("NONEXISTENT", "rf", max_age_days=1) is False

    def test_is_fresh_false_for_stale_file(self, tmp_cache):
        """is_fresh returns False for an old file."""
        _write_stale_pkl(tmp_cache, "AAPL", "hmm", days_old=3)
        assert tmp_cache.is_fresh("AAPL", "hmm", max_age_days=1) is False


# ══════════════════════════════════════════════════════════════════════════
# Test Group 3: load_model returns None when missing
# ══════════════════════════════════════════════════════════════════════════


class TestLoadModelMissing:
    """Tests for missing-cache behaviour."""

    def test_returns_none_no_file(self, tmp_cache):
        """load_model returns None when no pkl exists for symbol."""
        result = tmp_cache.load_model("ZZZDNE", "rf", max_age_days=1)
        assert result is None

    def test_returns_none_wrong_model_type(self, tmp_cache):
        """Saving as 'rf' does not pollute 'hmm' namespace."""
        tmp_cache.save_model("AAPL", "rf", {"x": 1})
        result = tmp_cache.load_model("AAPL", "hmm", max_age_days=1)
        assert result is None

    def test_returns_none_wrong_symbol(self, tmp_cache):
        """Saving for AAPL does not make MSFT cache available."""
        tmp_cache.save_model("AAPL", "rf", {"x": 1})
        result = tmp_cache.load_model("MSFT", "rf", max_age_days=1)
        assert result is None

    def test_load_metadata_returns_none_no_file(self, tmp_cache):
        """load_model_with_metadata also returns None when no file."""
        result = tmp_cache.load_model_with_metadata("ZZZDNE", "hmm")
        assert result is None


# ══════════════════════════════════════════════════════════════════════════
# Test Group 4: clear_cache
# ══════════════════════════════════════════════════════════════════════════


class TestClearCache:
    """Tests for clear_cache behaviour."""

    def test_clear_specific_symbol(self, tmp_cache):
        """clear_cache('AAPL') removes only AAPL files."""
        tmp_cache.save_model("AAPL", "rf", {"a": 1})
        tmp_cache.save_model("AAPL", "hmm", {"a": 2})
        tmp_cache.save_model("MSFT", "rf", {"b": 3})

        deleted = tmp_cache.clear_cache("AAPL")
        assert deleted == 2

        # AAPL gone
        assert tmp_cache.load_model("AAPL", "rf") is None
        assert tmp_cache.load_model("AAPL", "hmm") is None
        # MSFT untouched
        assert tmp_cache.load_model("MSFT", "rf") is not None

    def test_clear_all(self, tmp_cache):
        """clear_cache() with no symbol removes everything."""
        tmp_cache.save_model("AAPL", "rf", {"a": 1})
        tmp_cache.save_model("MSFT", "hmm", {"b": 2})
        tmp_cache.save_model("GOOG", "rf", {"c": 3})

        deleted = tmp_cache.clear_cache()
        assert deleted == 3

        assert tmp_cache.load_model("AAPL", "rf") is None
        assert tmp_cache.load_model("MSFT", "hmm") is None
        assert tmp_cache.load_model("GOOG", "rf") is None

    def test_clear_empty_cache(self, tmp_cache):
        """clear_cache on an empty directory returns 0 deleted."""
        deleted = tmp_cache.clear_cache()
        assert deleted == 0

    def test_clear_nonexistent_symbol(self, tmp_cache):
        """clear_cache('ZZZDNE') when symbol has no files returns 0."""
        tmp_cache.save_model("AAPL", "rf", {"x": 1})
        deleted = tmp_cache.clear_cache("ZZZDNE")
        assert deleted == 0
        # AAPL still intact
        assert tmp_cache.load_model("AAPL", "rf") is not None


# ══════════════════════════════════════════════════════════════════════════
# Test Group 5: Multi-symbol isolation
# ══════════════════════════════════════════════════════════════════════════


class TestMultiSymbolIsolation:
    """Different symbols must not interfere with each other."""

    def test_symbols_isolated(self, tmp_cache):
        """Saving different symbols produces different files."""
        tmp_cache.save_model("AAPL", "rf", {"sym": "AAPL"})
        tmp_cache.save_model("TSLA", "rf", {"sym": "TSLA"})
        tmp_cache.save_model("2330.TW", "rf", {"sym": "2330"})

        assert tmp_cache.load_model("AAPL", "rf")["sym"] == "AAPL"
        assert tmp_cache.load_model("TSLA", "rf")["sym"] == "TSLA"
        assert tmp_cache.load_model("2330.TW", "rf")["sym"] == "2330"

    def test_dot_in_symbol_sanitised(self, tmp_cache):
        """Symbols with dots (e.g. 2330.TW) are sanitised for filenames."""
        tmp_cache.save_model("2330.TW", "rf", {"ok": True})
        files = list(tmp_cache._cache_dir.glob("*.pkl"))
        assert len(files) == 1
        # Dot should be replaced by underscore in filename
        assert "." not in files[0].stem, f"Dot found in filename: {files[0].name}"

    def test_get_cache_info(self, tmp_cache):
        """get_cache_info returns correct info for all saved files."""
        tmp_cache.save_model("AAPL", "rf", {"x": 1})
        tmp_cache.save_model("MSFT", "hmm", {"y": 2})
        info = tmp_cache.get_cache_info()
        assert len(info) == 2
        assert all("filename" in i and "size_kb" in i and "fresh" in i for i in info)
        assert all(i["fresh"] is True for i in info)
