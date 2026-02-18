"""
stock-app/cache/model_cache.py
==============================

ModelCache — pickle-based model persistence layer for RandomForest and HMM models.

Design goals:
- Avoid re-training expensive ML models on every API call (RF/HMM training ~60s)
- Simple date-based TTL (max_age_days); default 1 day so models refresh daily
  with fresh market data
- Generic: can store any pickle-able Python object (trained predictors, result
  dicts, raw sklearn/hmmlearn models, etc.)
- Thread-safe for concurrent FastAPI requests (file write is atomic via
  temp-file rename pattern)

Cache path convention:
    <cache_dir>/models/{symbol}_{model_type}_{YYYY-MM-DD}.pkl

Metadata saved alongside every model:
    {
        "symbol":       str,
        "model_type":   str,
        "saved_at":     ISO-8601 datetime string,
        "date":         YYYY-MM-DD string,
        "version":      str,
        "features":     List[str] or None,
        **extra_metadata,  # caller-supplied
    }

Author: Bythos (sub-agent)
Created: 2026-02-18
"""

from __future__ import annotations

import glob
import os
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Default cache root ─────────────────────────────────────────────────────
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent  # stock-app/cache/
_DEFAULT_MODELS_DIR = _DEFAULT_CACHE_DIR / "models"

# ── Current schema version (bump when payload format changes) ──────────────
_CACHE_VERSION = "1.0.0"


class ModelCache:
    """
    Pickle-based model persistence cache.

    Parameters
    ----------
    cache_dir : Path | str | None
        Root directory for cache files.
        Defaults to ``stock-app/cache/models/``.

    Example
    -------
    >>> cache = ModelCache()
    >>> cache.save_model("AAPL", "rf", predictor, {"features": ["rsi_14"]})
    >>> obj = cache.load_model("AAPL", "rf", max_age_days=1)
    >>> cache.is_fresh("AAPL", "rf")
    True
    >>> cache.clear_cache("AAPL")
    """

    def __init__(self, cache_dir: Optional[Path | str] = None) -> None:
        if cache_dir is None:
            self._cache_dir = _DEFAULT_MODELS_DIR
        else:
            self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Path helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        """Sanitise symbol for use in a filename (replace '.' and '/')."""
        return symbol.replace(".", "_").replace("/", "_").upper()

    def _model_path(self, symbol: str, model_type: str, date_str: str) -> Path:
        """Return the canonical .pkl path for (symbol, model_type, date)."""
        safe_sym = self._safe_symbol(symbol)
        filename = f"{safe_sym}_{model_type}_{date_str}.pkl"
        return self._cache_dir / filename

    def _glob_paths(self, symbol: str, model_type: str) -> List[Path]:
        """Return all cache files matching (symbol, model_type), any date."""
        safe_sym = self._safe_symbol(symbol)
        pattern = str(self._cache_dir / f"{safe_sym}_{model_type}_*.pkl")
        return sorted(Path(p) for p in glob.glob(pattern))

    # ── Core API ───────────────────────────────────────────────────────────

    def save_model(
        self,
        symbol: str,
        model_type: str,
        model_obj: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Serialise ``model_obj`` to disk with accompanying metadata.

        The file is written atomically via a temp-file rename to prevent
        partial writes from being read by concurrent requests.

        Parameters
        ----------
        symbol : str
            Stock ticker symbol (e.g. "AAPL", "2330.TW").
        model_type : str
            Logical model identifier: "rf" | "hmm" | "hmm_filter" | …
        model_obj : Any
            Any pickle-able Python object.
        metadata : dict | None
            Extra key/value pairs stored alongside the model
            (e.g. ``{"features": ["rsi_14", ...], "n_samples": 500}``).

        Returns
        -------
        Path
            Absolute path of the saved .pkl file.
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        dest = self._model_path(symbol, model_type, today)

        payload = {
            "model": model_obj,
            "metadata": {
                "symbol": symbol,
                "model_type": model_type,
                "saved_at": datetime.utcnow().isoformat(),
                "date": today,
                "version": _CACHE_VERSION,
                "features": None,
                **(metadata or {}),
            },
        }

        # Atomic write: write to temp file, then rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._cache_dir, suffix=".pkl.tmp"
        )
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            # atomic rename (POSIX) / replace (Windows)
            Path(tmp_path).replace(dest)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        print(f"[ModelCache] Saved {model_type} for {symbol} → {dest.name}")
        return dest

    def load_model(
        self,
        symbol: str,
        model_type: str,
        max_age_days: int = 1,
    ) -> Optional[Any]:
        """
        Load a cached model if one exists and is not older than *max_age_days*.

        Returns ``None`` if:
        - No cache file exists for (symbol, model_type)
        - The most recent cache file is older than max_age_days

        Parameters
        ----------
        symbol : str
        model_type : str
        max_age_days : int
            Maximum acceptable age in calendar days (default 1).

        Returns
        -------
        Any | None
            The ``model_obj`` that was passed to ``save_model``, or ``None``.
        """
        paths = self._glob_paths(symbol, model_type)
        if not paths:
            print(f"[ModelCache] No cache for {symbol}/{model_type}")
            return None

        # Take the most recent file (glob returns sorted by name → date)
        latest = paths[-1]

        if not self._path_is_fresh(latest, max_age_days):
            print(
                f"[ModelCache] Stale cache for {symbol}/{model_type}: {latest.name}"
            )
            return None

        try:
            with open(latest, "rb") as fh:
                payload = pickle.load(fh)
            print(
                f"[ModelCache] Loaded {model_type} for {symbol} ← {latest.name}"
            )
            return payload["model"]
        except (pickle.UnpicklingError, KeyError, EOFError) as exc:
            print(f"[ModelCache] Corrupt cache {latest.name}: {exc}")
            return None

    def load_model_with_metadata(
        self,
        symbol: str,
        model_type: str,
        max_age_days: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Like ``load_model`` but returns ``{"model": obj, "metadata": dict}``
        instead of just the model object.

        Returns ``None`` if cache is missing or stale.
        """
        paths = self._glob_paths(symbol, model_type)
        if not paths:
            return None

        latest = paths[-1]
        if not self._path_is_fresh(latest, max_age_days):
            return None

        try:
            with open(latest, "rb") as fh:
                return pickle.load(fh)
        except (pickle.UnpicklingError, KeyError, EOFError):
            return None

    def is_fresh(
        self,
        symbol: str,
        model_type: str,
        max_age_days: int = 1,
    ) -> bool:
        """
        Return ``True`` if a fresh cache exists for (symbol, model_type).

        Parameters
        ----------
        symbol : str
        model_type : str
        max_age_days : int

        Returns
        -------
        bool
        """
        paths = self._glob_paths(symbol, model_type)
        if not paths:
            return False
        return self._path_is_fresh(paths[-1], max_age_days)

    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """
        Delete cached model files.

        Parameters
        ----------
        symbol : str | None
            If given, only delete files for that symbol.
            If ``None``, delete *all* .pkl files in the cache directory.

        Returns
        -------
        int
            Number of files deleted.
        """
        if symbol is None:
            pattern = str(self._cache_dir / "*.pkl")
        else:
            safe_sym = self._safe_symbol(symbol)
            pattern = str(self._cache_dir / f"{safe_sym}_*.pkl")

        files = glob.glob(pattern)
        deleted = 0
        for f in files:
            try:
                os.unlink(f)
                deleted += 1
            except OSError as exc:
                print(f"[ModelCache] Could not delete {f}: {exc}")

        print(f"[ModelCache] Cleared {deleted} file(s) "
              f"(symbol={symbol or 'ALL'})")
        return deleted

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _path_is_fresh(path: Path, max_age_days: int) -> bool:
        """
        Check if *path* is younger than *max_age_days* days.

        Uses the embedded date in the filename (``..._YYYY-MM-DD.pkl``) as
        the primary check (robust to file-system timestamp quirks), with the
        OS mtime as a fallback when the filename date cannot be parsed.
        """
        # ── Primary: parse date from filename ─────────────────────────────
        stem = path.stem  # e.g. "AAPL_rf_2026-02-18"
        parts = stem.rsplit("_", 3)  # split from right, at most 3 times
        # The last three parts should be YYYY, MM, DD
        if len(parts) >= 4:
            date_str = "_".join(parts[-3:])  # "2026_02_18" → won't parse
        # Try extracting 10-char date from the end
        if len(stem) >= 10:
            date_candidate = stem[-10:]  # "2026-02-18"
            try:
                file_date = datetime.strptime(date_candidate, "%Y-%m-%d")
                age = (datetime.utcnow() - file_date).days
                return age < max_age_days
            except ValueError:
                pass

        # ── Fallback: OS mtime ─────────────────────────────────────────────
        mtime = datetime.utcfromtimestamp(path.stat().st_mtime)
        age_seconds = (datetime.utcnow() - mtime).total_seconds()
        return age_seconds < max_age_days * 86400

    def get_cache_info(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return a list of dicts describing all cache files.

        Useful for debugging or admin endpoints.
        """
        if symbol is None:
            pattern = str(self._cache_dir / "*.pkl")
        else:
            safe_sym = self._safe_symbol(symbol)
            pattern = str(self._cache_dir / f"{safe_sym}_*.pkl")

        info = []
        for f in sorted(glob.glob(pattern)):
            p = Path(f)
            try:
                stat = p.stat()
                mtime = datetime.utcfromtimestamp(stat.st_mtime).isoformat()
                size_kb = stat.st_size / 1024
                fresh = self._path_is_fresh(p, max_age_days=1)
            except OSError:
                continue
            info.append(
                {
                    "filename": p.name,
                    "size_kb": round(size_kb, 1),
                    "mtime": mtime,
                    "fresh": fresh,
                }
            )
        return info
