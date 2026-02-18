"""
Phase 3.5 端對端測試腳本
POST /api/validate/walk-forward
"""
import requests
import json
import time
import sys

print("Testing POST /api/validate/walk-forward ...")
print("Symbol: 2330.TW, period=2y")
print()

start = time.time()
try:
    resp = requests.post(
        "http://localhost:8000/api/validate/walk-forward",
        params={
            "symbol": "2330.TW",
            "period": "2y",
            "train_window": 252,
            "test_window": 21,
            "label_horizon": 5,
            "embargo_bars": 5,
        },
        timeout=300
    )
    elapsed = time.time() - start
    print(f"Status: {resp.status_code}")
    print(f"Time: {elapsed:.1f}s")
    print()

    if resp.status_code == 200:
        data = resp.json()
        print("=== RESPONSE SUMMARY ===")
        print(f"symbol: {data.get('symbol')}")
        print(f"run_timestamp: {data.get('run_timestamp')}")
        print(f"folds count: {len(data.get('folds', []))}")
        print()
        print("=== CONFIG ===")
        cfg = data.get("config", {})
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        print()
        print("=== SUMMARY ===")
        summary = data.get("summary", {})
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print()
        print("=== FOLDS (first 3) ===")
        for fold in data.get("folds", [])[:3]:
            print(f"  fold {fold['fold_id']}: train={fold['train_bars']}b test={fold['test_bars']}b")
            ml = fold.get("ml", {})
            print(f"    ML: acc={ml.get('ensemble_accuracy')} roc_auc={ml.get('ensemble_roc_auc')}")
            bt = fold.get("backtest", {})
            print(f"    BT: return={bt.get('total_return_pct')}% sharpe={bt.get('sharpe_ratio')} dd={bt.get('max_drawdown_pct')}% win={bt.get('win_rate_pct')}% trades={bt.get('total_trades')}")
            print(f"    skipped={fold.get('skipped')}")

        # Save full response
        with open("walk_forward_e2e_result.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print()
        print("Full response saved to walk_forward_e2e_result.json")
        print()
        print("TEST PASSED")
    else:
        print("ERROR RESPONSE:")
        print(resp.text[:2000])
        sys.exit(1)

except requests.exceptions.Timeout:
    print("TIMEOUT after 300s")
    sys.exit(1)
except Exception as e:
    import traceback
    print(f"EXCEPTION: {e}")
    traceback.print_exc()
    sys.exit(1)
