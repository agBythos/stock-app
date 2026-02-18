import json
with open("walk_forward_e2e_result.json", "r") as f:
    data = json.load(f)

for fold in data["folds"]:
    skip = "(SKIPPED)" if fold["skipped"] else ""
    ml = fold["ml"]
    bt = fold["backtest"]
    fid = fold["fold_id"]
    tb = fold["train_bars"]
    teb = fold["test_bars"]
    acc = ml["ensemble_accuracy"]
    roc = ml["ensemble_roc_auc"]
    ret = bt["total_return_pct"]
    sharpe = bt["sharpe_ratio"]
    dd = bt["max_drawdown_pct"]
    wr = bt["win_rate_pct"]
    trades = bt["total_trades"]
    print(f"Fold {fid:2d}: {skip:10s} train={tb}b test={teb}b | acc={acc:.4f} roc={roc:.4f} | ret={ret:+.4f}% sharpe={sharpe:.4f} dd={dd:.4f}% wr={wr:.1f}% trades={trades}")
