"""
Walk-Forward 驗證功能測試腳本
測試 2330.TW (台積電) 2023-2024 數據
"""

import requests
import json
from datetime import datetime

# API 基礎 URL
BASE_URL = "http://localhost:8001"

def test_walk_forward():
    """測試 Walk-Forward 驗證端點"""
    
    print("="*60)
    print("Walk-Forward Validation Test")
    print("="*60)
    
    # 測試請求
    payload = {
        "symbol": "2330.TW",
        "strategy": "ma_crossover",
        "total_start": "2023-01-01",
        "total_end": "2024-12-31",
        "train_months": 6,
        "test_months": 1,
        "initial_capital": 100000,
        "parameters": {
            "fast_period": 10,
            "slow_period": 30
        }
    }
    
    print("\n請求參數:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    print("\n發送請求...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/backtest/walk-forward",
            json=payload,
            timeout=180  # 3分鐘超時
        )
        
        print(f"狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n"+"="*60)
            print("Walk-Forward 驗證結果")
            print("="*60)
            
            # 配置資訊
            config = result.get('config', {})
            print(f"\n【配置】")
            print(f"  股票代碼: {result.get('symbol')}")
            print(f"  總時間範圍: {config['total_period']['start']} ~ {config['total_period']['end']}")
            print(f"  訓練窗口: {config['train_months']} 個月")
            print(f"  測試窗口: {config['test_months']} 個月")
            print(f"  初始資金: ${config['initial_capital']:,.0f}")
            
            # 彙總績效
            summary = result.get('summary', {})
            print(f"\n【彙總績效】")
            print(f"  總窗口數: {summary['total_windows']}")
            print(f"  成功窗口數: {summary['successful_windows']}")
            print(f"  平均報酬: {summary['average_return']:.2f}%")
            print(f"  中位數報酬: {summary['median_return']:.2f}%")
            print(f"  最佳報酬: {summary['best_return']:.2f}%")
            print(f"  最差報酬: {summary['worst_return']:.2f}%")
            print(f"  平均 Sharpe Ratio: {summary['average_sharpe']:.2f}")
            print(f"  平均最大回撤: {summary['average_max_drawdown']:.2f}%")
            print(f"  總交易次數: {summary['total_trades']}")
            print(f"  平均每窗口交易: {summary['average_trades_per_window']:.1f}")
            
            # 穩定性指標
            print(f"\n【穩定性指標】")
            print(f"  報酬標準差: {summary['return_std']:.2f}% (越低越穩定)")
            print(f"  Sharpe 標準差: {summary['sharpe_std']:.2f} (越低越穩定)")
            
            # 計算穩定性評級
            if summary['return_std'] < 5:
                stability = "極佳"
            elif summary['return_std'] < 10:
                stability = "良好"
            elif summary['return_std'] < 20:
                stability = "中等"
            else:
                stability = "不穩定"
            
            print(f"  穩定性評級: {stability}")
            
            # 窗口明細
            windows = result.get('windows', [])
            print(f"\n【窗口明細】(前 5 個窗口)")
            print(f"{'窗口':<6} {'訓練期':<25} {'測試期':<25} {'報酬%':<10} {'Sharpe':<10} {'交易':<8}")
            print("-"*90)
            
            for window in windows[:5]:
                print(f"{window['window_id']:<6} "
                      f"{window['train_period']:<25} "
                      f"{window['test_period']:<25} "
                      f"{window['total_return']:<10.2f} "
                      f"{window['sharpe_ratio']:<10.2f} "
                      f"{window['total_trades']:<8}")
            
            if len(windows) > 5:
                print(f"... (還有 {len(windows) - 5} 個窗口)")
            
            print("\n"+"="*60)
            print("[SUCCESS] Walk-Forward 驗證完成")
            print("="*60)
            
            # 保存完整結果
            with open('walk_forward_result.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print("\n完整結果已保存至: walk_forward_result.json")
            
        else:
            print(f"\n錯誤: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("\n錯誤: 請求超時（超過 180 秒）")
    except requests.exceptions.ConnectionError:
        print(f"\n錯誤: 無法連接到伺服器 {BASE_URL}")
        print("請確認伺服器是否正在運行: python stock-app/server.py")
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()


def test_strategies_endpoint():
    """測試策略列表端點"""
    print("\n測試策略列表端點...")
    try:
        response = requests.get(f"{BASE_URL}/api/backtest/strategies")
        if response.status_code == 200:
            strategies = response.json()
            print(f"[OK] 可用策略數: {strategies['count']}")
            for s in strategies['strategies']:
                print(f"  - {s['name']}: {s['display_name']}")
        else:
            print(f"[ERROR] 錯誤: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] 錯誤: {e}")


if __name__ == "__main__":
    # 先測試策略列表
    test_strategies_endpoint()
    
    # 再測試 Walk-Forward 驗證
    test_walk_forward()
