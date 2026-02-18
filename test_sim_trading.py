"""
模擬交易系統完整測試腳本
測試所有 API endpoints 和場景
"""

import requests
import json
import time
import pytest

BASE_URL = "http://localhost:8877"


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def account_id():
    """建立測試帳戶1，yield 其 ID，測試結束後刪除。"""
    resp = requests.post(
        f"{BASE_URL}/api/sim/accounts",
        json={"name": "pytest-帳戶1", "initial_cash": 100000},
    )
    resp.raise_for_status()
    aid = resp.json()["id"]
    yield aid
    # cleanup
    requests.delete(f"{BASE_URL}/api/sim/accounts/{aid}")


@pytest.fixture
def account2_id():
    """建立測試帳戶2，yield 其 ID，測試結束後刪除。"""
    resp = requests.post(
        f"{BASE_URL}/api/sim/accounts",
        json={"name": "pytest-帳戶2", "initial_cash": 200000},
    )
    resp.raise_for_status()
    aid = resp.json()["id"]
    yield aid
    # cleanup
    requests.delete(f"{BASE_URL}/api/sim/accounts/{aid}")

def print_section(title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")

def test_create_account():
    """測試場景1：建立帳戶"""
    print_section("測試1：建立帳戶")
    
    # 建立第一個帳戶
    data = {
        "name": "測試帳戶1",
        "initial_cash": 100000
    }
    
    response = requests.post(f"{BASE_URL}/api/sim/accounts", json=data)
    print(f"建立帳戶1：{response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    assert response.status_code == 201, "建立帳戶失敗"
    account1_id = response.json()["id"]
    
    # 建立第二個帳戶（驗證多帳戶隔離）
    data2 = {
        "name": "測試帳戶2",
        "initial_cash": 200000
    }
    
    response2 = requests.post(f"{BASE_URL}/api/sim/accounts", json=data2)
    print(f"\n建立帳戶2：{response2.status_code}")
    print(json.dumps(response2.json(), indent=2, ensure_ascii=False))
    
    assert response2.status_code == 201, "建立帳戶2失敗"
    account2_id = response2.json()["id"]
    
    print("\n[OK] 測試1通過：成功建立2個帳戶")
    return account1_id, account2_id


def test_list_accounts():
    """測試場景2：列出所有帳戶"""
    print_section("測試2：列出所有帳戶")
    
    response = requests.get(f"{BASE_URL}/api/sim/accounts")
    print(f"列出帳戶：{response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    assert response.status_code == 200, "列出帳戶失敗"
    assert response.json()["count"] >= 2, "帳戶數量不符"
    
    print("\n[OK] 測試2通過：成功列出所有帳戶")


def test_buy_stock(account_id):
    """測試場景3：買入股票"""
    print_section(f"測試3：帳戶 {account_id} 買入 AAPL 10股")
    
    trade_data = {
        "symbol": "AAPL",
        "action": "buy",
        "quantity": 10
    }
    
    response = requests.post(f"{BASE_URL}/api/sim/accounts/{account_id}/trade", json=trade_data)
    print(f"買入交易：{response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    assert response.status_code == 200, "買入失敗"
    assert response.json()["success"] == True, "買入交易未成功"
    
    print("\n[OK] 測試3通過：成功買入 AAPL 10股")


def test_account_details(account_id):
    """測試場景4：查詢帳戶詳情（含持倉和淨值）"""
    print_section(f"測試4：查詢帳戶 {account_id} 詳情")
    
    response = requests.get(f"{BASE_URL}/api/sim/accounts/{account_id}")
    print(f"帳戶詳情：{response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    assert response.status_code == 200, "查詢帳戶詳情失敗"
    account = response.json()
    
    assert len(account["positions"]) > 0, "持倉為空"
    assert account["net_value"] > 0, "淨值應大於0"
    
    print(f"\n當前淨值：${account['net_value']:,.2f}")
    print(f"總損益：${account['total_pnl']:,.2f} ({account['total_pnl_percent']}%)")
    
    print("\n[OK] 測試4通過：成功查詢帳戶詳情")


def test_sell_partial(account_id):
    """測試場景5：部分賣出"""
    print_section(f"測試5：帳戶 {account_id} 賣出 AAPL 5股")
    
    trade_data = {
        "symbol": "AAPL",
        "action": "sell",
        "quantity": 5
    }
    
    response = requests.post(f"{BASE_URL}/api/sim/accounts/{account_id}/trade", json=trade_data)
    print(f"賣出交易：{response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    assert response.status_code == 200, "賣出失敗"
    assert response.json()["success"] == True, "賣出交易未成功"
    
    print("\n[OK] 測試5通過：成功賣出 AAPL 5股")


def test_net_value_change(account_id):
    """測試場景6：查詢淨值變化"""
    print_section(f"測試6：查詢帳戶 {account_id} 淨值變化")
    
    response = requests.get(f"{BASE_URL}/api/sim/accounts/{account_id}")
    print(f"帳戶詳情：{response.status_code}")
    
    account = response.json()
    print(f"現金：${account['cash']:,.2f}")
    print(f"持倉市值：${account['total_market_value']:,.2f}")
    print(f"總淨值：${account['net_value']:,.2f}")
    print(f"總損益：${account['total_pnl']:,.2f} ({account['total_pnl_percent']}%)")
    
    # 檢查持倉數量應該是 5 股（買入10股，賣出5股）
    aapl_position = next((p for p in account["positions"] if p["symbol"] == "AAPL"), None)
    if aapl_position:
        print(f"\nAAPL 持倉：{aapl_position['quantity']} 股")
        assert aapl_position['quantity'] == 5, f"AAPL 持倉數量應為5股，實際為 {aapl_position['quantity']}"
    
    print("\n[OK] 測試6通過：淨值正確變化")


def test_account_isolation(account2_id):
    """測試場景7：多帳戶隔離驗證"""
    print_section(f"測試7：驗證帳戶 {account2_id} 隔離性")
    
    response = requests.get(f"{BASE_URL}/api/sim/accounts/{account2_id}")
    account = response.json()
    
    print(f"帳戶2 持倉數量：{len(account['positions'])}")
    assert len(account['positions']) == 0, "帳戶2不應該有持倉"
    assert account['cash'] == 200000, "帳戶2現金應為初始資金"
    
    print("\n[OK] 測試7通過：帳戶隔離正確")


def test_buy_insufficient_cash(account_id):
    """測試場景8：買超過現金的量（應失敗）"""
    print_section(f"測試8：帳戶 {account_id} 買超過現金的量")
    
    # 嘗試買入 1000 股 AAPL（肯定超過現金）
    trade_data = {
        "symbol": "AAPL",
        "action": "buy",
        "quantity": 1000
    }
    
    response = requests.post(f"{BASE_URL}/api/sim/accounts/{account_id}/trade", json=trade_data)
    print(f"買入交易：{response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    assert response.status_code == 400, "應該返回錯誤（現金不足）"
    assert "現金不足" in response.json()["detail"], "錯誤訊息應包含「現金不足」"
    
    print("\n[OK] 測試8通過：正確阻止現金不足的買入")


def test_sell_insufficient_position(account_id):
    """測試場景9：賣超過持倉的量（應失敗）"""
    print_section(f"測試9：帳戶 {account_id} 賣超過持倉的量")
    
    # 帳戶目前持有 5 股 AAPL，嘗試賣出 10 股
    trade_data = {
        "symbol": "AAPL",
        "action": "sell",
        "quantity": 10
    }
    
    response = requests.post(f"{BASE_URL}/api/sim/accounts/{account_id}/trade", json=trade_data)
    print(f"賣出交易：{response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    assert response.status_code == 400, "應該返回錯誤（持倉不足）"
    assert "持倉不足" in response.json()["detail"], "錯誤訊息應包含「持倉不足」"
    
    print("\n[OK] 測試9通過：正確阻止持倉不足的賣出")


def test_trading_costs(account_id):
    """測試場景10：驗證交易成本正確扣除"""
    print_section(f"測試10：驗證交易成本計算")
    
    # 獲取當前現金
    response = requests.get(f"{BASE_URL}/api/sim/accounts/{account_id}")
    cash_before = response.json()["cash"]
    
    # 買入 10 股 AAPL
    trade_data = {
        "symbol": "AAPL",
        "action": "buy",
        "quantity": 10
    }
    
    buy_response = requests.post(f"{BASE_URL}/api/sim/accounts/{account_id}/trade", json=trade_data)
    buy_result = buy_response.json()
    
    print(f"買入價格：${buy_result['price']:.2f}")
    print(f"買入手續費：${buy_result['commission']:.2f}")
    print(f"總成本：${buy_result['total_cost']:.2f}")
    
    # 驗證手續費計算
    expected_commission = buy_result['price'] * buy_result['quantity'] * 0.001425 * 0.6
    assert abs(buy_result['commission'] - expected_commission) < 0.01, "手續費計算不正確"
    
    # 賣出 10 股
    sell_data = {
        "symbol": "AAPL",
        "action": "sell",
        "quantity": 10
    }
    
    sell_response = requests.post(f"{BASE_URL}/api/sim/accounts/{account_id}/trade", json=sell_data)
    sell_result = sell_response.json()
    
    print(f"\n賣出價格：${sell_result['price']:.2f}")
    print(f"賣出手續費：${sell_result['commission']:.2f}")
    print(f"賣出證交稅：${sell_result['tax']:.2f}")
    print(f"總收入：${sell_result['total_proceeds']:.2f}")
    
    # 驗證證交稅計算
    expected_tax = sell_result['price'] * sell_result['quantity'] * 0.003
    assert abs(sell_result['tax'] - expected_tax) < 0.01, "證交稅計算不正確"
    
    print("\n[OK] 測試10通過：交易成本計算正確")


def test_trade_history(account_id):
    """測試場景11：查詢交易歷史"""
    print_section(f"測試11：查詢帳戶 {account_id} 交易歷史")
    
    response = requests.get(f"{BASE_URL}/api/sim/accounts/{account_id}/history")
    print(f"交易歷史：{response.status_code}")
    
    history = response.json()
    print(f"交易筆數：{history['count']}")
    
    assert history['count'] > 0, "應該有交易記錄"
    
    # 顯示前5筆交易
    print("\n最近5筆交易：")
    for i, tx in enumerate(history['transactions'][:5], 1):
        print(f"{i}. {tx['timestamp']} - {tx['action']} {tx['quantity']} {tx['symbol']} @ ${tx['price']:.2f}")
    
    print("\n[OK] 測試11通過：交易歷史查詢正確")


def test_delete_account(account2_id):
    """測試場景12：刪除帳戶"""
    print_section(f"測試12：刪除帳戶 {account2_id}")
    
    response = requests.delete(f"{BASE_URL}/api/sim/accounts/{account2_id}")
    print(f"刪除帳戶：{response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    assert response.status_code == 200, "刪除帳戶失敗"
    
    # 驗證帳戶已刪除
    check_response = requests.get(f"{BASE_URL}/api/sim/accounts/{account2_id}")
    assert check_response.status_code == 404, "帳戶應該已被刪除"
    
    print("\n[OK] 測試12通過：帳戶成功刪除")


def main():
    """執行所有測試"""
    print("\n" + "="*60)
    print("模擬交易系統完整測試開始")
    print("="*60)
    
    try:
        # 測試1-2：建立帳戶和列出帳戶
        account1_id, account2_id = test_create_account()
        test_list_accounts()
        
        # 測試3-6：基本交易流程
        test_buy_stock(account1_id)
        test_account_details(account1_id)
        test_sell_partial(account1_id)
        test_net_value_change(account1_id)
        
        # 測試7：多帳戶隔離
        test_account_isolation(account2_id)
        
        # 測試8-9：錯誤處理
        test_buy_insufficient_cash(account1_id)
        test_sell_insufficient_position(account1_id)
        
        # 測試10：交易成本驗證
        test_trading_costs(account1_id)
        
        # 測試11：交易歷史
        test_trade_history(account1_id)
        
        # 測試12：刪除帳戶
        test_delete_account(account2_id)
        
        print("\n" + "="*60)
        print("[OK] 所有測試通過！")
        print("="*60)
        
        return True
    
    except AssertionError as e:
        print(f"\n[FAIL] 測試失敗：{str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    except Exception as e:
        print(f"\n[FAIL] 測試錯誤：{str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
