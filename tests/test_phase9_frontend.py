"""
test_phase9_frontend.py — Phase 9 前端 UX 測試
================================================

靜態 HTML 結構驗證：
  TC-F01 — help modal 存在
  TC-F02 — help button 在 header 中
  TC-F03 — help tooltip 在 AI 預測訊號 section
  TC-F04 — help tooltip 在 模擬交易 section
  TC-F05 — help tooltip 在 回測系統 section
  TC-F06 — toast CSS class 定義
  TC-F07 — toastContainer div 存在
  TC-F08 — toggleHelp function 存在
  TC-F09 — showToast function 存在
  TC-F10 — ESC keyboard handler 存在

Run:
    cd stock-app
    python -m pytest tests/test_phase9_frontend.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

HTML_PATH = Path(__file__).resolve().parent.parent / "static" / "index.html"


@pytest.fixture(scope="module")
def html_content() -> str:
    assert HTML_PATH.exists(), f"index.html not found at {HTML_PATH}"
    return HTML_PATH.read_text(encoding="utf-8")


def test_tcf01_help_modal_exists(html_content: str):
    """TC-F01: #helpModal div 存在"""
    assert 'id="helpModal"' in html_content, "#helpModal not found in index.html"


def test_tcf02_help_button_in_header(html_content: str):
    """TC-F02: header 中有 help-btn"""
    assert "help-btn" in html_content, "help-btn class not found"
    assert "toggleHelp()" in html_content, "toggleHelp() call not found"


def test_tcf03_tooltip_ai_prediction(html_content: str):
    """TC-F03: AI 預測訊號 section 有 tooltip"""
    assert "AI 預測訊號" in html_content
    # Find the section and check tooltip follows
    idx = html_content.index("AI 預測訊號")
    section = html_content[idx: idx + 500]
    assert "help-tooltip" in section, "No help-tooltip near AI 預測訊號 section"


def test_tcf04_tooltip_simulated_trading(html_content: str):
    """TC-F04: 模擬交易 section 有 tooltip"""
    assert "模擬交易" in html_content
    idx = html_content.index("模擬交易")
    section = html_content[idx: idx + 500]
    assert "help-tooltip" in section, "No help-tooltip near 模擬交易 section"


def test_tcf05_tooltip_backtest(html_content: str):
    """TC-F05: 回測系統 section 有 tooltip"""
    assert "回測系統" in html_content
    idx = html_content.index("回測系統")
    section = html_content[idx: idx + 500]
    assert "help-tooltip" in section, "No help-tooltip near 回測系統 section"


def test_tcf06_toast_css_defined(html_content: str):
    """TC-F06: toast CSS class 已定義"""
    assert ".toast {" in html_content or ".toast{" in html_content, \
        ".toast CSS class not defined"
    assert ".toast.success" in html_content, ".toast.success not defined"
    assert ".toast.error" in html_content, ".toast.error not defined"


def test_tcf07_toast_container_exists(html_content: str):
    """TC-F07: #toastContainer div 存在"""
    assert 'id="toastContainer"' in html_content, "#toastContainer div not found"


def test_tcf08_toggle_help_function(html_content: str):
    """TC-F08: toggleHelp JS function 定義"""
    assert "function toggleHelp()" in html_content, "toggleHelp() function not defined"


def test_tcf09_show_toast_function(html_content: str):
    """TC-F09: showToast JS function 定義"""
    assert "function showToast(" in html_content, "showToast() function not defined"


def test_tcf10_esc_keyboard_handler(html_content: str):
    """TC-F10: ESC 鍵盤事件 handler 存在"""
    assert "Escape" in html_content, "Escape key handler not found"
    assert "addEventListener('keydown'" in html_content or 'addEventListener("keydown"' in html_content, \
        "keydown event listener not found"
