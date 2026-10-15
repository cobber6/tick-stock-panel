"""策略自带回测成交口径默认值 (META.backtest_defaults) 的详情透传回归测试.

背景 (bug): 打板策略的信号必须在封板当日成交, 而回测页默认口径是「次日开盘」,
会把打板单推迟到次日 —— 次日低开低走也照样成交, 与「板上买入」完全不是一回事.
这类策略因此在 META 里自带成交口径声明, 由 _strategy_detail 透传给前端自动套用.

契约:
  - 未声明 / 非法声明 → backtest_defaults 为 None (坏配置一律不透传)
  - 合法声明         → 原样透传 entry_fill / exit_fill
"""
from __future__ import annotations

import pytest

from app.api.strategy import _strategy_detail
from app.strategy.engine import StrategyDef


def _make_strategy(meta: dict) -> StrategyDef:
    """构造最小可用的 StrategyDef (只填必填字段)."""
    return StrategyDef(
        meta=meta,
        basic_filter={"enabled": True},
        entry_signals=[],
        exit_signals=[],
        stop_loss=-0.08,
        trailing_stop=None,
        trailing_take_profit_activate=None,
        trailing_take_profit_drawdown=None,
        max_hold_days=10,
        filter_fn=None,
        filter_history_fn=None,
        lookback_days=60,
        source="builtin",
    )


def test_declared_fills_are_passed_through():
    s = _make_strategy({
        "id": "test_strat",
        "backtest_defaults": {"entry_fill": "close_t", "exit_fill": "close_t"},
    })
    assert _strategy_detail(s, overrides=None)["backtest_defaults"] == {
        "entry_fill": "close_t",
        "exit_fill": "close_t",
    }


def test_absent_declaration_is_none():
    """绝大多数策略不声明口径, 前端应保持页面当前选择不变."""
    s = _make_strategy({"id": "test_strat"})
    assert _strategy_detail(s, overrides=None)["backtest_defaults"] is None


def test_partial_declaration_keeps_only_declared_side():
    s = _make_strategy({"id": "test_strat", "backtest_defaults": {"entry_fill": "open_t+1"}})
    assert _strategy_detail(s, overrides=None)["backtest_defaults"] == {"entry_fill": "open_t+1"}


@pytest.mark.parametrize(
    "declared",
    [
        "close_t",                                   # 不是 dict
        {"entry_fill": "next_close"},                # 未知建仓口径
        {"exit_fill": "open_t+2"},                   # 未知清仓口径
        {"entry_fill": None, "exit_fill": ""},       # 空值
        {},                                          # 空声明
    ],
)
def test_invalid_declarations_are_dropped(declared):
    """META 是策略源文件里的自由 dict: 坏配置必须丢弃而非透传到前端."""
    s = _make_strategy({"id": "test_strat", "backtest_defaults": declared})
    assert _strategy_detail(s, overrides=None)["backtest_defaults"] is None


def test_high_t_is_a_valid_entry_declaration():
    """打板策略的「当日最高价」建仓口径必须被白名单接受。

    否则声明会被静默丢弃, 前端回落到「次日开盘」—— 打板单被推迟到次日开盘成交, 结果失真。
    """
    s = _make_strategy({
        "id": "test_strat",
        "backtest_defaults": {"entry_fill": "high_t", "exit_fill": "close_t"},
    })
    assert _strategy_detail(s, overrides=None)["backtest_defaults"] == {
        "entry_fill": "high_t",
        "exit_fill": "close_t",
    }


def test_minute_exit_fill_is_a_valid_exit_declaration():
    s = _make_strategy({
        "id": "test_strat",
        "backtest_defaults": {"entry_fill": "close_t", "exit_fill": "signal_next_minute"},
    })
    assert _strategy_detail(s, overrides=None)["backtest_defaults"] == {
        "entry_fill": "close_t",
        "exit_fill": "signal_next_minute",
    }
