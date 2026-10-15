"""high_t 成交口径: 入场价取**信号当日最高价**(触板价)。

打板类策略需要「触板即买」的成交价。封板日 high == close == 涨停价, 因此该口径下
「触板即买」与「封板才买」两种触发拿到的都是打板单的真实成交价。

与 close_t 的差别不止取价: close_t 会套用「收盘封板不可成交」的拦截
(见 test_limit_up_close_fill.py), high_t 不套用 —— 封板本身就是打板单的成交条件。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from app.backtest.engine import BacktestEngine, MatcherConfig, _fill_base_prices

_SIGNAL_DAY = date(2024, 1, 4)


def _panel(sealed_limit_day: date | None = None) -> pl.DataFrame:
    """单标的 6 日面板。

    信号日 (01-04) 特意拉开 open/high/close 三个价位, 便于区分成交口径;
    sealed_limit_day 指定某日收盘封涨停(high == close == 涨停价, 非一字)。
    """
    start = date(2024, 1, 1)
    rows = []
    for i in range(6):
        day = start + timedelta(days=i)
        px = 10.0 + i * 0.05
        if day == sealed_limit_day:
            prev_close = 10.0 + (i - 1) * 0.05
            limit = round(prev_close * 1.1, 2)
            o, lo = px, px - 0.1
            h = c = limit
        else:
            o, lo = round(px - 0.03, 2), round(px - 0.10, 2)
            h, c = round(px * 1.04, 2), px
        rows.append(
            {
                "symbol": "A",
                "date": day,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": 100_000,
                "signal_limit_up": day == sealed_limit_day,
                "signal_limit_down": False,
            }
        )
    return pl.DataFrame(rows).sort(["symbol", "date"])


def _run(panel: pl.DataFrame, signal_day: date, **fill: str):
    entry_mask = pl.Series(
        [r["date"] == signal_day for r in panel.select(["symbol", "date"]).iter_rows(named=True)],
        dtype=pl.Boolean,
    )
    exit_mask = pl.Series([False] * len(panel), dtype=pl.Boolean)
    return BacktestEngine(repo=None).simulate_independent_candidates(  # type: ignore[arg-type]
        panel,
        entry_mask,
        exit_mask,
        MatcherConfig(fees_pct=0, slippage_bps=0, max_hold_days=1, **fill),
    )


def _high_on(panel: pl.DataFrame, day: date) -> float:
    return float(panel.filter(pl.col("date") == day)["high"][0])


def _close_on(panel: pl.DataFrame, day: date) -> float:
    return float(panel.filter(pl.col("date") == day)["close"][0])


def test_fill_base_prices_selects_column_per_convention():
    """单元: 三种口径各自取对应列, 未识别值回退收盘价。"""
    opens = np.array([1.0])
    closes = np.array([2.0])
    highs = np.array([3.0])
    assert _fill_base_prices("open_t+1", opens, closes, highs)[0] == 1.0
    assert _fill_base_prices("high_t", opens, closes, highs)[0] == 3.0
    assert _fill_base_prices("close_t", opens, closes, highs)[0] == 2.0
    assert _fill_base_prices(None, opens, closes, highs)[0] == 2.0


def test_high_t_fills_at_signal_day_high():
    """high_t 在信号当日以最高价成交, 且不被推迟到次日。"""
    panel = _panel()
    signal_high = _high_on(panel, _SIGNAL_DAY)

    result = _run(panel, _SIGNAL_DAY, entry_fill="high_t")

    assert result.stats.get("n_trades") == 1
    trade = result.trades[0]
    assert trade.entry_date == "2024-01-04"
    assert trade.entry_price == signal_high


def test_high_t_price_differs_from_close_t_and_open_t1():
    """同一天三个口径取到三个不同价位 —— 证明 high_t 没退化成收盘价。"""
    panel = _panel()
    signal_high = _high_on(panel, _SIGNAL_DAY)
    signal_close = _close_on(panel, _SIGNAL_DAY)
    next_open = float(panel.filter(pl.col("date") == _SIGNAL_DAY + timedelta(days=1))["open"][0])
    assert len({signal_high, signal_close, next_open}) == 3  # 面板本身须能区分

    assert _run(panel, _SIGNAL_DAY, entry_fill="high_t").trades[0].entry_price == signal_high
    assert _run(panel, _SIGNAL_DAY, entry_fill="close_t").trades[0].entry_price == signal_close
    assert _run(panel, _SIGNAL_DAY, entry_fill="open_t+1").trades[0].entry_price == next_open


def test_high_t_fills_on_sealed_limit_up_day():
    """信号日收盘封涨停: high_t 以涨停价成交, 而 close_t 被既有拦截挡住。"""
    sealed = _panel(sealed_limit_day=_SIGNAL_DAY)
    limit_price = _high_on(sealed, _SIGNAL_DAY)
    assert limit_price == _close_on(sealed, _SIGNAL_DAY)  # 封板: high == close == 涨停价

    high_t = _run(sealed, _SIGNAL_DAY, entry_fill="high_t")
    assert high_t.stats.get("n_trades") == 1
    assert high_t.trades[0].entry_date == "2024-01-04"
    assert high_t.trades[0].entry_price == limit_price

    assert _run(sealed, _SIGNAL_DAY, entry_fill="close_t").stats.get("n_trades") == 0
