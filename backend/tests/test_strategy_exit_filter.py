"""python_history_legacy 策略的自定义卖点 exit_filter 契约测试。"""
from __future__ import annotations

import textwrap
from datetime import date

import polars as pl
import pytest

from app.strategy.engine import StrategyDataContext, StrategyEngine

_LEGACY_STRATEGY = '''
import polars as pl

META = {
    "id": "custom_exit_filter_test",
    "name": "exit filter test",
    "description": "exit filter test",
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "params": [],
    "scoring": {},
}
EXECUTION_BACKEND = "python_history_legacy"


def filter_history(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    return df.filter(pl.col("date") == pl.col("date").min())


def exit_filter(df: pl.DataFrame, params: dict) -> pl.Expr:
    return pl.col("close") < pl.col("open")
'''

_POLARS_STRATEGY = '''
import polars as pl

META = {
    "id": "custom_exit_filter_polars",
    "name": "exit filter polars",
    "description": "exit filter polars",
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "params": [],
    "scoring": {},
}
EXECUTION_BACKEND = "polars_expr"


def filter(df: pl.DataFrame, params: dict) -> pl.Expr:
    return pl.lit(True)


def exit_filter(df: pl.DataFrame, params: dict) -> pl.Expr:
    return pl.col("close") < pl.col("open")
'''


def _write(tmp_path, name: str, source: str):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def test_loader_accepts_exit_filter_for_history_backend(tmp_path):
    strategy = StrategyEngine._load_file(
        _write(tmp_path, "custom_exit_filter_test", _LEGACY_STRATEGY)
    )

    assert strategy.exit_filter_fn is not None


def test_loader_rejects_exit_filter_for_polars_backend(tmp_path):
    with pytest.raises(ValueError, match="exit_filter is only supported"):
        StrategyEngine._load_file(
            _write(tmp_path, "custom_exit_filter_polars", _POLARS_STRATEGY)
        )


def test_loader_reads_entry_max_gap(tmp_path):
    source = _LEGACY_STRATEGY.replace(
        'EXECUTION_BACKEND = "python_history_legacy"',
        'ENTRY_MAX_GAP_PCT = 0.03\nEXECUTION_BACKEND = "python_history_legacy"',
    )
    strategy = StrategyEngine._load_file(
        _write(tmp_path, "custom_exit_filter_test", source)
    )

    assert strategy.entry_max_gap_pct == pytest.approx(0.03)


def test_loader_rejects_invalid_entry_max_gap(tmp_path):
    source = _LEGACY_STRATEGY.replace(
        'EXECUTION_BACKEND = "python_history_legacy"',
        'ENTRY_MAX_GAP_PCT = 0\nEXECUTION_BACKEND = "python_history_legacy"',
    )
    with pytest.raises(ValueError, match="ENTRY_MAX_GAP_PCT"):
        StrategyEngine._load_file(
            _write(tmp_path, "custom_exit_filter_test", source)
        )


def test_run_collects_exit_filter_hits(tmp_path):
    _write(tmp_path, "custom_exit_filter_test", _LEGACY_STRATEGY)
    engine = StrategyEngine(strategy_dirs=[tmp_path])
    day = date(2024, 1, 2)
    frame = pl.DataFrame({
        "symbol": ["A"],
        "name": ["A"],
        "date": [day],
        "open": [10.0],
        "high": [10.2],
        "low": [9.4],
        "close": [9.5],
        # 收盘阴线: exit_filter 命中
        "amount": [3e7],
        "signal_limit_up": [False],
        "signal_limit_down": [False],
    })
    context = StrategyDataContext(
        asset_type="stock",
        timeframe="1d",
        as_of=day,
        current=frame,
        history=frame,
    )

    result = engine.run("custom_exit_filter_test", context=context, params={})

    assert result.exit_signal_hits == [{"symbol": "A", "signals": ["exit_filter"]}]
