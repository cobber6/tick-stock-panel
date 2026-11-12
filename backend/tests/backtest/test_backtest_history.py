from __future__ import annotations

import json
from datetime import date

import pytest

from app.backtest import history as history_module
from app.backtest.history import (
    BacktestHistoryError,
    BacktestHistoryNotFoundError,
    BacktestHistoryStore,
)


def _result(**stats_overrides):
    stats = {
        "total_return": 0.32,
        "annual_return": 0.11,
        "max_drawdown": -0.08,
        "sharpe": 1.4,
        "sortino": 1.9,
        "win_rate": 0.55,
        "n_trades": 42,
        "profit_factor": 1.7,
        "final_equity": 1_320_000.0,
    }
    stats.update(stats_overrides)
    return {
        "run_id": "run-1",
        "config": {
            "strategy_id": "demo",
            "symbols": ["000001.SZ"],
            "start": "2025-01-01",
            "end": "2025-12-31",
            "asset_type": "stock",
            "fees_pct": 0.0002,
        },
        "stats": stats,
        "equity_curve": [{"date": "2025-01-02", "value": 1.0}],
        "trades": [],
        "strategy_info": {"id": "demo", "name": "演示策略"},
        "elapsed_ms": 1234.5,
        "error": None,
    }


def test_history_record_list_get_delete(tmp_path):
    store = BacktestHistoryStore(tmp_path)

    summary = store.record(_result())

    assert summary["strategy_name"] == "演示策略"
    assert summary["metrics"]["total_return"] == 0.32
    assert summary["has_result"] is True
    assert not store.index_path.with_suffix(".json.tmp").exists()

    items = store.list()
    assert len(items) == 1
    assert "result" not in items[0]

    detail = store.get(summary["id"])
    assert detail["result"]["run_id"] == "run-1"
    assert detail["result"]["trades"] == []

    store.delete(summary["id"])
    assert store.list() == []
    assert not (store.root / f"{summary['id']}.json").exists()


def test_history_rejects_unknown_and_invalid_ids(tmp_path):
    store = BacktestHistoryStore(tmp_path)
    summary = store.record(_result())

    with pytest.raises(BacktestHistoryNotFoundError):
        store.get("0" * 32)
    with pytest.raises(BacktestHistoryNotFoundError):
        store.get("../../etc/passwd")
    with pytest.raises(BacktestHistoryNotFoundError):
        store.delete("../../etc/passwd")

    # 合法 id 但已被删除
    store.delete(summary["id"])
    with pytest.raises(BacktestHistoryNotFoundError):
        store.delete(summary["id"])


def test_history_prunes_oldest_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(history_module, "MAX_ENTRIES", 2)
    store = BacktestHistoryStore(tmp_path)

    first = store.record(_result())
    second = store.record(_result())
    third = store.record(_result())

    items = store.list()
    assert [item["id"] for item in items] == [third["id"], second["id"]]
    assert not (store.root / f"{first['id']}.json").exists()


def test_history_sanitizes_non_finite_and_dates(tmp_path):
    store = BacktestHistoryStore(tmp_path)
    result = _result(total_return=float("nan"), sharpe=float("inf"))
    result["config"]["end"] = date(2025, 12, 31)

    summary = store.record(result)

    assert "total_return" not in summary["metrics"]
    assert "sharpe" not in summary["metrics"]
    assert summary["config"]["end"] == "2025-12-31"
    # 索引与结果文件都必须能被严格 JSON 解析 (NaN/Infinity 会让前端 JSON.parse 崩)
    json.loads(store.index_path.read_text(encoding="utf-8"))
    detail = store.get(summary["id"])
    assert detail["result"]["stats"]["total_return"] is None


def test_history_oversize_result_keeps_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(history_module, "MAX_RESULT_BYTES", 64)
    store = BacktestHistoryStore(tmp_path)

    summary = store.record(_result())

    assert summary["has_result"] is False
    assert not (store.root / f"{summary['id']}.json").exists()
    detail = store.get(summary["id"])
    assert detail["result"] is None
    assert detail["metrics"]["total_return"] == 0.32


def test_history_corrupt_index_fails_closed(tmp_path):
    store = BacktestHistoryStore(tmp_path)
    store.root.mkdir(parents=True)
    store.index_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(BacktestHistoryError, match="损坏"):
        store.list()
    with pytest.raises(BacktestHistoryError, match="损坏"):
        store.record(_result())
    assert store.index_path.read_text(encoding="utf-8") == "{broken"
