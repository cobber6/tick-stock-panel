from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api import backtest as api
from app.backtest.history import BacktestHistoryStore


def _result(error=None):
    return {
        "run_id": "run-1",
        "config": {"strategy_id": "demo", "start": "2025-01-01", "end": "2025-12-31"},
        "stats": {"total_return": 0.1, "n_trades": 3},
        "equity_curve": [],
        "trades": [],
        "strategy_info": {"id": "demo", "name": "演示策略"},
        "elapsed_ms": 10.0,
        "error": error,
    }


def test_history_api_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(api.settings, "data_dir", tmp_path)
    summary = BacktestHistoryStore(tmp_path).record(_result())

    assert api.history_list()["items"][0]["id"] == summary["id"]
    assert api.history_get(summary["id"])["result"]["run_id"] == "run-1"
    assert api.history_delete(summary["id"]) == {"ok": True}

    with pytest.raises(HTTPException) as exc_info:
        api.history_get(summary["id"])
    assert exc_info.value.status_code == 404


def test_record_history_skips_failed_results(monkeypatch):
    recorded = []

    class _Store:
        def record(self, payload):
            recorded.append(payload)

    monkeypatch.setattr(api, "_history_store", lambda: _Store())

    api._record_backtest_history(_result(error="cancelled"))
    api._record_backtest_history(_result())

    assert len(recorded) == 1
    assert recorded[0]["run_id"] == "run-1"


def test_record_history_never_breaks_backtest(monkeypatch):
    def _boom():
        raise OSError("disk full")

    monkeypatch.setattr(api, "_history_store", _boom)

    # 记录失败只告警, 不影响回测结果返回
    api._record_backtest_history(_result())
