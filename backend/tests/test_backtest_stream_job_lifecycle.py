"""回测 SSE 作业表生命周期 — 被拒请求与从未启动的作业都不得永久滞留。

回归场景: `_cleanup_stale_jobs` 只回收 `done=True` 的条目, 而三个 stream 端点在
入表之后有多条"yield error 后 return"的早退分支 (守卫拒绝 / 权限不足 / 分钟K区间
不足 / 非法参数网格) 从不置 `done`, 这些条目永远匹配不到清理条件 —— 换一个不参与
守卫的参数就是新 job_key, 反复请求即让 `_running_jobs` 无界增长。
"""
from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import backtest as bt
from app.api.backtest import router
from app.config import settings

STREAMS = [
    ("/api/backtest/strategy/stream", {"strategy_id": "ma_cross"}),
    ("/api/backtest/optimize/stream", {"strategy_id": "ma_cross", "param_grid": '{"n": [5]}'}),
    ("/api/backtest/walkforward/stream", {"strategy_id": "ma_cross", "param_grid": '{"n": [5]}'}),
]
STREAM_IDS = [path.rsplit("/", 2)[1] for path, _ in STREAMS]


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    with bt._jobs_lock:
        bt._running_jobs.clear()
    with TestClient(app, raise_server_exceptions=False) as c:
        try:
            yield c
        finally:
            with bt._jobs_lock:
                bt._running_jobs.clear()


def _pending_jobs() -> dict:
    with bt._jobs_lock:
        return {k: j for k, j in bt._running_jobs.items() if not j.done}


@pytest.mark.parametrize("path,extra", STREAMS, ids=STREAM_IDS)
def test_guard_rejected_stream_leaves_no_pending_job(client, monkeypatch, path, extra):
    """服务端范围保护拒绝的请求, 事件流以 error 收尾后不得留下未终结条目。"""
    monkeypatch.setattr(settings, "backtest_range_guard", True)
    params = {**extra, "start": "2020-01-01", "end": "2026-09-04"}
    if "walkforward" in path:
        # walkforward 的守卫作用于单折窗口, 不是总区间
        params.update({"train_days": 400, "test_days": 400})

    resp = client.get(path, params=params)
    assert resp.status_code == 200, resp.text
    assert "event: error" in resp.text

    assert _pending_jobs() == {}, f"守卫拒绝的请求留下了未终结任务: {_pending_jobs()}"


def test_rejection_path_arms_expiry_timer(client, monkeypatch):
    """被拒作业必须挂 TTL 定时器自清, 不能只等下一次请求触发 cleanup。"""
    monkeypatch.setattr(bt, "_JOB_TTL", 0.2)
    monkeypatch.setattr(settings, "backtest_range_guard", False)

    resp = client.get(
        "/api/backtest/walkforward/stream",
        # 显式给 start: 缺省时会走 app.state.repo 取最早日期, 与本用例无关
        params={
            "strategy_id": "ma_cross",
            "param_grid": "{}",
            "start": "2025-01-01",
            "end": "2025-06-30",
        },
    )
    assert resp.status_code == 200, resp.text
    assert "event: error" in resp.text

    for _ in range(50):
        if not _pending_jobs() and not bt._running_jobs:
            return
        time.sleep(0.1)
    with bt._jobs_lock:
        assert not bt._running_jobs, f"未挂 TTL 定时器, 条目滞留: {bt._running_jobs}"


def test_never_started_job_is_reaped(client):
    """已入表但从未启动的作业 (客户端在建流前断开) 超过 TTL 后应被清理。"""
    key = "deadbeef0000"
    with bt._jobs_lock:
        job = bt._BacktestJob(key)
        job.created_ts = time.time() - (bt._JOB_TTL + 1)
        bt._running_jobs[key] = job

    bt._cleanup_stale_jobs()

    with bt._jobs_lock:
        assert key not in bt._running_jobs, "从未启动的作业未被回收"


def test_running_job_is_not_reaped(client):
    """已启动但仍在跑的作业不得被当成滞留条目误清 (守卫: 只清从未启动的)。"""
    key = "cafebabe0000"
    with bt._jobs_lock:
        job = bt._BacktestJob(key)
        job.created_ts = time.time() - (bt._JOB_TTL + 1)
        job.started = True
        bt._running_jobs[key] = job

    bt._cleanup_stale_jobs()

    with bt._jobs_lock:
        assert key in bt._running_jobs, "运行中的作业被误清"
