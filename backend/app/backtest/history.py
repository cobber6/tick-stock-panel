"""策略回测历史 —— 每次成功回测自动落盘, 供跨会话对比与复看。

存储布局 (均在 ``data_dir/user_data/backtest_history/`` 下):

- ``index.json``   轻量摘要列表 (按时间倒序), 列表接口只读它
- ``<id>.json``    单次回测的完整结果, 仅在查看详情时读取

摘要与完整结果分开, 避免列表接口反序列化全部交易与净值曲线。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_ENTRIES = 100
MAX_RESULT_BYTES = 64 * 1024 * 1024
MAX_INDEX_BYTES = 8 * 1024 * 1024
_ID_PATTERN = re.compile(r"[0-9a-f]{32}")

# 列表/对比所需的指标摘要 (取自 result.stats)
_METRIC_KEYS = (
    "total_return",
    "annual_return",
    "max_drawdown",
    "sharpe",
    "sortino",
    "calmar",
    "win_rate",
    "n_trades",
    "profit_factor",
    "avg_pnl",
    "final_equity",
    "benchmark_return",
    "excess",
)

# 载入历史记录时回填给回测表单的配置字段
_CONFIG_KEYS = (
    "strategy_id",
    "symbols",
    "start",
    "end",
    "params",
    "overrides",
    "matching",
    "entry_fill",
    "exit_fill",
    "fees_pct",
    "commission_pct",
    "stamp_tax_pct",
    "slippage_bps",
    "max_positions",
    "max_exposure_pct",
    "initial_capital",
    "position_sizing",
    "mode",
    "holding_days",
    "asset_type",
    "minute_fill",
    "regime_filter",
)

_lock = threading.RLock()


class BacktestHistoryError(RuntimeError):
    """历史记录读写失败 (文件损坏、超限等), 调用方应明确报错而非静默丢数据。"""


class BacktestHistoryNotFoundError(KeyError):
    """指定的历史记录不存在。"""


def _json_safe(value: Any) -> Any:
    """把 NaN/Infinity 置 None, 日期转 ISO 字符串 —— 保证落盘 JSON 合法。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _finite_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _valid_entry_id(entry_id: str) -> bool:
    return isinstance(entry_id, str) and _ID_PATTERN.fullmatch(entry_id) is not None


class BacktestHistoryStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "user_data" / "backtest_history"
        self.index_path = self.root / "index.json"

    def list(self) -> list[dict[str, Any]]:
        with _lock:
            return self._load_index()

    def get(self, entry_id: str) -> dict[str, Any]:
        if not _valid_entry_id(entry_id):
            raise BacktestHistoryNotFoundError(entry_id)
        with _lock:
            entry = self._find(entry_id)
            entry["result"] = self._read_result(entry_id) if entry["has_result"] else None
            return entry

    def record(self, result: dict[str, Any]) -> dict[str, Any]:
        """保存一次成功回测。结果中的 error/cancelled 由调用方提前过滤。"""
        if not isinstance(result, dict):
            raise BacktestHistoryError("回测结果格式无效")
        summary = self._build_summary(result)
        try:
            # 常见路径直接序列化, 避免复制整份结果 (交易/净值曲线可能很大)
            payload_text = json.dumps(
                result, ensure_ascii=False, allow_nan=False, default=str,
            )
        except (TypeError, ValueError):
            # NaN/Infinity 或不可序列化对象: 递归清洗后再写
            payload_text = json.dumps(
                _json_safe(result), ensure_ascii=False, allow_nan=False, default=str,
            )
        has_result = len(payload_text.encode("utf-8")) <= MAX_RESULT_BYTES
        if not has_result:
            logger.warning("回测结果超过 %s 字节, 仅保存指标摘要", MAX_RESULT_BYTES)
        summary["has_result"] = has_result

        with _lock:
            entries = self._load_index()
            if has_result:
                self._write_text(self.root / f"{summary['id']}.json", payload_text)
            merged = [summary, *entries]
            kept = merged[:MAX_ENTRIES]
            dropped = merged[MAX_ENTRIES:]
            try:
                self._write_index(kept)
            except BacktestHistoryError:
                if has_result:
                    (self.root / f"{summary['id']}.json").unlink(missing_ok=True)
                raise
            for item in dropped:
                self._delete_result_file(item.get("id"))
            return summary

    def delete(self, entry_id: str) -> None:
        if not _valid_entry_id(entry_id):
            raise BacktestHistoryNotFoundError(entry_id)
        with _lock:
            entries = self._load_index()
            remaining = [item for item in entries if item.get("id") != entry_id]
            if len(remaining) == len(entries):
                raise BacktestHistoryNotFoundError(entry_id)
            self._write_index(remaining)
            self._delete_result_file(entry_id)

    # ── 内部实现 ──────────────────────────────────────────────

    @staticmethod
    def _build_summary(result: dict[str, Any]) -> dict[str, Any]:
        stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
        info = result.get("strategy_info") if isinstance(result.get("strategy_info"), dict) else {}
        config = result.get("config") if isinstance(result.get("config"), dict) else {}
        strategy_id = str(config.get("strategy_id") or info.get("id") or "")
        metrics = {
            key: _finite_number(stats.get(key))
            for key in _METRIC_KEYS
        }
        metrics = {key: value for key, value in metrics.items() if value is not None}
        clean_config = {
            key: _json_safe(config.get(key))
            for key in _CONFIG_KEYS
            if config.get(key) is not None
        }
        return {
            "id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "run_id": str(result.get("run_id") or ""),
            "strategy_id": strategy_id,
            "strategy_name": str(info.get("name") or strategy_id or "策略"),
            "asset_type": str(config.get("asset_type") or "stock"),
            "start": str(config.get("start") or ""),
            "end": str(config.get("end") or ""),
            "metrics": metrics,
            "config": clean_config,
            "elapsed_ms": _finite_number(result.get("elapsed_ms")),
        }

    def _load_index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        try:
            if self.index_path.stat().st_size > MAX_INDEX_BYTES:
                raise BacktestHistoryError("回测历史索引过大, 已停止读取")
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except BacktestHistoryError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BacktestHistoryError("回测历史索引损坏或无法读取") from exc
        if not isinstance(raw, list):
            raise BacktestHistoryError("回测历史索引格式无效")
        return [entry for value in raw if (entry := self._normalize(value)) is not None]

    def _find(self, entry_id: str) -> dict[str, Any]:
        for entry in self._load_index():
            if entry["id"] == entry_id:
                return entry
        raise BacktestHistoryNotFoundError(entry_id)

    def _read_result(self, entry_id: str) -> dict[str, Any]:
        path = self.root / f"{entry_id}.json"
        try:
            if not path.exists():
                raise BacktestHistoryError("回测结果文件缺失")
            if path.stat().st_size > MAX_RESULT_BYTES:
                raise BacktestHistoryError("回测结果文件过大, 已停止读取")
            value = json.loads(path.read_text(encoding="utf-8"))
        except BacktestHistoryError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BacktestHistoryError("回测结果文件损坏或无法读取") from exc
        if not isinstance(value, dict):
            raise BacktestHistoryError("回测结果文件格式无效")
        return value

    def _write_index(self, entries: list[dict[str, Any]]) -> None:
        try:
            payload = json.dumps(entries, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise BacktestHistoryError("回测历史索引无法序列化") from exc
        self._write_text(self.index_path, payload)

    def _write_text(self, path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise BacktestHistoryError("回测历史保存失败") from exc

    def _delete_result_file(self, entry_id: Any) -> None:
        if not _valid_entry_id(entry_id):
            return
        try:
            (self.root / f"{entry_id}.json").unlink(missing_ok=True)
        except OSError:
            # 索引已删除, 残留结果文件不影响列表; 记录日志便于人工清理
            logger.warning("回测历史结果文件删除失败: %s", entry_id, exc_info=True)

    @staticmethod
    def _normalize(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        entry_id = value.get("id")
        if not _valid_entry_id(entry_id):
            return None
        metrics = value.get("metrics") if isinstance(value.get("metrics"), dict) else {}
        clean_metrics = {
            key: number
            for key, item in metrics.items()
            if key in _METRIC_KEYS and (number := _finite_number(item)) is not None
        }
        config = value.get("config") if isinstance(value.get("config"), dict) else {}
        clean_config = {
            key: item for key, item in config.items() if key in _CONFIG_KEYS
        }
        return {
            "id": entry_id,
            "created_at": str(value.get("created_at") or ""),
            "run_id": str(value.get("run_id") or ""),
            "strategy_id": str(value.get("strategy_id") or ""),
            "strategy_name": str(value.get("strategy_name") or "策略"),
            "asset_type": str(value.get("asset_type") or "stock"),
            "start": str(value.get("start") or ""),
            "end": str(value.get("end") or ""),
            "metrics": clean_metrics,
            "config": clean_config,
            "elapsed_ms": _finite_number(value.get("elapsed_ms")),
            "has_result": bool(value.get("has_result", True)),
        }
