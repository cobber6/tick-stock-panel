"""策略可访问的只读数据入口(指数/ETF 日K、概念/行业成分) — 白名单放行。

供 Custom/AI 策略在 filter_history 内读取任意指数(及 ETF)的完整日K, 以及
当前概念/行业成分映射。策略通过白名单 import 本模块, 调用纯读函数;
禁止写操作或任意文件访问。

口径提示: 概念/行业成分来自 ext_data 快照, 只有当前版本, 没有历史版本。
按快照回看历史存在归属漂移(见 market_mainline.MEMBERSHIP_NOTE), 越近越准。

设计要点:
  - 模块自身是框架侧信任代码, 对策略的沙箱逃逸拦截(ai_generator._validate_safety)照旧生效。
  - repo 线程安全懒加载(首次调用才构建); DataStore() 默认 settings.data_dir, 与 main.py 同源。
  - 未知 symbol / 数据缺失 → 返回空 DataFrame(不抛), 与 repo 语义一致。
"""
from __future__ import annotations

import logging
import threading
from datetime import date
from typing import Any

import polars as pl

from app.market_time import cn_today

logger = logging.getLogger(__name__)

# 完整历史默认区间下界(A股数据远晚于此, 仅作"全量"占位)。
_FULL_START = date(1990, 1, 1)

# ── repo 懒加载(线程安全) ─────────────────────────────
_repo = None
_lock = threading.Lock()


def _get_repo():
    global _repo
    if _repo is None:
        with _lock:
            if _repo is None:
                from app.tickflow.repository import DataStore, KlineRepository
                _repo = KlineRepository(DataStore())
    return _repo


def _set_repo(repo: Any) -> None:
    """测试注入: 用 fake repo 替换单例。"""
    global _repo
    with _lock:
        _repo = repo


def _reset_repo() -> None:
    """测试清理: 重置单例, 下次调用重新懒加载。"""
    global _repo
    with _lock:
        _repo = None


# ── 参数规范化 ─────────────────────────────────────────
def _norm_date(value, default: date) -> date:
    if value is None:
        return default
    if isinstance(value, str):
        return date.fromisoformat(value)
    return value


def _validate_symbol(symbol: Any) -> bool:
    return isinstance(symbol, str) and bool(symbol.strip())


# ── 公开只读 API ───────────────────────────────────────
def get_index_daily(symbol, start=None, end=None, columns=None):
    """读取指数日K(含技术指标)。未知 symbol / 无数据返回空 DataFrame。"""
    if not _validate_symbol(symbol):
        logger.warning("market_data: 非法指数 symbol %r", symbol)
        return pl.DataFrame()
    s = _norm_date(start, _FULL_START)
    # 窗口右端必须是北京今天: 策略对照指数/ETF 的分区日期是北京交易日。
    # 美西主机整个 A 股交易时段、UTC 主机北京 00:00-08:00, date.today() 比北京早一天,
    # 未传 end 时会把当日官方 K 排除在窗口外。
    e = _norm_date(end, cn_today())
    try:
        return _get_repo().get_index_daily(symbol, s, e, columns)
    except Exception as exc:
        logger.warning("market_data get_index_daily failed %s: %s", symbol, exc)
        return pl.DataFrame()


def get_etf_daily(symbol, start=None, end=None, columns=None):
    """读取 ETF 日K(含技术指标)。同 get_index_daily 语义。"""
    if not _validate_symbol(symbol):
        logger.warning("market_data: 非法 ETF symbol %r", symbol)
        return pl.DataFrame()
    s = _norm_date(start, _FULL_START)
    e = _norm_date(end, cn_today())
    try:
        return _get_repo().get_etf_daily(symbol, s, e, columns)
    except Exception as exc:
        logger.warning("market_data get_etf_daily failed %s: %s", symbol, exc)
        return pl.DataFrame()


def get_daily(symbol, start=None, end=None, columns=None):
    """按资产类型自动分派读取日K: 指数 → get_index_daily; ETF → get_etf_daily; 股票 → get_daily。"""
    if not _validate_symbol(symbol):
        logger.warning("market_data: 非法 symbol %r", symbol)
        return pl.DataFrame()
    s = _norm_date(start, _FULL_START)
    e = _norm_date(end, cn_today())
    repo = _get_repo()
    try:
        asset_type = repo.resolve_asset_type(symbol)
        if asset_type == "index":
            return repo.get_index_daily(symbol, s, e, columns)
        if asset_type == "etf":
            return repo.get_etf_daily(symbol, s, e, columns)
        return repo.get_daily(symbol, s, e, columns)
    except Exception as exc:
        logger.warning("market_data get_daily failed %s: %s", symbol, exc)
        return pl.DataFrame()


def get_dimension_members(kind: str = "concept") -> pl.DataFrame:
    """读取概念/行业成分映射(当前快照), 返回 symbol/member 两列。

    kind: "concept"(概念) 或 "industry"(行业)。成分来自 ext_data 快照,
    无历史版本; 按快照回看历史存在归属漂移, 越近越准。无数据返回空表,
    不抛异常(与 get_daily 同语义), 调用方需自行判空。
    """
    if kind not in ("concept", "industry"):
        raise ValueError(f"unsupported dimension kind: {kind!r}")
    try:
        from app.services.rps_rotation import _load_concept_map_df

        map_df, _ = _load_concept_map_df(_get_repo(), kind)
    except Exception as exc:  # noqa: BLE001
        logger.warning("market_data get_dimension_members failed %s: %s", kind, exc)
        return pl.DataFrame(schema={"symbol": pl.Utf8, "member": pl.Utf8})
    if map_df.is_empty() or "_sym_up" not in map_df.columns or kind not in map_df.columns:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "member": pl.Utf8})
    return (
        map_df.select(
            pl.col("_sym_up").cast(pl.Utf8).alias("symbol"),
            pl.col(kind).cast(pl.Utf8).alias("member"),
        )
        .drop_nulls()
        .unique()
    )


def get_market_regime(start=None, end=None, columns=None) -> pl.DataFrame:
    """读取市场环境日频时序 (date/state/score...), 缺文件/空数据返回空表。

    state ∈ {strong, lean_strong, range, lean_weak, weak}。数据由盘后管道
    生成, 当日行在管道完成后才可见; 缺失时调用方应 fail-closed。
    """
    s = _norm_date(start, _FULL_START)
    e = _norm_date(end, cn_today())
    try:
        data_dir = _get_repo().store.data_dir
        path = data_dir / "regime_history" / "part.parquet"
        if not path.exists():
            return pl.DataFrame()
        df = pl.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("market_data get_market_regime failed: %s", exc)
        return pl.DataFrame()
    if df.is_empty() or "date" not in df.columns:
        return df
    df = df.filter((pl.col("date") >= s) & (pl.col("date") <= e))
    if columns:
        keep = [c for c in columns if c in df.columns]
        if keep:
            df = df.select(keep)
    return df


def list_index_symbols() -> list[dict]:
    """列出已收录的指数符号(含名称)。无数据返回空列表。"""
    try:
        df = _get_repo().get_instruments_asset("index")
    except Exception as exc:
        logger.warning("market_data list_index_symbols failed: %s", exc)
        return []
    if df.is_empty() or "symbol" not in df.columns:
        return []
    name_col = "name" if "name" in df.columns else None
    cols = ["symbol"] + ([name_col] if name_col else [])
    return [
        {"symbol": row["symbol"], "name": row.get("name")}
        for row in df.select(cols).iter_rows(named=True)
    ]
