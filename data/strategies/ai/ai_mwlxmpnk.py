"""20cm套利：最强板块的创业板首板股，次日开盘买入，第3天收盘卖出

时间口径（按交易日计）：
  第1天 = 信号日：收盘后统计各概念/行业当日涨停家数，取涨停家数最多的板块，
          并在其中选一只创业板首板（连板数=1，低于2板）个股作为候选。
  第2天 = 信号日次日：以开盘价买入。
  第3天 = 买入次一交易日：以收盘价卖出（固定持有 1 个交易日）。

选股（B 口径）：板块内按成交额最大取一只，并只在情绪状态 range/lean_weak
时交易（可在参数里关闭情绪过滤）。
优化（C 口径）：次日开盘 gap >= 3% 不买（ENTRY_MAX_GAP_PCT），避免追高开。
限制：本地日线没有“涨停时间”字段，板块内“涨停时间最早”无法精确还原。
概念/行业成分为当前快照回看历史，存在归属漂移，越近越准。
"""
from __future__ import annotations

import polars as pl

from app.strategy.market_data import get_dimension_members, get_market_regime

META = {
    "id": "ai_mwlxmpnk",
    "name": "20cm套利",
    "description": "涨停数最多板块里成交额最大的创业板首板股，次日小高开买入，第3天收盘卖出",
    "tags": ["创业板", "涨停", "首板", "板块轮动", "套利"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    # 用户口径只有“创业板、低于2板、板块涨停数最多”，不再叠加 AI 默认的
    # 价格/市值/换手率边界，避免候选被事后过滤掉导致当日静默无交易。
    "basic_filter": {
        "price_min": None,
        "price_max": None,
        "market_cap_min": None,
        "market_cap_max": None,
        "float_cap_min": None,
        "float_cap_max": None,
        "amount_min": None,
        "amount_max": None,
        "turnover_min": None,
        "turnover_max": None,
        "exclude_st": True,
        "boards": ["创业板"],
    },
    # 信号日候选 → 次日开盘成交；第3天收盘离场。固定持有 1 个交易日，
    # 相邻信号会重叠一天，2 个仓位保证每个信号都能成交（各 50% 资金）。
    "backtest_defaults": {
        "entry_fill": "open_t+1",
        "exit_fill": "close_t",
        "max_positions": 2,
        "max_exposure_pct": 1.0,
    },
    "params": [
        {
            "id": "sector_kind",
            "label": "板块口径",
            "type": "select",
            "options": ["概念板块", "行业板块"],
            "default": "概念板块",
        },
        {
            "id": "pick_by",
            "label": "板块内择股",
            "type": "select",
            "options": ["成交额最大", "成交额最小", "流通市值最小", "换手率最高"],
            "default": "成交额最大",
        },
        {
            "id": "use_regime_filter",
            "label": "仅震荡/偏弱情绪交易",
            "type": "bool",
            "default": True,
        },
    ],
    "scoring": {},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "python_history_legacy"
ENTRY_SIGNALS = []
EXIT_SIGNALS = []
# 选股只用信号日当天横截面; 保留几天的历史窗口, 让选股/监控路径拿到完整帧。
LOOKBACK_DAYS = 5
# 不做成本止损：用户口径是固定持有到第3天收盘卖出。
STOP_LOSS = None
# 建仓日 hold_days=0，下一交易日变成 1 → 第3天（买入次一交易日）收盘离场。
MAX_HOLD_DAYS = 1
# C 口径优化：次日开盘 gap >= 3% 不买（回测引擎在入场日按 open/前收判断）。
# 设为 None 可关闭，回到纯 B 口径。
ENTRY_MAX_GAP_PCT = 0.03
# B 口径：只在震荡/偏弱情绪阶段交易。
REGIME_STATES = ("range", "lean_weak")

REQUIRED_FEATURES = {
    "consecutive_limit_ups",
    "amount",
    "turnover_rate",
    "float_shares",
}

RULES = """
1. 信号日（第1天）按概念/行业统计当日涨停家数（连板数>=1），取涨停家数最多的板块。
2. 板块内选创业板（300/301）且连板数=1（低于2板）的个股，按成交额最大取一只，
   每个信号日最多一只（可切换成交额最小/流通市值最小/换手率最高）。
3. B 口径：只在市场情绪状态 range/lean_weak 时交易；可在参数里关闭。
4. 信号日次日（第2天）以开盘价买入；C 口径：次日开盘 gap>=3% 不买。
5. 第3天（买入次一交易日）以收盘价卖出，固定持有 1 个交易日，不设成本止损。
6. 概念/行业成分为当前快照回看历史，存在归属漂移，越近越准。
"""

_BOARD_PREFIXES = ("300", "301")
_PICK_AMOUNT_MIN = "成交额最小"
_PICK_AMOUNT_MAX = "成交额最大"
_PICK_FLOAT_CAP = "流通市值最小"
_PICK_TURNOVER = "换手率最高"


def _normalize_members(kind: str) -> pl.DataFrame:
    """当前概念/行业成分（symbol 大写）；行业只取前两级，与主线口径一致。"""
    members = get_dimension_members(kind)
    if members.is_empty():
        return members
    member_expr = (
        pl.col("member").str.split("-").list.head(2).list.join("-")
        if kind == "industry"
        else pl.col("member")
    )
    return (
        members.with_columns(
            pl.col("symbol").str.to_uppercase(),
            member_expr.alias("member"),
        )
        .drop_nulls()
        .unique()
    )


def _top_sector_per_day(
    panel: pl.DataFrame, members: pl.DataFrame
) -> pl.DataFrame:
    """每日涨停家数最多的板块（date/member 两列）。

    一只股票属于多个概念时，每个概念各自计数，这是板块涨停家数的正常口径。
    """
    if panel.is_empty() or members.is_empty():
        return pl.DataFrame(schema={"date": panel.schema.get("date"), "member": pl.Utf8})
    limit_hits = (
        panel.filter(pl.col("consecutive_limit_ups") >= 1)
        .select(["symbol", "date"])
        .with_columns(pl.col("symbol").str.to_uppercase())
    )
    counts = (
        limit_hits.join(members, on="symbol", how="inner")
        .group_by(["date", "member"])
        .agg(pl.len().alias("limit_up_count"))
    )
    # 先按涨停数降序、板块名升序排序，再按日取第一条 → 并列时结果稳定可复现。
    return (
        counts.sort(
            ["date", "limit_up_count", "member"],
            descending=[False, True, False],
        )
        .group_by("date", maintain_order=True)
        .head(1)
        .select(["date", "member"])
    )


def _pick_one_per_day(candidates: pl.DataFrame, pick_by: str) -> pl.DataFrame:
    """每个信号日只留一只：按选定代理指标排序，并列时按代码升序。"""
    if candidates.is_empty():
        return candidates
    if pick_by == _PICK_FLOAT_CAP:
        candidates = candidates.with_columns(
            (pl.col("close") * pl.col("float_shares")).alias("_pick_metric")
        )
        descending = False
    elif pick_by == _PICK_TURNOVER:
        candidates = candidates.with_columns(pl.col("turnover_rate").alias("_pick_metric"))
        descending = True
    elif pick_by == _PICK_AMOUNT_MAX:
        candidates = candidates.with_columns(pl.col("amount").alias("_pick_metric"))
        descending = True
    else:  # 成交额最小（默认）
        candidates = candidates.with_columns(pl.col("amount").alias("_pick_metric"))
        descending = False
    return (
        candidates.sort(
            ["date", "_pick_metric", "symbol"],
            descending=[False, descending, False],
            nulls_last=True,
        )
        .group_by("date", maintain_order=True)
        .head(1)
        .drop("_pick_metric")
    )


def filter_history(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    """返回信号日候选（每个信号日最多一只）；次日开盘买入由回测口径完成。"""
    if df.is_empty() or "date" not in df.columns:
        return df

    required = {
        "symbol",
        "date",
        "open",
        "close",
        "amount",
        "turnover_rate",
        "float_shares",
        "consecutive_limit_ups",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("20cm套利策略缺少字段: " + ", ".join(missing))

    kind = "industry" if str(params.get("sector_kind") or "") == "行业板块" else "concept"
    members = _normalize_members(kind)
    if members.is_empty():
        return df.clear()

    panel = df.with_columns(pl.col("symbol").str.to_uppercase())
    top = _top_sector_per_day(panel, members)
    if top.is_empty():
        return df.clear()

    first_board = panel.filter(
        (pl.col("consecutive_limit_ups") == 1)
        & pl.any_horizontal(
            [pl.col("symbol").str.starts_with(prefix) for prefix in _BOARD_PREFIXES]
        )
    )
    if first_board.is_empty():
        return df.clear()

    candidates = (
        first_board.join(members, on="symbol", how="inner")
        .join(top, on=["date", "member"], how="inner")
        .select(df.columns)
    )
    picked = _pick_one_per_day(
        candidates, str(params.get("pick_by") or _PICK_AMOUNT_MAX)
    ).select(df.columns)

    use_regime = params.get("use_regime_filter", True)
    if isinstance(use_regime, str):
        use_regime = use_regime.strip().lower() == "true"
    if use_regime:
        regime = get_market_regime(columns=["date", "state"])
        if regime.is_empty() or "state" not in regime.columns:
            return df.clear()
        picked = (
            picked.join(regime, on="date", how="inner")
            .filter(pl.col("state").is_in(list(REGIME_STATES)))
            .select(df.columns)
        )
    return picked.sort(["date", "symbol"])
