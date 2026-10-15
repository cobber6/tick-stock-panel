"""标的维表同步服务。

盘前 9:10 调用 tf.exchanges.get_instruments("SH"/"SZ"/"BJ", type="stock")
获取全量标的元数据，flatten ext 字段，写入 instruments.parquet。

Starter+ 盘后可用 quotes.get(universes) 顺便补充 name。
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl

from app.services.fs_utils import atomic_write_parquet
from app.tickflow.client import get_client

logger = logging.getLogger(__name__)

_EXCHANGES = ["SH", "SZ", "BJ"]

# 覆盖写前的健全性护栏(2026-10-14 深市 2905 只被残缺覆盖丢失后补):
# 新数据不得缺少既有维表里出现过的交易所; 既有维表规模可观时, 新行数也不得跌破该比例。
# 阈值刻意宽松 —— 只挡「整体丢所/大规模丢行」这类灾难, 不挡正常的新增退市。
_MIN_ROWS_RATIO = 0.5
_SHRINK_GUARD_MIN_ROWS = 1000


def _flatten_instruments(items: list[dict]) -> list[dict]:
    """把 SDK 返回的 Instrument 列表 flatten 成扁平行。"""
    rows = []
    for item in items:
        row = {
            "symbol": item.get("symbol"),
            "name": item.get("name"),
            "code": item.get("code"),
            "exchange": item.get("exchange"),
            "region": item.get("region"),
            "type": item.get("type"),
        }
        ext = item.get("ext") or {}
        row["listing_date"] = ext.get("listing_date")
        row["total_shares"] = ext.get("total_shares")
        row["float_shares"] = ext.get("float_shares")
        row["tick_size"] = ext.get("tick_size")
        row["limit_up"] = ext.get("limit_up")
        row["limit_down"] = ext.get("limit_down")
        rows.append(row)
    return rows


def _fetch_instruments_via_provider() -> list[dict] | None:
    """若当前日K数据源不是 tickflow 且该 provider 提供 get_instruments, 用它拉标的维表。

    返回 flatten 行列表; 未命中(仍应走 tickflow)时返回 None。
    标的维表跟随日K数据源(二者天然耦合, 无独立偏好项)。
    """
    from app.services import preferences

    provider_name = preferences.get_daily_data_provider()
    if provider_name == "tickflow":
        return None
    from app.data_providers import custom as custom_sources

    if not custom_sources.is_custom_provider(provider_name):
        return None
    provider = custom_sources.get_provider(provider_name)
    if not hasattr(provider, "get_instruments"):
        return None
    try:
        items = provider.get_instruments("stock") or []
    except Exception as e:  # noqa: BLE001
        logger.warning("provider %s get_instruments 失败: %s", provider_name, e)
        return None
    rows = _flatten_instruments(items)
    logger.info("instruments via %s: %d stocks", provider_name, len(rows))
    return rows


def _fetch_via_tickflow() -> list[dict] | None:
    """逐交易所直连拉取标的维表。

    任一交易所失败或返回空即返回 None(放弃本次同步) —— 覆盖写是全量的, 少一所等于把
    那个市场的标的从维表里删掉, 宁可不更新。
    """
    tf = get_client()
    all_rows: list[dict] = []
    for ex in _EXCHANGES:
        try:
            items = tf.exchanges.get_instruments(ex, instrument_type="stock")
        except Exception as e:  # noqa: BLE001
            logger.error("get_instruments(%s) failed, 放弃本次维表同步(保留现有维表): %s", ex, e)
            return None
        if not items:
            logger.error("get_instruments(%s) 返回空, 放弃本次维表同步(保留现有维表)", ex)
            return None
        all_rows.extend(_flatten_instruments(items))
        logger.info("instruments %s: %d stocks", ex, len(items))
    return all_rows


def _overwrite_hazard(out: Path, new_df: pl.DataFrame) -> str | None:
    """覆盖写前的健全性校验: 返回拒绝覆盖的原因, None 表示可以写。"""
    if not out.exists():
        return None
    try:
        old = pl.read_parquet(out)
    except Exception as e:  # noqa: BLE001
        logger.warning("读取既有维表失败, 跳过覆盖前校验: %s", e)
        return None

    if "exchange" in old.columns and "exchange" in new_df.columns:
        lost = sorted(
            set(old["exchange"].drop_nulls().unique().to_list())
            - set(new_df["exchange"].drop_nulls().unique().to_list()),
        )
        if lost:
            return f"新数据缺少既有交易所 {lost}"
    if old.height >= _SHRINK_GUARD_MIN_ROWS and new_df.height < old.height * _MIN_ROWS_RATIO:
        return f"行数由 {old.height} 跌至 {new_df.height} (不足 {_MIN_ROWS_RATIO:.0%})"
    return None


def sync_instruments(data_dir: Path) -> int:
    """全量同步标的维表 → data/instruments/instruments.parquet。

    返回写入的行数; 0 表示未写入 —— 拉取失败或未通过覆盖前校验时保留现有维表原样。
    """
    all_rows = _fetch_instruments_via_provider()
    if all_rows is None:
        # 未命中非 tickflow provider → 走 tickflow 直连
        all_rows = _fetch_via_tickflow()
        if all_rows is None:
            return 0

    if not all_rows:
        return 0

    df = pl.DataFrame(all_rows)
    df = df.with_columns(pl.lit(date.today()).alias("as_of"))

    out = data_dir / "instruments" / "instruments.parquet"
    hazard = _overwrite_hazard(out, df)
    if hazard:
        logger.error("维表疑似残缺(%s), 放弃覆盖 %s, 保留现有维表", hazard, out)
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(df, out)

    logger.info("instruments synced: %d rows → %s", df.height, out)
    return df.height


def enrich_names_from_quotes(
    data_dir: Path,
    quotes_data: list[dict],
) -> int:
    """从 quotes 响应中提取 name，更新 instruments 维表（兜底补充）。

    盘后 quotes.get(universes) 返回的数据中包含 ext.name，
    用来补充 instruments 中可能缺失的 name。
    """
    if not quotes_data:
        return 0

    # 构建 symbol → name 映射
    name_map: dict[str, str] = {}
    for q in quotes_data:
        symbol = q.get("symbol", "")
        ext = q.get("ext") or {}
        name = ext.get("name") or q.get("name", "")
        if symbol and name:
            name_map[symbol] = name

    if not name_map:
        return 0

    inst_path = data_dir / "instruments" / "instruments.parquet"
    if not inst_path.exists():
        return 0

    df = pl.read_parquet(inst_path)

    # 只更新空 name 的行
    updates = pl.DataFrame({
        "symbol": list(name_map.keys()),
        "_new_name": list(name_map.values()),
    })
    df = df.join(updates, on="symbol", how="left")
    df = df.with_columns(
        pl.when(pl.col("name").is_null() | (pl.col("name") == ""))
        .then(pl.col("_new_name"))
        .otherwise(pl.col("name"))
        .alias("name"),
    ).drop("_new_name")

    atomic_write_parquet(df, inst_path)
    logger.info("instruments name enriched from quotes: %d names", len(name_map))
    return len(name_map)
