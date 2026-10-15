"""标的维表全量覆盖写的健全性护栏。

2026-10-14 实例: `tf.exchanges.get_instruments("SZ")` 失败后只记了一条 warning, 循环照常
结束, 2,817 行的残缺维表仍被覆盖上去 —— 深市 2,905 只标的从维表里静默消失, 连带
total_shares 缺失 → market_cap 为 NaN → 选股/回测把整个深市剔除 (笔数腰斩)。

覆盖写是全量的, 所以「少一所」等于「删一所」。这里锁定: 任一交易所拉取失败或返回空,
或新数据相对既有维表缺少某个交易所 / 行数暴跌时, 都必须放弃本次覆盖并保留既有维表;
同时不得把正常的新增退市 (小幅变化) 也挡掉。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from app.services import instrument_sync

_TABLE = Path("instruments") / "instruments.parquet"


class _FakeExchanges:
    """get_instruments 桩: 每所的值可以是 Instrument 列表, 也可以是待抛出的异常。"""

    def __init__(self, per_exchange: dict[str, object]) -> None:
        self._per_exchange = per_exchange

    def get_instruments(self, exchange: str, instrument_type: str = "stock") -> list[dict]:
        result = self._per_exchange[exchange]
        if isinstance(result, BaseException):
            raise result
        assert isinstance(result, list)
        return result


@pytest.fixture
def tickflow(monkeypatch: pytest.MonkeyPatch):
    """让 sync_instruments 走 tickflow 直连分支, 并把 get_client 换成桩。"""
    monkeypatch.setattr(instrument_sync, "_fetch_instruments_via_provider", lambda: None)

    def _use(per_exchange: dict[str, object]) -> None:
        monkeypatch.setattr(
            instrument_sync,
            "get_client",
            lambda: SimpleNamespace(exchanges=_FakeExchanges(per_exchange)),
        )

    return _use


def _items(exchange: str, n: int) -> list[dict]:
    """n 条 SDK Instrument 形状的记录。"""
    base = {"SH": 600000, "SZ": 1, "BJ": 830000}[exchange]
    return [
        {
            "symbol": f"{base + i:06d}.{exchange}",
            "name": f"{exchange}{i:04d}",
            "code": f"{base + i:06d}",
            "exchange": exchange,
            "region": "CN",
            "type": "stock",
            "ext": {"total_shares": 1e8, "float_shares": 8e7, "limit_up": 11.0, "limit_down": 9.0},
        }
        for i in range(n)
    ]


def _flat(exchange: str, n: int) -> list[dict]:
    """落表形状的记录 (非 tickflow provider 分支的返回值)。"""
    return instrument_sync._flatten_instruments(_items(exchange, n))


def _flat_all(counts: dict[str, int]) -> list[dict]:
    return [row for ex, n in counts.items() for row in _flat(ex, n)]


def _read(tmp_path: Path) -> pl.DataFrame:
    return pl.read_parquet(tmp_path / _TABLE)


def _seed(tmp_path: Path, counts: dict[str, int]) -> Path:
    """铺一张既有维表 (只需 symbol/exchange, 护栏看的就是这两列 + 行数)。"""
    path = tmp_path / _TABLE
    path.parent.mkdir(parents=True, exist_ok=True)
    symbols: list[str] = []
    exchanges: list[str] = []
    for ex, n in counts.items():
        for i in range(n):
            symbols.append(f"{i:06d}.{ex}")
            exchanges.append(ex)
    pl.DataFrame({"symbol": symbols, "exchange": exchanges}).write_parquet(path)
    return path


@pytest.mark.parametrize("mode", ["raises", "empty"])
def test_single_exchange_failure_keeps_existing_table(tmp_path: Path, tickflow, mode: str) -> None:
    """深市拉取失败/返回空时, 不得用残缺数据覆盖维表 (2026-10-14 事故)。"""
    _seed(tmp_path, {"SH": 2400, "SZ": 900, "BJ": 100})
    before = _read(tmp_path)
    per_exchange: dict[str, object] = {"SH": _items("SH", 2400), "BJ": _items("BJ", 100)}
    per_exchange["SZ"] = RuntimeError("simulated SZ timeout") if mode == "raises" else []
    tickflow(per_exchange)

    assert instrument_sync.sync_instruments(tmp_path) == 0
    after = _read(tmp_path)
    assert_frame_equal(after, before)
    assert set(after["exchange"].unique().to_list()) == {"SH", "SZ", "BJ"}


def test_all_exchanges_fetched_writes_full_table(tmp_path: Path, tickflow) -> None:
    _seed(tmp_path, {"SH": 2400, "SZ": 900, "BJ": 100})
    tickflow({"SH": _items("SH", 2400), "SZ": _items("SZ", 900), "BJ": _items("BJ", 100)})

    assert instrument_sync.sync_instruments(tmp_path) == 3400
    df = _read(tmp_path)
    assert set(df["exchange"].unique().to_list()) == {"SH", "SZ", "BJ"}
    assert df["as_of"].n_unique() == 1
    assert df["total_shares"].null_count() == 0


def test_first_sync_without_existing_table_writes(tmp_path: Path, tickflow) -> None:
    tickflow({"SH": _items("SH", 2400), "SZ": _items("SZ", 900), "BJ": _items("BJ", 100)})

    assert instrument_sync.sync_instruments(tmp_path) == 3400
    assert set(_read(tmp_path)["exchange"].unique().to_list()) == {"SH", "SZ", "BJ"}


def test_full_fetch_repairs_previously_incomplete_table(tmp_path: Path, tickflow) -> None:
    """既有维表本身就是残缺的 (无 SZ) 时, 正常的全量数据必须能写进去。"""
    _seed(tmp_path, {"SH": 2400, "BJ": 100})
    tickflow({"SH": _items("SH", 2400), "SZ": _items("SZ", 900), "BJ": _items("BJ", 100)})

    assert instrument_sync.sync_instruments(tmp_path) == 3400
    assert set(_read(tmp_path)["exchange"].unique().to_list()) == {"SH", "SZ", "BJ"}


def test_provider_result_missing_exchange_keeps_existing_table(tmp_path: Path, monkeypatch) -> None:
    """非 tickflow provider 分支同样受护栏保护 (它拿不到逐所的成功与否)。"""
    _seed(tmp_path, {"SH": 2400, "SZ": 900})
    before = _read(tmp_path)
    monkeypatch.setattr(
        instrument_sync,
        "_fetch_instruments_via_provider",
        lambda: _flat_all({"SH": 2400}),
    )

    assert instrument_sync.sync_instruments(tmp_path) == 0
    assert_frame_equal(_read(tmp_path), before)


def test_catastrophic_shrink_keeps_existing_table(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path, {"SH": 2400, "SZ": 900, "BJ": 100})
    before = _read(tmp_path)
    monkeypatch.setattr(
        instrument_sync,
        "_fetch_instruments_via_provider",
        lambda: _flat_all({"SH": 20, "SZ": 10, "BJ": 5}),
    )

    assert instrument_sync.sync_instruments(tmp_path) == 0
    assert_frame_equal(_read(tmp_path), before)


def test_normal_day_over_day_change_still_writes(tmp_path: Path, monkeypatch) -> None:
    """护栏不能挡掉正常的新增/退市 (行数小幅波动)。"""
    _seed(tmp_path, {"SH": 2400, "SZ": 900, "BJ": 100})
    monkeypatch.setattr(
        instrument_sync,
        "_fetch_instruments_via_provider",
        lambda: _flat_all({"SH": 2390, "SZ": 880, "BJ": 100}),
    )

    assert instrument_sync.sync_instruments(tmp_path) == 3370
    assert _read(tmp_path).height == 3370
