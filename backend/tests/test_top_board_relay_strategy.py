from __future__ import annotations

from datetime import date

import polars as pl

from app.backtest.matrix import (
    MatrixPipelineConfig,
    MatrixStrategyPipeline,
    build_market_data_matrix,
)
from app.strategy.builtin import top_board_relay

DAY1 = date(2024, 1, 2)
DAY2 = date(2024, 1, 3)

_SHAPES = {
    "swing": lambda close: (close * 0.97, close, close * 0.96),
    "one_price": lambda close: (close, close, close),
    "t_open": lambda close: (close, close, close * 0.95),
}


def _market(rows: list[dict]):
    """rows 每项: symbol / date / boards / turnover / close / locked
                       / [name] / [high] / [shape]

    name  覆盖股票名(默认用 symbol), 用于风险警示股(ST)场景。
    high  覆盖当日最高价, 用于「盘中触板但收盘未封住」场景。
    shape 决定当日 K 线形态(默认 swing):
      swing     — 开高低收不等价, 盘中可成交的换手板
      one_price — 全天单一价格, 一字板(撮合层拒绝买入, 策略按「一字顶死」剔除)
      t_open    — 一字开盘、盘中开板后回封的 T 字板(仍可成交, 不剔除)

    不复权原始价与复权价取同一组数值(测试数据无除权), 使「触板」判定简化为
    「当日最高价 == 上一日收盘 × (1+涨跌幅)」。
    """
    panel = []
    for row in rows:
        open_, high, low = _SHAPES[row.get("shape", "swing")](row["close"])
        high = row.get("high", high)
        panel.append({
            "symbol": row["symbol"],
            "name": row.get("name", row["symbol"]),
            "date": row["date"],
            "open": open_,
            "high": high,
            "low": low,
            "close": row["close"],
            "volume": 1_000_000.0,
            "signal_limit_up": row["locked"],
            "signal_limit_down": False,
            "amount": row.get("amount", 1e8),
            "consecutive_limit_ups": row["boards"],
            "turnover_rate": row["turnover"],
            "raw_close": row["close"],
            "raw_high": high,
        })
    return build_market_data_matrix(
        pl.DataFrame(panel),
        field_columns={
            "amount",
            "consecutive_limit_ups",
            "turnover_rate",
            "raw_close",
            "raw_high",
        },
    )


def _selected(market, params: dict | None = None) -> list[str]:
    signals = top_board_relay.MATRIX_STRATEGY.compute_signals(market, params or {})
    return [s for s, hit in zip(market.symbols, signals.entry[-1], strict=True) if hit]


def test_leader_sealing_next_day_triggers_entry():
    market = _market([
        {"symbol": "000001.SZ", "date": DAY1, "boards": 5, "turnover": 5.0, "close": 10.0, "locked": True},
        {"symbol": "600000.SH", "date": DAY1, "boards": 4, "turnover": 8.0, "close": 20.0, "locked": True},
        {"symbol": "000001.SZ", "date": DAY2, "boards": 6, "turnover": 5.0, "close": 11.0, "locked": True},
        {"symbol": "600000.SH", "date": DAY2, "boards": 5, "turnover": 8.0, "close": 22.0, "locked": True},
    ])

    # 只有昨日最高板(5板)今日进6板才打; 4板那只进5板不是当日梯队最高, 不入场
    assert _selected(market) == ["000001.SZ"]


def test_leader_not_sealing_next_day_has_no_entry():
    market = _market([
        {"symbol": "000001.SZ", "date": DAY1, "boards": 5, "turnover": 5.0, "close": 10.0, "locked": True},
        # 次日最高价 10.2 远低于涨停价 11.0 => 没触板, 不进板
        {"symbol": "000001.SZ", "date": DAY2, "boards": 0, "turnover": 5.0, "close": 10.2, "locked": False},
    ])

    assert _selected(market) == []


def test_tied_top_boards_pick_highest_turnover():
    market = _market([
        {"symbol": "000001.SZ", "date": DAY1, "boards": 4, "turnover": 3.0, "close": 10.0, "locked": True},
        {"symbol": "600000.SH", "date": DAY1, "boards": 4, "turnover": 3.0, "close": 20.0, "locked": True},
        {"symbol": "000001.SZ", "date": DAY2, "boards": 5, "turnover": 3.0, "close": 11.0, "locked": True},
        {"symbol": "600000.SH", "date": DAY2, "boards": 5, "turnover": 9.0, "close": 22.0, "locked": True},
    ])

    assert _selected(market) == ["600000.SH"]


def test_top_board_below_min_boards_has_no_entry():
    rows = [
        {"symbol": "000001.SZ", "date": DAY1, "boards": 1, "turnover": 5.0, "close": 10.0, "locked": True},
        {"symbol": "000001.SZ", "date": DAY2, "boards": 2, "turnover": 5.0, "close": 11.0, "locked": True},
    ]

    # 全市场最高只有首板 => 不构成连板梯队, 默认(min_top_boards=2)不出手
    assert _selected(_market(rows)) == []
    assert _selected(_market(rows), {"min_top_boards": 1}) == ["000001.SZ"]


def test_holding_exits_when_board_not_sealed():
    market = _market([
        {"symbol": "000001.SZ", "date": DAY1, "boards": 6, "turnover": 5.0, "close": 10.0, "locked": True},
        {"symbol": "000001.SZ", "date": DAY2, "boards": 0, "turnover": 5.0, "close": 10.2, "locked": False},
    ])
    signals = top_board_relay.MATRIX_STRATEGY.compute_signals(market, {})
    idx = market.symbols.index("000001.SZ")

    assert signals.exit[-1, idx] == 1


def test_holding_keeps_position_while_board_sealed():
    market = _market([
        {"symbol": "000001.SZ", "date": DAY1, "boards": 5, "turnover": 5.0, "close": 10.0, "locked": True},
        {"symbol": "000001.SZ", "date": DAY2, "boards": 6, "turnover": 5.0, "close": 11.0, "locked": True},
    ])
    signals = top_board_relay.MATRIX_STRATEGY.compute_signals(market, {})
    idx = market.symbols.index("000001.SZ")

    assert signals.exit[-1, idx] == 0


def test_engine_basic_filter_can_drop_the_only_entry():
    """基础过滤在 compute_signals 之后生效: 龙头被基本面条件滤掉 => 当日不出手。"""
    market = _market([
        {"symbol": "000001.SZ", "date": DAY1, "boards": 5, "turnover": 5.0, "close": 10.0, "locked": True},
        {"symbol": "000001.SZ", "date": DAY2, "boards": 6, "turnover": 5.0, "close": 11.0, "locked": True},
    ])

    def _entry(amount_min: float):
        return MatrixStrategyPipeline().run(
            top_board_relay.MATRIX_STRATEGY,
            market,
            {},
            MatrixPipelineConfig(
                basic_filter={"enabled": True, "amount_min": amount_min},
                scoring={"close": 1.0},
                order_by="score",
                descending=True,
            ),
        ).entry

    # 成交额 1 亿: 门槛 2 亿时被基础过滤剔除, 门槛 0.5 亿时保留
    assert _entry(2e8).tolist() == [[0], [0]]
    assert _entry(5e7).tolist() == [[0], [1]]


def test_one_price_top_board_is_skipped_for_next_board():
    """最高标一字顶死买不进去 => 不进梯队, 龙头落到次高标, 由次高标次日进板触发买入。"""
    market = _market([
        # 5 板且一字顶死: 买不进去, 当最高标没有接力意义
        {"symbol": "000001.SZ", "date": DAY1, "boards": 5, "turnover": 9.0, "close": 10.0,
         "locked": True, "shape": "one_price"},
        # 次高标 4 板, 可成交 => 当日龙头
        {"symbol": "600000.SH", "date": DAY1, "boards": 4, "turnover": 3.0, "close": 20.0, "locked": True},
        {"symbol": "000001.SZ", "date": DAY2, "boards": 6, "turnover": 9.0, "close": 11.0, "locked": True},
        {"symbol": "600000.SH", "date": DAY2, "boards": 5, "turnover": 3.0, "close": 22.0, "locked": True},
    ])

    assert _selected(market) == ["600000.SH"]
    # 关掉一字剔除后, 一字顶死的 5 板重新当龙头
    assert _selected(market, {"skip_one_price_leader": False}) == ["000001.SZ"]


def test_intraday_opened_board_is_not_one_price():
    """盘中开过板再回封(T 字板)不算一字顶死: 买得进去, 仍作为最高标接力。"""
    market = _market([
        # 开=高=收 都在涨停价, 但盘中最低价明显低于涨停 => T 字板
        {"symbol": "000001.SZ", "date": DAY1, "boards": 5, "turnover": 3.0, "close": 10.0,
         "locked": True, "shape": "t_open"},
        {"symbol": "600000.SH", "date": DAY1, "boards": 4, "turnover": 9.0, "close": 20.0, "locked": True},
        {"symbol": "000001.SZ", "date": DAY2, "boards": 6, "turnover": 3.0, "close": 11.0, "locked": True},
        {"symbol": "600000.SH", "date": DAY2, "boards": 5, "turnover": 9.0, "close": 22.0, "locked": True},
    ])

    assert _selected(market) == ["000001.SZ"]


def test_top_board_above_cap_is_skipped_for_next_board():
    """最高标板数超过上限(默认 5 板)不进梯队 => 看次高标。"""
    market = _market([
        # 9 板: 位置太高没有博弈空间, 不参与接力
        {"symbol": "000001.SZ", "date": DAY1, "boards": 9, "turnover": 9.0, "close": 10.0, "locked": True},
        {"symbol": "600000.SH", "date": DAY1, "boards": 4, "turnover": 3.0, "close": 20.0, "locked": True},
        {"symbol": "000001.SZ", "date": DAY2, "boards": 10, "turnover": 9.0, "close": 11.0, "locked": True},
        {"symbol": "600000.SH", "date": DAY2, "boards": 5, "turnover": 3.0, "close": 22.0, "locked": True},
    ])

    assert _selected(market) == ["600000.SH"]
    # 上限放宽到 9 板后, 9 板那只重回梯队并当龙头
    assert _selected(market, {"max_top_boards": 9}) == ["000001.SZ"]


def test_risk_warning_leader_is_skipped_for_next_leader():
    """ST 占着最高标会让真正的最高标(非 ST)顶不上来, 次日买不到 —— 默认剔除 ST。

    回归: 2026-07-14 最高标是 ST 龙元(5 板), 哈药股份(3 板)才是非 ST 的最高标。
    剔除 ST 后哈药成为当日龙头, 07-15 触板即买入(旧口径要等到 07-16)。
    """
    market = _market([
        {"symbol": "600491.SH", "name": "ST龙元", "date": DAY1, "boards": 5, "turnover": 12.5,
         "close": 10.0, "locked": True},
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY1, "boards": 3, "turnover": 5.0,
         "close": 10.0, "locked": True},
        {"symbol": "600491.SH", "name": "ST龙元", "date": DAY2, "boards": 6, "turnover": 12.5,
         "close": 11.0, "locked": True},
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY2, "boards": 4, "turnover": 5.0,
         "close": 11.0, "locked": True},
    ])

    # 默认剔除 ST: 龙头是哈药股份, 次日哈药进板 => 打哈药
    assert _selected(market) == ["600664.SH"]
    # 关掉剔除 => 回到旧口径, 龙头变成 ST 龙元
    assert _selected(market, {"skip_st": False}) == ["600491.SH"]


def test_touched_but_not_sealed_leader_does_not_enter_by_default():
    """默认必须收盘封板才买: 盘中触板但收盘没封住(炸板)不买。

    实测(2024-01..2026-10, 全市场, max_positions=10): 放宽成「触板即买」把累计收益从
    +90.8% 打到 -41.7%, 胜率 46.0%→33.6%, 最大回撤 -7.6%→-43.9%。触板未封的板大多是
    「炸板」弱板, 按涨停价买进去等于买在情绪顶 —— 故默认保留「封板才买」, 触板仅作可选开关。
    """
    market = _market([
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY1, "boards": 3, "turnover": 5.0,
         "close": 10.0, "locked": True},
        # 涨停价 11.00: 最高价摸到 11.00(触板), 收盘回落到 10.50(未封住)
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY2, "boards": 0, "turnover": 5.0,
         "close": 10.5, "high": 11.0, "locked": False},
    ])

    assert _selected(market) == []
    # 显式关掉才放宽为「触板即买」
    assert _selected(market, {"require_seal": False}) == ["600664.SH"]


def test_touch_below_limit_price_is_not_an_entry():
    """盘中最高价差一分钱没摸到涨停价 => 不算触板。"""
    market = _market([
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY1, "boards": 3, "turnover": 5.0,
         "close": 10.0, "locked": True},
        # 涨停价 11.00, 最高价只有 10.99
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY2, "boards": 0, "turnover": 5.0,
         "close": 10.8, "high": 10.99, "locked": False},
    ])

    assert _selected(market) == []


def test_meta_declares_limit_price_fill():
    """打板必须按信号日当天、涨停价成交: 策略自带口径声明。

    建仓用「当日最高价」(entry_fill=high_t): 封板日 high == close == 涨停价, 与旧的
    「信号日收盘」完全等价; 触板未封日 high == 涨停价而 close 已回落, 只有按最高价成交
    才等于打板单的真实成交价(按收盘成交会系统性高估收益)。
    清仓仍用「信号日收盘」: 尾盘未封板时按当日收盘离场。
    """
    assert top_board_relay.META["backtest_defaults"] == {
        "entry_fill": "high_t",
        "exit_fill": "close_t",
    }


def test_params_expose_new_switches():
    """两个新增开关必须出现在 META.params 里(前端按 META 渲染, 用户可自行改口径)。

    默认值都是**实测占优**的一侧: 剔 ST 为 True(显著提高收益/降低回撤),
    require_seal 为 True(放宽触板会大幅亏损, 见上一个用例)。
    """
    params = {p["id"]: p for p in top_board_relay.META["params"]}

    assert params["skip_st"]["type"] == "bool"
    assert params["skip_st"]["default"] is True
    assert params["require_seal"]["type"] == "bool"
    assert params["require_seal"]["default"] is True


def test_module_declares_prev_close_stop_constants():
    """退出规则(二)「杀到绿盘立刻走」交给引擎的风控线实现: 策略只声明两个止损锚点。

    引擎取两条止损线里**较高**的一条, 且风控块排在 max_high 更新之前, 于是
    持有第 1 天两条线都落在买入价(= 前收盘价)上, 第 2 天起落在上一交易日的涨停收盘价
    (= 前收盘价)上 —— 合起来就是「跌破前收盘价」。停牌/跌停卖不出时引擎会挂起卖出意向。
    STOP_LOSS=0.0 锚在买入价(引擎对 stop_loss 无下限钳制);
    TRAILING_STOP=-0.005 而非 0, 因为引擎对 trailing_stop 有 0.005 的下限钳制。
    """
    assert top_board_relay.STOP_LOSS == 0.0
    assert top_board_relay.TRAILING_STOP == -0.005


def test_weak_seal_high_turnover_exits_even_when_board_sealed():
    """规则(三): 当天收盘封板, 但换手率 >= 40% 时仍以当日收盘价离场。

    原口径还要求「全天炸板多次」, 但数据里没有开板次数/炸板次数字段 ——
    signal_broken_limit_up 只在「触板未封」时为真, 收盘封住的当天即使盘中炸过多次也是
    False。故只保留可观测的「封板 + 放量换手 + 换手率 >= 阈值」这一半。
    """
    market = _market([
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY1, "boards": 5, "turnover": 12.0,
         "close": 10.0, "locked": True},
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY2, "boards": 6, "turnover": 45.0,
         "close": 11.0, "locked": True},
    ])
    idx = market.symbols.index("600664.SH")
    signals = top_board_relay.MATRIX_STRATEGY.compute_signals(market, {})

    # 封板了, 但放量换手 => 仍然离场(按当日收盘价)
    assert signals.exit[-1, idx] == 1

    # 关掉该规则后, 封板就继续持有(只剩规则(一))
    off = top_board_relay.MATRIX_STRATEGY.compute_signals(market, {"exit_weak_seal": False})
    assert off.exit[-1, idx] == 0


def test_weak_seal_below_threshold_keeps_position():
    """换手率没到阈值(39.9% < 40%)且封板 => 继续持有。"""
    market = _market([
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY1, "boards": 5, "turnover": 12.0,
         "close": 10.0, "locked": True},
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY2, "boards": 6, "turnover": 39.9,
         "close": 11.0, "locked": True},
    ])
    idx = market.symbols.index("600664.SH")
    signals = top_board_relay.MATRIX_STRATEGY.compute_signals(market, {})

    assert signals.exit[-1, idx] == 0


def test_weak_seal_turnover_threshold_is_configurable():
    """阈值可调: 换手 25% 在默认 40% 下继续持有, 调到 20% 则离场。"""
    market = _market([
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY1, "boards": 5, "turnover": 12.0,
         "close": 10.0, "locked": True},
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY2, "boards": 6, "turnover": 25.0,
         "close": 11.0, "locked": True},
    ])
    idx = market.symbols.index("600664.SH")

    assert top_board_relay.MATRIX_STRATEGY.compute_signals(market, {}).exit[-1, idx] == 0
    looser = top_board_relay.MATRIX_STRATEGY.compute_signals(
        market, {"weak_seal_turnover": 20.0}
    )
    assert looser.exit[-1, idx] == 1


def test_weak_seal_nan_turnover_does_not_exit():
    """换手率缺失(NaN)时不触发规则(三): 缺数据不等于「没放量」, 保守方向是不卖。

    矩阵里 turnover_rate 由 float_shares 派生, 新股/缺股本数据的票会是 NaN。
    """
    market = _market([
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY1, "boards": 5, "turnover": 12.0,
         "close": 10.0, "locked": True},
        {"symbol": "600664.SH", "name": "哈药股份", "date": DAY2, "boards": 6,
         "turnover": float("nan"), "close": 11.0, "locked": True},
    ])
    idx = market.symbols.index("600664.SH")
    signals = top_board_relay.MATRIX_STRATEGY.compute_signals(market, {})

    assert signals.exit[-1, idx] == 0


def test_params_expose_exit_switches():
    """两个新增的退出开关必须出现在 META.params 里(前端按 META 渲染)。

    exit_weak_seal 默认 True, weak_seal_turnover 默认 40.0 —— 单位是百分比
    (矩阵换手率由 volume(手)*10000/float_shares(股) 得出), 40% 对应 40.0。
    """
    params = {p["id"]: p for p in top_board_relay.META["params"]}

    assert params["exit_weak_seal"]["type"] == "bool"
    assert params["exit_weak_seal"]["default"] is True
    assert params["weak_seal_turnover"]["type"] == "float"
    assert params["weak_seal_turnover"]["default"] == 40.0
