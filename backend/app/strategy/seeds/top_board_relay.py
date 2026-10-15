"""最高板打板接力 — 当日可接力的最高连板股, 次日封板即打板买入; 持有期跌破前收即离场, 未封板则卖出"""

from __future__ import annotations

from datetime import date

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    matrix_feature,
)
from app.price_limits import (
    MAIN_BOARD_ST_LIMIT_CHANGE_DATE,
    is_risk_warning_name,
    numpy_limit_pct_vectors,
    numpy_limit_price,
)

META = {
    "id": "top_board_relay",
    "name": "最高板打板接力",
    "description": (
        "取全市场可接力的最高连板股(默认剔除风险警示股), 次日收盘封板时按涨停价打板买入; "
        "持有期跌破前收盘价即离场(低开则竞价就走), 收盘未封板则当日收盘卖出, "
        "封板但放量换手(换手率 >= 40%)同样当日收盘离场"
    ),
    "tags": ["涨停", "连板", "打板", "龙头"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    # 打板必须按「信号日(=触板/封板当日)」成交, 且成交价就是当日涨停价, 不能推迟到次日开盘。
    # 建仓用「当日最高价」: 封板日 high == close == 涨停价(与旧的「信号日收盘」完全一致);
    # 触板未封日 high == 涨停价而 close 已回落 —— 此时若按收盘成交, 等于用低于涨停价的
    # 价格买到打板单, 会系统性高估收益。清仓仍按「信号日收盘」。
    # 回测页默认口径是「次日开盘」, 会把打板单推迟到次日开盘(次日低开低走也照样成交)。
    # 前端选中本策略时自动套用, 用户仍可在回测页手动覆盖。
    "backtest_defaults": {"entry_fill": "high_t", "exit_fill": "close_t"},
    "params": [
        {
            "id": "min_top_boards",
            "label": "最高板最低板数",
            "type": "int",
            "default": 2,
            "min": 1,
            "max": 20,
            "step": 1,
        },
        {
            "id": "max_top_boards",
            "label": "最高板板数上限",
            "type": "int",
            "default": 5,
            "min": 1,
            "max": 30,
            "step": 1,
        },
        {"id": "skip_one_price_leader", "label": "一字顶死不接力", "type": "bool", "default": True},
        {"id": "skip_st", "label": "剔除风险警示股", "type": "bool", "default": True},
        {"id": "require_seal", "label": "仅收盘封板才买（关闭则触板即买）", "type": "bool", "default": True},
        {
            "id": "exit_weak_seal",
            "label": "放量换手涨停当日离场",
            "type": "bool",
            "default": True,
        },
        {
            "id": "weak_seal_turnover",
            "label": "放量换手离场换手率阈值（%）",
            "type": "float",
            "default": 40.0,
            "min": 5.0,
            "max": 100.0,
            "step": 1.0,
        },
    ],
    "scoring": {"consecutive_limit_ups": 0.4, "turnover_rate": 0.3, "amount": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = ["signal_top_board_relay_entry"]
EXIT_SIGNALS = ["signal_top_board_relay_exit"]

# ── 退出规则(二)「龙头杀到绿盘立刻走」────────────────────────────────────────
# 该规则必须在**触发价**上成交(低开按开盘价、盘中杀破按止损线), 而不是等到收盘 ——
# 这一点策略自己发出的 exit 信号做不到(信号只在收盘价成交), 但引擎的风控线天然支持:
# 引擎把下面两个常量换算成两条止损线, 取**较高**的一条作为实际止损价
#     stop_loss     = 买入价       * (1 - |STOP_LOSS|)       # 以买入价为锚
#     trailing_stop = 持仓期最高价 * (1 - |TRAILING_STOP|)   # 以最高价为锚
# 打板买入价 == 买入日收盘价 == 买入日涨停价; 而引擎判定风控时 pos["max_high"] 尚未纳入
# 当日 K 线(backtest/engine.py 的风控块排在 max_high 更新之前), 于是
#     持有第 1 天  : max_high == 买入价, 两条线都落在「前收盘价 == 买入价」上
#     持有第 2 天起: 只有封板才继续持有, 而封板日 high == close == 涨停价,
#                    故 max_high == 前一交易日的涨停收盘价 == 前收盘价
# 即两条线合起来正好是「跌破前收盘价就走」—— 用户口径里的「绿盘」。
# 成交价由引擎给出, 语义完全吻合(见 engine._simulate_portfolio_matrix 的风控块):
#     开盘价 <= 止损线 -> 按**开盘价**成交   (开盘就低开, 竞价就走)
#     最低价 <= 止损线 -> 按**止损线**成交   (盘中杀到绿盘马上走, 不等收盘)
# STOP_LOSS 取 0.0 而不是负数: 直接锚在买入价上, 恰好等于持有第 1 天的前收盘价, 且
# 引擎对 stop_loss 没有下限钳制(直接 abs())。取 0 会被 `is not None` 正常识别为「已启用」。
# TRAILING_STOP 取 -0.005 而不是 0: 引擎对 trailing_stop 有 0.005 的下限钳制
# (backtest/strategy.py _normalize_pct), 因此棘轮线实际落在前收盘价下方 0.5% 处。
# 停牌/跌停封死而卖不出时, 引擎会挂起卖出意向, 复牌或打开跌停后再按收盘价成交。
STOP_LOSS = 0.0
TRAILING_STOP = -0.005

RULES = """
1. 每个交易日先构造「可接力梯队」, 再取其中连板数最高的一只作为「最高板龙头」。
   进梯队的条件: 当日有行情、连板数不超过 max_top_boards(默认 5 板, 见第 3 条)、
   不是风险警示股(默认剔除, 见第 4 条)、且当日不是「一字顶死」(见第 2 条)。
   梯队最高板须达到 min_top_boards(默认 2 板), 达不到则当日不参与。
2. 一字顶死的股票不进梯队, 直接看次高标。一字顶死 = 当日开=高=低=收 且收盘封住涨停
   (口径与撮合层拒绝买入的判定一致): 这种板买不进去, 当最高标没有接力意义。
   判定基于**当日**形态 —— 前几日一字、当日已开板换手的(T 字板)仍算可接力。
3. 板数过高的最高标同样不进梯队(默认 >5 板)。连板数堆到 10 板以上的票, 位置太高、
   没有博弈空间, 用户口径是「完全没法下手」, 此时看次高标。
4. 风险警示股(ST/*ST)默认剔除出梯队。ST 的连板天梯与主板是两个博弈池, 「最高标」的
   口径按非 ST 股的连板高度算 —— 否则一只 ST 的 5 板会把真正的最高标(如 3 板)顶掉,
   该买的票反而买不到。置 skip_st=False 回到旧口径(不剔除)。
5. 次日(买入日)昨日的最高板龙头**收盘封住涨停**才打板买入(默认)。
   触板口径与撮合层算「封板」完全同源: 在**不复权原始价**空间上, 用当日最高价与按
   交易所规则算出的涨停价比较(前收整数分半进位; 除权日改用前复权前收作参考价)。
   置 require_seal=False 可放宽为「盘中触及涨停即买」, 不要求收盘封住。
   ⚠ 实测(2024-01..2026-10, 全市场, max_positions=10): 放宽触板把累计收益从 +90.8%
   打到 -41.7%, 胜率 46.0%→33.6%, 最大回撤 -7.6%→-43.9%。触板未封的板大多是「炸板」
   弱板, 按涨停价买进去等于买在情绪顶 —— 故默认仍要求收盘封板, 触板仅作可选开关。
6. 成交价口径: 当日最高价(META.backtest_defaults 的 entry_fill=high_t)。触板日的最高价
   就是涨停价, 封板日 high == close == 涨停价 —— 两种触发口径下都等于打板成交价。
   买入日当天是一字板的仍由撮合层按「涨停不可成交」拒绝, 买不进就不买。
7. 并列最高板(如昨日 4/4) 只打一只: 在当日成功触板的并列龙头中, 取**买入日当天**
   换手率最高的一只。
8. 退出规则(一) —— 第二天不封板必走(最重要): 持有期每个交易日收盘仍未封住涨停, 即当日
   收盘卖出; 封住则继续持有(连板接力)。T+1 由撮合层保证: 买入当日不可卖; 停牌日
   locked 为 False, 会挂起卖出意向, 复牌可成交时再成交, 避免长期停牌把仓位永久锁死。
9. 退出规则(二) —— 龙头杀到绿盘立刻走, 不等收盘: 持有期任一日跌破**前一交易日收盘价**
   (即「绿盘」)即离场, 且不在收盘价上等 —— 开盘就低开的按开盘价(竞价)走, 盘中才杀到
   绿盘的按止损线成交。理由是最高标龙头不能弱到连红盘都维持不住。实现方式见模块常量
   STOP_LOSS / TRAILING_STOP 的注释(由引擎风控线给出触发价成交, 不经过收盘价)。
   ⚠ 该规则同时会在「盘中炸板、尾盘又回封涨停」的日子提前离场, 等于把当天的涨停让给
   市场 —— 这是它的主要代价。实测(2024-01..2026-10, 全市场, max_positions=10, 规则(三)关闭):
   不启用本规则时 +142.2% / 胜率 47.2% / 回撤 -7.3% / Sharpe 2.75 / PF 2.43(231 笔, 全为 signal);
   启用后 +80.1% / 胜率 24.0% / 回撤 -3.4% / Sharpe 2.99 / PF 8.00(238 笔: stop_loss 178、
   trailing_stop 49、signal 11)。即用约一半的累计收益换掉一半的回撤, 净值曲线更平。
   止损线深度扫描(0.0/0.5%/1%/2%/3%/5%): 累计收益 80.1/82.0/77.4/70.3/54.7/52.8%,
   回撤 -3.4..-5.3% —— 线放得越松反而越差(跌得更深才走, 卖在更低处且拿得更久), 故默认 0.0。
10. 退出规则(三) —— 涨停但放量换手则以收盘价走: 当日收盘封板, 但换手率 >=
    weak_seal_turnover(默认 40%)时, 以当日收盘价离场(exit_fill=close_t 即当日收盘价)。
    原口径是「涨停 + 全天炸板多次 + 放量换手」, 但**数据里没有开板次数/炸板次数字段**:
    signal_broken_limit_up 是二值, 且仅在「盘中触板但收盘未封住」时为真 —— 收盘封住的
    当天即使盘中炸过多次也是 False。故只保留可观测的「封板 + 放量换手 + 换手率 >= 阈值」
    这一半。置 exit_weak_seal=False 可关闭。
    注: 该规则与规则(二)高度重叠 —— 放量换手的封板日绝大多数盘中都跌破过前收盘价, 会先被
    规则(二)在止损线上卖掉, 只有「从没跌破前收盘价又放量换手」的封板日才轮到本条。
    实测(同窗口, 规则(二)关闭): 只开本规则 +142.2%→+132.2%(成交 231→232 笔); 与规则(二)同开
    +80.1%→+81.9%(238→238 笔)。影响量级在噪声内。原因: 换手率 >= 40% 的封板日只占全部封板日
    的 0.73%(335/46245), 且其中约 3/4 盘中已跌破前收盘、被规则(二)先在止损线上收掉。
    注: 本规则的离场走 exit_mask, 在 exit_reason 里记为 signal, 无法与规则(一)在明细上区分。
11. 口径近似: 原规则为「14:30 未涨停则卖出」, 回测引擎无盘中定时出场能力, 规则(一)以
    「收盘未封板」近似 —— 尾盘炸板会多持有一天, 尾盘拉板会提前一天卖出。另: 触板入场
    假设板上排队能成交, 实际打板可能排不上队; 撮合层只拒绝「一字顶死」这一天买不进的形态。
    (require_seal=False 时)触板入场后若当日未封板, 卖出信号当日即成立, 由 T+1 顺延到次一交易日收盘成交。
"""


class TopBoardRelayMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        # raw_close / raw_high 是不复权原始价: 「触板」必须在交易所价格空间判定,
        # 前复权价经过复权因子缩放, 与前收的整数分关系不可直接比较。
        return frozenset(
            {
                "consecutive_limit_ups",
                "turnover_rate",
                "raw_close",
                "raw_high",
            }
        )

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 60

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        min_top_boards = max(1, int(params.get("min_top_boards", 2)))
        max_top_boards = max(min_top_boards, int(params.get("max_top_boards", 5)))
        skip_one_price = bool(params.get("skip_one_price_leader", True))
        skip_st = bool(params.get("skip_st", True))
        require_seal = bool(params.get("require_seal", True))
        exit_weak_seal = bool(params.get("exit_weak_seal", True))
        weak_seal_turnover = float(params.get("weak_seal_turnover", 40.0))
        shape = market.shape
        n_times = shape[0]

        streak = matrix_feature(market, "consecutive_limit_ups")
        turnover = matrix_feature(market, "turnover_rate")
        locked = market.limit_up_locked.astype(bool)

        # 可接力梯队: 有行情、板数不超上限、非风险警示股、且当日不是一字顶死(买不进去的板
        # 当最高标没意义)。被这些条件筛掉的票不占最高标位置, 龙头自然落到次高标。
        # 停牌/无数据行同样不参与。
        has_bar = np.isfinite(streak)
        eligible = has_bar & (streak <= max_top_boards)
        if skip_st:
            eligible &= ~_risk_warning_mask(market)
        if skip_one_price:
            eligible &= ~_one_price_limit_up(market, locked)

        market_top = np.where(eligible, streak, -np.inf).max(axis=1)
        leader = eligible & (streak == market_top[:, None])

        # 昨日的龙头与最高板用整行平移取, 不做逐股「跳过缺失 bar」的回溯 ——
        # 停牌股昨天不在涨停梯队里, 不应把更早的板数当成「昨日龙头」。
        leader_prev = np.zeros(shape, dtype=bool)
        leader_prev[1:] = leader[:-1]
        top_prev = np.full(n_times, -np.inf, dtype=np.float64)
        top_prev[1:] = market_top[:-1]

        # 昨日最高板龙头今日封板(默认)或触板(require_seal=False) => 进 min_top_boards+1 板, 打板买入。
        trigger = locked if require_seal else _limit_up_touched(market, locked)
        entry = (
            trigger
            & market.tradable.astype(bool)
            & leader_prev
            & (top_prev >= min_top_boards)[:, None]
        )
        entry = _narrow_to_single(entry, turnover)

        # ── 退出规则 ────────────────────────────────────────────────────────
        # (一) 第二天不封板必走。停牌日 locked 为 False, 会挂起卖出意向, 复牌可成交时再成交,
        # 避免长期停牌把仓位永久锁死。
        # (二)「龙头杀到绿盘立刻走」不在这里 —— 它必须在**触发价**上成交(低开按开盘价、
        # 盘中杀破按止损线)而不是收盘价, 由引擎的风控线实现, 见模块常量 STOP_LOSS /
        # TRAILING_STOP。风控线的判定优先级高于本处的 exit 信号(引擎在风控块里 continue),
        # 故同一天两条规则都成立时, 按更好的触发价成交。
        exit_mask = ~locked

        # (三) 封板但放量换手 => 当日收盘走(exit_fill=close_t 就是当日收盘价)。
        # 「全天炸板多次」在日线上不可观测(见 RULES 第 10 条), 只保留可观测的换手率条件;
        # 换手率缺失(NaN)时不触发 —— 少卖一次, 而不是拿缺数据当「没放量」错卖。
        if exit_weak_seal:
            exit_mask = exit_mask | (locked & _turnover_at_least(turnover, weak_seal_turnover))

        return make_signal_matrix(
            shape,
            entry=entry.astype(np.uint8),
            exit=exit_mask.astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(exit_mask, 0, -1).astype(np.int16),
            entry_signal_ids=("signal_top_board_relay_entry",),
            exit_signal_ids=("signal_top_board_relay_exit",),
        )


def _risk_warning_mask(market: MarketDataMatrix) -> np.ndarray:
    """风险警示股(ST/*ST)掩码, 形状 (1, 资产数) 以便与「时间 × 资产」矩阵按位与非。

    股票名由 instruments 维表 JOIN 进矩阵的 names (matrix 装载时强制带上 name 列);
    名称为空(维表缺失)时该股不会被剔除 —— 这是保守方向: 少剔一只, 而不是错剔一只。
    """
    flags = np.fromiter(
        (is_risk_warning_name(name) for name in market.names),
        dtype=bool,
        count=len(market.names),
    )
    return flags.reshape(1, -1)


def _limit_up_touched(market: MarketDataMatrix, locked: np.ndarray) -> np.ndarray:
    """当日盘中是否触及涨停价(触板) —— 返回「触板 或 收盘封板」。

    口径与撮合层算「封板」(_limit_lock_matrices)完全同源, 保证「触板」严格包含「封板」:
      1. 在**不复权原始价**空间比较(raw_high vs 涨停价), 前收同样取原始价;
      2. 涨停价按交易所规则对前收做整数分半进位(app.price_limits.numpy_limit_price);
      3. 除权日(复权因子与前一交易日不同)原始价不可比, 参考价改用前一交易日的复权收盘价;
      4. 停牌/无数据行不更新「前收」, 复牌时沿用停牌前最后一个有效收盘价。
    当日最高价触到涨停价即算触板 —— 触板时的最高价就是打板单的成交价(entry_fill=high_t)。
    """
    raw_close = market.fields.get("raw_close")
    raw_high = market.fields.get("raw_high")
    if raw_close is None or raw_high is None:
        # required_fields 已声明这两个字段; 真取不到时退化为「只有封板算触板」(旧口径),
        # 而不是在缺数据的情况下静默放大买入范围。
        return locked

    n_times, n_assets = market.shape
    adjusted = market.close
    legacy_pct, current_pct = numpy_limit_pct_vectors(market.symbols, market.names)
    session_dates = _session_dates(market)

    touched = np.zeros(market.shape, dtype=bool)
    previous_close = np.full(n_assets, np.nan, dtype=np.float64)
    previous_raw = np.full(n_assets, np.nan, dtype=np.float64)
    previous_factor = np.full(n_assets, np.nan, dtype=np.float64)

    for time_id in range(n_times):
        present = np.isfinite(adjusted[time_id])
        factor = np.full(n_assets, np.nan, dtype=np.float64)
        np.divide(
            adjusted[time_id],
            raw_close[time_id],
            out=factor,
            where=np.isfinite(raw_close[time_id]) & (raw_close[time_id] != 0),
        )
        factor_changed = (
            np.isfinite(factor)
            & np.isfinite(previous_factor)
            & (np.abs(factor - previous_factor) > 1e-6)
        )
        reference = np.where(factor_changed, previous_close, previous_raw)
        high = raw_high[time_id]
        valid = (
            present
            & np.isfinite(reference)
            & (reference > 0)
            & np.isfinite(high)
            & (high > 0)
        )
        if valid.any():
            limit_pct = (
                legacy_pct
                if session_dates[time_id] < MAIN_BOARD_ST_LIMIT_CHANGE_DATE
                else current_pct
            )
            up_price = numpy_limit_price(reference, limit_pct, up=True)
            touched[time_id, valid] = high[valid] >= up_price[valid] - 0.005
        previous_close[present] = adjusted[time_id][present]
        previous_raw[present] = raw_close[time_id][present]
        previous_factor[present] = factor[present]

    return touched | locked


def _session_dates(market: MarketDataMatrix) -> list[date]:
    """矩阵时间轴 → 交易日列表(timestamp_labels 是 'YYYY-MM-DD...' 标签)。"""
    dates: list[date] = []
    for label in market.timestamp_labels:
        try:
            dates.append(date.fromisoformat(str(label)[:10]))
        except ValueError:
            # 标签异常时按「新规日期」处理: 取当前涨跌幅, 不用旧 ST 5% 口径。
            dates.append(MAIN_BOARD_ST_LIMIT_CHANGE_DATE)
    return dates


def _one_price_limit_up(market: MarketDataMatrix, locked: np.ndarray) -> np.ndarray:
    """当日「一字顶死」矩阵: 开=高=低=收 且收盘封住涨停。

    口径与撮合层的拒绝买入判定(engine._one_price_limit)保持一致, 包括容差与可交易性:
    只有全天单一价格且封板的票才是「买不进去」, 盘中开过板再回封的 T 字板不算。
    """
    prices = np.stack(
        [
            np.asarray(market.open, dtype=np.float64),
            np.asarray(market.high, dtype=np.float64),
            np.asarray(market.low, dtype=np.float64),
            np.asarray(market.close, dtype=np.float64),
        ]
    )
    finite = np.isfinite(prices).all(axis=0)
    spread = np.where(finite, prices.max(axis=0) - prices.min(axis=0), np.inf)
    tolerance = np.maximum(np.abs(prices[3]) * 1e-4, 0.01)
    return locked & market.tradable.astype(bool) & finite & (spread <= tolerance)


def _turnover_at_least(turnover: np.ndarray, threshold: float) -> np.ndarray:
    """换手率是否达到阈值。

    换手率单位是**百分比**(矩阵由 volume(手) * 10000 / float_shares(股) 得出, 见
    matrix._write_turnover_rate_matrix), 故 40% 对应 40.0。缺失值为 NaN —— NaN 不参与
    比较, 一律算「未达到」, 避免 float_shares 缺失的票被误判为放量换手而错卖。
    """
    return np.isfinite(turnover) & (turnover >= float(threshold))


def _narrow_to_single(entry: np.ndarray, turnover: np.ndarray) -> np.ndarray:
    """并列最高板只打一只: 取当日候选中换手率最高者, 同值时取 symbol 靠前者。

    换手率全缺时 argmax 会落到非候选列上, 此时退回该日首个候选, 保证结果确定。
    """
    rows = np.flatnonzero(entry.any(axis=1))
    if rows.size == 0:
        return entry

    candidates = entry[rows]
    values = np.where(np.isfinite(turnover[rows]), turnover[rows], -np.inf)
    values = np.where(candidates, values, -np.inf)
    best = values.argmax(axis=1)
    no_value = ~np.isfinite(values[np.arange(rows.size), best])
    best = np.where(no_value, candidates.argmax(axis=1), best)

    narrowed = np.zeros_like(entry)
    narrowed[rows, best] = True
    return narrowed


MATRIX_STRATEGY = TopBoardRelayMatrixStrategy()
