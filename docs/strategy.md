# 策略指南

策略是选股引擎、回测、监控的基础。本文介绍策略体系与三种扩展方式。

完整策略开发规范(AI 生成与手写)见 [`backend/app/strategy/prompts/strategy-guide.md`](../backend/app/strategy/prompts/strategy-guide.md)。

---

## 内置策略

**26 个内置策略**,每个策略一个独立 Python 文件,基于 Polars 表达式向量化实现(`backend/app/strategy/builtin/`):

| 类型        | 代表策略                                                 |
| :---------- | :------------------------------------------------------- |
| 趋势 / 形态 | 趋势突破 · 均线多头 · MA 金叉 · MACD 金叉放量 · 布林突破 · 均线粘合突破 · 平台整理突破 · 放量创60日新高 |
| 量价 / 涨停 | 量价齐升 · 高换手强势 · 连板股 · 最高板打板接力 · 断板反包 · 涨停动量 · 接近涨停 · 涨停基因活跃股 |
| 反转 / 波动 | 超跌反弹 · 超卖反转 · 新低反转 · 低波动龙头 · 回踩 MA20 · 回踩支撑 · 强势开盘 · MACD 零下回升 · 长下影反击 · RSI 中轴回踩 |

内置目录 `backend/app/strategy/builtin/` 还包含一个仅供挖掘 worker 使用的受控因子排名研究模板。它不出现在普通选股列表，也不能通过普通策略 API 直接运行或保存 override；挖掘结果发布时会生成独立策略。详见 [因子与策略挖掘](./mining.md)。

内置目录 `backend/app/strategy/builtin/` 由项目维护,**AI 生成的策略不会落入此目录**。

---

## 扩展策略的三种方式

### 🎛️ 方式一:自定义信号(不写代码)

在选股页 UI 上用 `字段 + 操作符 + 阈值` 组合,编译成 Polars 表达式热加载。适合:

- 快速验证一个简单的筛选思路(如 `RSI < 30 AND 量比 > 2`)
- 不熟悉 Python 但想自定义筛选条件

底层实现在 `backend/app/strategy/custom_signals.py`。

### 🤖 方式二:AI 生成

一句话描述思路,LLM 读取精简运行时指南生成完整策略文件:

1. **配置 AI 接口**(留空即关闭,见 [configuration.md → AI](./configuration.md#ai可选)):
   ```ini
   AI_PROVIDER=openai_compat
   AI_BASE_URL=https://api.deepseek.com/v1
   AI_API_KEY=sk-...
   AI_MODEL=deepseek-chat
   ```
2. 在选股页打开「AI 策略生成器」,用自然语言描述你的策略思路
3. 前端流式接收生成代码,后端经 `ast` 安全校验(禁止 import os/sys/subprocess 等危险模块)后返回结果
4. 保存后落入 `data/strategies/ai/`,文件名/ID 用 `ai_` 前缀

生成策略相关提示词位于 `backend/app/strategy/prompts/`:

- `strategy-guide-compact.md` — AI 运行时精简指南(用于降低长请求超时概率)
- `strategy-guide.md` — 完整策略开发规范(供人工开发和详细参考)
- `strategy-builder-step2.md` — 步骤 2 提示词模板(修改已有策略)
- `strategy-example.md` — 从零创建强势反包策略的三步演示

> 💡 **文件与范围铁律**:AI 生成的策略只生成一个 `.py` 文件,只 `import polars as pl`,绝不修改 `backend/`、`docs/`、`frontend/` 等现有文件。

生成之后的持续改进(假设—验证—判定闭环),见 [AI 策略迭代指南](./strategy-iteration.md)。

### 📝 方式三:自定义编写 / 代码迁移

可以在选股页「自定义编写」中直接编辑策略代码并保存,新建自定义策略会落入 `data/strategies/custom/`,文件名/ID 用 `custom_` 前缀。也可以手动把已有策略改写为 Polars 文件后放入该目录,引擎会自动发现。

手写策略需遵循 [`strategy-guide.md`](../backend/app/strategy/prompts/strategy-guide.md) 的文件结构(META / basic_filter / scoring / ENTRY_SIGNALS / filter 等),完整规范见该文档。

---

## 策略文件结构(简述)

一个策略 `.py` 文件通常包含:

| 部分 | 作用 |
| :--- | :--- |
| `META` | 策略元信息(名称、参数、方向等),用户可在 UI 调整阈值 |
| `basic_filter(df, params)` | 模式 A:单日过滤,返回 `pl.Expr` |
| `filter_history(df, params)` | 模式 B:历史窗口过滤,返回 `pl.DataFrame`(配 `LOOKBACK_DAYS`) |
| `scoring` | 评分权重,总和 = 1.0 |
| `ENTRY_SIGNALS` / `EXIT_SIGNALS` | 进出场信号列(回测用) |

完整字段说明与示例见 [`strategy-guide.md`](../backend/app/strategy/prompts/strategy-guide.md)。

---

## 策略自带成交口径(`backtest_defaults`,可选)

回测页默认按**次日开盘**成交。少数策略的信号只有在**信号日当天**成交才成立,打板就是典型:
必须在封板当日按涨停价成交 —— 若推到次日开盘,次日低开低走也照样买入,就不再是打板了。

这类策略可以在 `META` 里声明自己的默认成交口径:

```python
META = {
    # ... 其余字段
    "backtest_defaults": {"entry_fill": "close_t", "exit_fill": "close_t"},
}
```

| 字段         | 取值                                              | 说明           |
| :----------- | :------------------------------------------------ | :------------- |
| `entry_fill` | `close_t` / `open_t+1`                            | 建仓口径       |
| `exit_fill`  | `close_t` / `open_t+1` / `signal_next_minute`     | 清仓口径       |

取值含义: `close_t` = 信号日收盘价成交, `open_t+1` = 次日开盘价成交, `signal_next_minute` = 分钟信号确认后下一分钟成交。

行为约定:

- 回测页**选中该策略时自动套用**声明的口径;用户仍可在页面上手动改回,手动值优先。
- 只覆盖声明过的字段,未声明的策略保持页面当前口径不变。
- 声明取值非法时整个声明被丢弃,不透传给前端(策略源文件里的 `META` 是自由 dict)。
- 该声明计入策略配置签名:声明变化的策略会让页面上已保存的表单状态失效并回到新默认值,
  否则老用户会把旧口径一直沿用下去。

内置策略 [`top_board_relay`](../backend/app/strategy/builtin/top_board_relay.py)(最高板打板接力)使用了该机制。

---

## 新增内置策略(贡献者)

如果你想为项目贡献一个内置策略:在 `backend/app/strategy/builtin/` 参照现有文件实现 `StrategyDef`,引擎会自动发现并加载。欢迎提交 PR。
