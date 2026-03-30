# A股量化研究框架

基于 AKShare + LightGBM 的 A股因子研究与回测系统，使用完全免费数据。

## 项目结构

```
a_share_quant/
├── config.py              # ← 所有参数在这里改
├── requirements.txt
├── data/
│   ├── downloader.py      # AKShare数据下载（可单独运行）
│   └── universe.py        # 股票池构建与过滤（可单独运行）
├── factors/
│   ├── base.py            # 因子基类 + 注册表
│   └── factors.py         # 所有因子实现（可单独运行）
├── model/
│   └── trainer.py         # LightGBM滚动训练 + IC评估（可单独运行）
├── backtest/
│   └── engine.py          # 向量化回测 + 绩效指标（可单独运行）
├── scripts/
│   └── run_pipeline.py    # 一键运行全流程
└── notebooks/
    └── research.ipynb     # 完整可视化分析
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一键运行（约需1-2小时，取决于股票池大小）
python scripts/run_pipeline.py

# 3. 查看结果
jupyter lab notebooks/research.ipynb
```

## 各模块单独运行

```bash
python -m data.downloader      # 只下载数据
python -m data.universe        # 只构建股票池
python -m factors.factors      # 只计算因子
python -m model.trainer        # 只训练模型
python -m backtest.engine      # 只运行回测
```

## 添加新因子

在 `factors/factors.py` 中：

```python
class MyNewFactor(BaseFactor):
    name = "my_factor"
    description = "我的新因子"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        # 只使用 t 日或之前的数据
        # 返回 MultiIndex(date, stock_code) Series
        return panel["close"].rename(self.name)

# 在 build_default_registry() 中注册：
registry.register(MyNewFactor())
```

## 调整模型参数

在 `config.py` 中修改 `LGBM_PARAMS`，无需改动其他代码。

## 因子说明

### A股特色因子（重点）

| 因子 | 原理 | 在美股是否有效 |
|------|------|--------------|
| `reversal_1m` | 散户过度反应导致月度反转 | 已大幅弱化 |
| `reversal_5d` | 游资打板后次日卖出 | 不适用 |
| `turnover_mean_20d` | 高换手=散户情绪过热=预期弱 | 效果弱 |
| `price_to_52w_high` | 散户锚定历史高点 | 弱化 |
| `limit_up_reversal` | 涨停板后反转 | 无此机制 |
| `earnings_momentum` | PEAD在A股持续更久 | 已部分套利 |

## 注意事项

1. **look-ahead bias**：财务数据使用`announce_date`对齐，不使用`report_date`
2. **幸存者偏差**：yfinance/AKShare均有，已在代码中注明
3. **ST股票**：已过滤，避免退市风险
4. **涨跌停**：信号生成后检查执行可行性
5. **交易成本**：默认0.3%双边（含印花税0.1%+佣金0.1%×2）

## Zipline 接入

详见 `backtest/engine.py` 中的 `ZIPLINE_SETUP_GUIDE` 和
`generate_zipline_bundle_template()` 函数。

由于 Zipline 与 A股交易日历需要额外配置，建议先用内置向量化回测，
验证策略有效后再接入 Zipline 做更精细的模拟。
