"""
config.py — 全局配置中心
所有参数都在这里修改，各模块从这里导入
"""

from pathlib import Path

# ─── 路径 ──────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data" / "storage"
RESULT_DIR = ROOT_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 研究时间范围 ───────────────────────────────────────────────────
START_DATE = "2015-01-01"
END_DATE   = "2024-06-30"

# in-sample / out-of-sample 切割
TRAIN_END  = "2021-12-31"
TEST_START = "2022-01-01"

# ─── 股票池 ────────────────────────────────────────────────────────
# "000300" = 沪深300, "000905" = 中证500, "000906" = 中证800
UNIVERSE_INDEX = "000300"

# 过滤条件
MIN_LISTED_DAYS   = 252     # 上市不足1年的新股剔除
FILTER_ST         = True    # 剔除ST股票
FILTER_LIMIT      = True    # 剔除涨跌停日的交易信号
MIN_PRICE         = 1.0     # 剔除低价股（元）
MIN_MKT_CAP_PCTL  = 5       # 剔除市值最小5%分位（壳股）

# ─── 因子参数 ───────────────────────────────────────────────────────
FACTOR_PARAMS = {
    # 动量
    "momentum_12_1": {"lookback": 12, "skip": 1},
    "momentum_6_1":  {"lookback": 6,  "skip": 1},
    # 短期反转（A股特色，散户过度反应）
    "reversal_1m":   {"lookback": 1},
    "reversal_5d":   {"lookback": 5, "freq": "D"},
    # 换手率（A股特色因子）
    "turnover_mean_20d": {"window": 20},
    "turnover_std_20d":  {"window": 20},
    # 规模
    "log_mkt_cap": {},
    # 估值
    "pb_ratio":  {},
    "pe_ratio":  {},
    # 质量
    "roe":       {"lag_quarters": 1},  # 财务数据必须lag
    "roa":       {"lag_quarters": 1},
    # 波动率
    "volatility_20d":  {"window": 20},
    "idio_vol_20d":    {"window": 60},
    # 趋势（A股散户追涨）
    "price_to_52w_high": {},
    # 资金流向（A股特有）
    "net_mf_amount_5d": {"window": 5},
}

# ─── 模型参数 ───────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":        "regression",
    "metric":           "mse",
    "num_leaves":       31,
    "learning_rate":    0.05,
    "n_estimators":     300,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples": 20,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}

# 训练窗口（滚动训练）
TRAIN_WINDOW_MONTHS = 36   # 用过去36个月训练
RETRAIN_FREQ_MONTHS = 3    # 每3个月重新训练

# ─── 组合构建参数 ───────────────────────────────────────────────────
PORTFOLIO_PARAMS = {
    "n_long":           30,      # 多头持股数量
    "n_short":          30,      # 空头持股数量（模拟用，A股实际限制做空）
    "rebalance_freq":   "M",     # 调仓频率：M=月, W=周, Q=季
    "long_only":        True,    # True=纯多头, False=多空
    "weight_scheme":    "equal", # "equal"=等权, "score"=按得分加权
    "transaction_cost": 0.003,   # 双边交易成本（含印花税+佣金，约0.15%+0.1%*2）
}

# ─── 回测基准 ───────────────────────────────────────────────────────
BENCHMARK = "000300"  # 沪深300

# ─── 数据下载参数 ───────────────────────────────────────────────────
DOWNLOAD_SLEEP = 0.5   # AKShare请求间隔（秒），避免限流
CACHE_ENABLED  = True  # 是否缓存已下载数据
