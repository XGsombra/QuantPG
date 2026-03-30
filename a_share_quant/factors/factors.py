"""
factors/factors.py — 所有因子实现
包含：
  - 通用因子（动量、规模、估值、质量、波动率）
  - A股特色因子（短期反转、换手率、资金流向、涨停板效应）

添加新因子：
  1. 新建一个继承 BaseFactor 的类
  2. 实现 compute(panel) 方法
  3. 在 build_default_registry() 中注册
"""

from pathlib import Path
import logging

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from factors.base import BaseFactor, FactorRegistry

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# 一、通用因子（美股也有，A股仍有效）
# ════════════════════════════════════════════════════════════════════

class Momentum12m1m(BaseFactor):
    """
    12-1月动量因子
    定义：过去12个月收益率，跳过最近1个月（避免短期反转噪音）
    A股有效性：中等，在牛市中有效，熊市中反转明显
    """
    name = "momentum_12_1"
    description = "12-1月价格动量"

    def __init__(self, lookback: int = 12, skip: int = 1):
        self.lookback = lookback
        self.skip     = skip

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        close   = panel["close"].unstack("stock_code")
        # 月度收益率
        monthly = close.resample("ME").last()
        # 累积收益：从 t-lookback 到 t-skip
        ret = (monthly.shift(self.skip) /
               monthly.shift(self.lookback) - 1)
        # 重新对齐到日频（前向填充，截面日用月末信号）
        ret_daily = ret.reindex(close.index, method="ffill")
        return ret_daily.stack(future_stack=True).rename(self.name).sort_index()


class Momentum6m1m(Momentum12m1m):
    """6-1月动量"""
    name = "momentum_6_1"
    description = "6-1月价格动量"

    def __init__(self):
        super().__init__(lookback=6, skip=1)


class LogMarketCap(BaseFactor):
    """
    对数市值因子
    定义：log(收盘价 × 估算流通股本)
    A股：小市值效应显著强于美股（散户偏好）
    """
    name = "log_mkt_cap"
    description = "对数市值（规模因子）"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if "turnover" not in panel.columns:
            # 退化为对数价格
            return self.safe_log(panel["close"]).rename(self.name).sort_index()
        # 估算市值：成交量 / (换手率/100) × 收盘价
        turnover_frac = (panel["turnover"] / 100).replace(0, np.nan)
        float_shares  = panel["volume"] / turnover_frac
        mkt_cap       = float_shares * panel["close"]
        return self.safe_log(mkt_cap).rename(self.name).sort_index()


class PriceToBook(BaseFactor):
    """
    市净率因子（价值因子）
    ⚠️ 使用财务数据时必须确保已经过 announce_date lag
    这里用 panel 中预处理好的 pb_ratio 列（已处理lag）
    """
    name = "pb_ratio"
    description = "市净率（越低越好，因此取负值）"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if "pb_ratio" not in panel.columns:
            raise ValueError("panel中缺少pb_ratio列，请先合并财务数据")
        # 低PB = 高价值，取负值使因子方向统一（值越大越好）
        return (-panel["pb_ratio"]).rename(self.name).sort_index()


class PriceToEarnings(BaseFactor):
    """市盈率因子"""
    name = "pe_ratio"
    description = "市盈率倒数（盈利收益率）"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if "pe_ratio" not in panel.columns:
            raise ValueError("panel中缺少pe_ratio列")
        # 用EP ratio（PE的倒数），过滤负PE
        ep = 1.0 / panel["pe_ratio"].replace(0, np.nan)
        ep[panel["pe_ratio"] < 0] = np.nan  # 亏损股不适用
        return ep.rename(self.name).sort_index()


class ROE(BaseFactor):
    """
    净资产收益率（质量因子）
    ⚠️ 财务数据已在 panel 中用 announce_date 对齐（下游处理）
    """
    name = "roe"
    description = "ROE（净资产收益率）"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if "roe" not in panel.columns:
            raise ValueError("panel中缺少roe列")
        return panel["roe"].rename(self.name).sort_index()


class ROA(BaseFactor):
    """总资产收益率"""
    name = "roa"
    description = "ROA（总资产净利率）"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if "roa" not in panel.columns:
            raise ValueError("panel中缺少roa列")
        return panel["roa"].rename(self.name).sort_index()


class Volatility20d(BaseFactor):
    """
    20日收益率波动率
    低波动异常：A股中低波动股票未必跑赢（与美股不同）
    """
    name = "volatility_20d"
    description = "20日收益率标准差（波动率）"

    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        close  = panel["close"].unstack("stock_code")
        ret    = close.pct_change()
        vol    = ret.rolling(self.window, min_periods=self.window // 2).std()
        vol_daily = vol.stack(future_stack=True)
        # 低波动 = 高因子值（取负值）
        return (-vol_daily).rename(self.name).sort_index()


# ════════════════════════════════════════════════════════════════════
# 二、A股特色因子（在美股已弱化，A股散户主导市场中仍有效）
# ════════════════════════════════════════════════════════════════════

class ShortTermReversal(BaseFactor):
    """
    ★ A股核心特色因子：短期反转
    定义：过去1个月收益率（取负值，反转）
    原理：A股散户比例高（约70%），追涨杀跌导致严重过度反应
          上月输家次月跑赢上月赢家（效果比美股强3-5倍）
    证据：在A股月度换仓策略中IC约0.04-0.06（显著）
    """
    name = "reversal_1m"
    description = "1月短期反转（A股最强单因子之一）"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        close   = panel["close"].unstack("stock_code")
        monthly = close.resample("ME").last()
        ret_1m  = monthly.pct_change(1)
        # 反转：上月大涨的股票下月卖出（取负值）
        reversal = -ret_1m
        reversal_daily = reversal.reindex(close.index, method="ffill")
        return reversal_daily.stack(future_stack=True).rename(self.name).sort_index()


class ShortTermReversal5d(BaseFactor):
    """
    5日超短期反转
    A股T+1限制下，游资打板后需要次日卖出，形成强烈的5日反转
    """
    name = "reversal_5d"
    description = "5日超短期反转"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        close  = panel["close"].unstack("stock_code")
        ret_5d = close.pct_change(5)
        # 5日涨幅越大，预期下周越弱（取负值）
        return (-ret_5d.stack(future_stack=True)).rename(self.name).sort_index()


class TurnoverMean(BaseFactor):
    """
    ★ A股特色因子：换手率均值
    定义：过去20日日均换手率
    原理：A股换手率是散户情绪的直接代理
          高换手 = 散户热情高涨 = 过度定价 = 预期收益低
          这在美股中不显著，但A股中效果明显
    方向：高换手 → 预期收益低（因子值取负）
    """
    name = "turnover_mean_20d"
    description = "20日平均换手率（A股情绪代理）"

    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if "turnover" not in panel.columns:
            raise ValueError("panel中缺少turnover列")
        turnover = panel["turnover"].unstack("stock_code")
        mean_to  = turnover.rolling(self.window, min_periods=5).mean()
        # 高换手率预期收益低，取负值
        result   = -mean_to.stack(future_stack=True)
        return result.rename(self.name).sort_index()


class TurnoverVolatility(BaseFactor):
    """
    换手率波动率
    定义：过去20日换手率的标准差
    原理：换手率波动大 = 市场对该股争议大 = 信息不对称高
    在A股中，高换手率波动与散户的不稳定持仓行为相关
    """
    name = "turnover_std_20d"
    description = "20日换手率波动率"

    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if "turnover" not in panel.columns:
            raise ValueError("panel中缺少turnover列")
        turnover = panel["turnover"].unstack("stock_code")
        std_to   = turnover.rolling(self.window, min_periods=5).std()
        return (-std_to.stack(future_stack=True)).rename(self.name).sort_index()


class PriceTo52WeekHigh(BaseFactor):
    """
    ★ A股特色因子：相对52周高点
    定义：当前价格 / 过去252日最高价
    原理：A股散户对历史高价有强锚定效应（"还有多远到最高点"）
          接近52周高点时，散户犹豫追高 → 实际上容易突破（正向）
          远离52周高点时，散户恐慌 → 往往超跌
    在美股中效果减弱，但在A股散户市场中仍显著
    """
    name = "price_to_52w_high"
    description = "价格相对52周高点（A股锚定效应）"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        close      = panel["close"].unstack("stock_code")
        high_252d  = close.rolling(252, min_periods=126).max()
        ratio      = close / high_252d
        return ratio.stack(future_stack=True).rename(self.name).sort_index()


class IdiosyncraticVolatility(BaseFactor):
    """
    特质波动率（IVOL）
    定义：用市场收益率回归后的残差波动率
    原理：美股中低IVOL溢价（Ang et al. 2006）在A股中更强
          A股机构做空受限，错误定价持续时间更长
          高特质波动率的股票往往被散户炒作定价过高
    """
    name = "idio_vol_20d"
    description = "20日特质波动率（相对市场）"

    def __init__(self, window: int = 60):
        self.window = window

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        close    = panel["close"].unstack("stock_code")
        ret      = close.pct_change()
        # 用等权市场收益率作为市场因子代理
        mkt_ret  = ret.mean(axis=1)

        ivol_dict = {}
        for col in ret.columns:
            stock_ret = ret[col].dropna()
            mkt_aligned = mkt_ret.reindex(stock_ret.index)
            # 滚动回归残差
            def _ivol(y):
                if len(y) < 20:
                    return np.nan
                y_arr = y
                x_arr = mkt_aligned.reindex(
                    ret.index[max(0, len(ret)-len(y)):len(ret)]
                ).values[-len(y):]
                if len(x_arr) != len(y_arr):
                    return np.nan
                cov = np.cov(y_arr, x_arr)
                if cov[1, 1] < 1e-12:
                    return np.nan
                beta   = cov[0, 1] / cov[1, 1]
                resid  = y_arr - beta * x_arr
                return resid.std()

            ivol = stock_ret.rolling(self.window, min_periods=30).std()
            ivol_dict[col] = ivol

        ivol_df = pd.DataFrame(ivol_dict)
        # 高特质波动 → 预期收益低（取负值）
        return (-ivol_df.stack(future_stack=True)).rename(self.name).sort_index()


class EarningsMomentum(BaseFactor):
    """
    ★ 盈利动量（A股PEAD效应）
    定义：最近一期净利润同比增长率
    原理：A股市场信息传导慢（散户信息处理能力弱），
          盈利公布后的漂移（PEAD）持续时间比美股更长（可达3-6个月）
    ⚠️ 必须使用 announce_date 对齐，报告期数据不可用
    """
    name = "earnings_momentum"
    description = "净利润同比增长率（盈利动量，PEAD）"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if "net_profit_growth" not in panel.columns:
            raise ValueError("panel中缺少net_profit_growth列（需合并财务数据）")
        return panel["net_profit_growth"].rename(self.name).sort_index()


class NetMainInflowRatio(BaseFactor):
    """
    ★ 主力资金净流入比（A股独有数据）
    定义：5日主力净流入 / 5日成交额
    原理：A股有主力资金（游资/机构）追踪系统，数据公开
          主力持续净流入 → 筹码集中 → 短期上涨概率高
          这个数据在美股不存在
    注意：效果在短周期（5-20日）更显著，月频因子效果弱化
    """
    name = "net_mf_ratio_5d"
    description = "5日主力资金净流入比（A股特有）"

    def __init__(self, window: int = 5):
        self.window = window

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if "main_net_inflow" not in panel.columns:
            logger.warning("panel中缺少main_net_inflow列，跳过资金流因子")
            return pd.Series(dtype=float, name=self.name)
        inflow_5d = (panel["main_net_inflow"]
                         .unstack("stock_code")
                         .rolling(self.window, min_periods=1).sum())
        amount_5d = (panel["amount"]
                         .unstack("stock_code")
                         .rolling(self.window, min_periods=1).sum())
        ratio = inflow_5d / amount_5d.replace(0, np.nan)
        return ratio.stack(future_stack=True).rename(self.name).sort_index()


class LimitUpConsecutive(BaseFactor):
    """
    ★ 连续涨停板后反转（A股高度特色）
    定义：过去10日内涨停板天数（取负值，预期反转）
    原理：A股涨停板机制导致流动性溢价。
          连续涨停后打开，往往是游资出货，随后大幅回落。
          这是A股特有的微观结构现象，美股没有对应机制。
    """
    name = "limit_up_reversal"
    description = "连续涨停后反转（A股特有）"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if "pct_change" not in panel.columns:
            return pd.Series(dtype=float, name=self.name)
        daily_pct = panel["pct_change"].unstack("stock_code")
        # 涨停标记（±9.9%）
        limit_up  = (daily_pct >= 9.9).astype(float)
        # 过去10日涨停次数
        limit_count = limit_up.rolling(10, min_periods=1).sum()
        # 涨停越多，预期未来越弱（反转）→ 取负值
        result = -limit_count.stack(future_stack=True)
        return result.rename(self.name).sort_index()


class RevenueGrowth(BaseFactor):
    """
    营收同比增长率（成长因子）
    """
    name = "revenue_growth"
    description = "营业收入同比增长率"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if "revenue_growth" not in panel.columns:
            raise ValueError("panel中缺少revenue_growth列")
        return panel["revenue_growth"].rename(self.name).sort_index()


# ════════════════════════════════════════════════════════════════════
# 工厂函数：构建默认因子注册表
# ════════════════════════════════════════════════════════════════════

def build_default_registry(include_financial: bool = True) -> FactorRegistry:
    """
    构建默认因子注册表
    
    Args:
        include_financial: 是否包含需要财务数据的因子（需要panel中有对应列）
    
    Returns:
        FactorRegistry 实例，已注册所有因子
    
    添加新因子：
        registry.register(MyNewFactor())
    """
    registry = FactorRegistry()

    # ── 价格/量 因子（只需日线数据）──────────────────────────────────
    registry.register_many([
        Momentum12m1m(),
        Momentum6m1m(),
        LogMarketCap(),
        Volatility20d(),
        ShortTermReversal(),        # ★ A股特色
        ShortTermReversal5d(),      # ★ A股特色
        TurnoverMean(),             # ★ A股特色
        TurnoverVolatility(),       # ★ A股特色
        PriceTo52WeekHigh(),        # ★ A股特色
        IdiosyncraticVolatility(),
        LimitUpConsecutive(),       # ★ A股特色
    ])

    # ── 财务因子（需要panel合并财务数据）────────────────────────────
    if include_financial:
        registry.register_many([
            PriceToBook(),
            PriceToEarnings(),
            ROE(),
            ROA(),
            EarningsMomentum(),     # ★ A股PEAD
            RevenueGrowth(),
        ])

    logger.info(f"因子注册完成，共 {len(registry.list_factors())} 个因子: "
                f"{registry.list_factors()}")
    return registry


# ─── 单独运行：测试因子计算 ──────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    from config import DATA_DIR

    panel_path = DATA_DIR / "processed" / "panel_with_universe.parquet"
    if not panel_path.exists():
        print("请先运行 data/universe.py")
        sys.exit(1)

    panel = pd.read_parquet(panel_path)
    universe_mask = panel.get("in_universe", pd.Series(True, index=panel.index))

    # 只用价格量因子测试
    registry = build_default_registry(include_financial=False)
    factors_df = registry.compute_all(panel, universe_mask)

    print(f"\n因子矩阵形状: {factors_df.shape}")
    print(f"因子列表: {factors_df.columns.tolist()}")
    print(f"\n因子统计:")
    print(factors_df.describe().round(3))
    print(f"\n因子缺失率:")
    print((factors_df.isna().mean() * 100).round(1).astype(str) + "%")

    out_path = DATA_DIR / "processed" / "factors.parquet"
    factors_df.to_parquet(out_path)
    print(f"\n因子已保存: {out_path}")
