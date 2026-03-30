"""
backtest/engine.py — 向量化回测引擎 + Zipline接口
职责：
  - 向量化回测（速度快，适合因子研究）
  - 完整绩效指标计算（夏普、最大回撤、Calmar等）
  - Zipline数据包构建指引
可单独运行：python -m backtest.engine
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (DATA_DIR, RESULT_DIR, PORTFOLIO_PARAMS,
                    BENCHMARK, TRAIN_END, TEST_START)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# 一、向量化回测引擎
# ════════════════════════════════════════════════════════════════════

class VectorizedBacktest:
    """
    向量化回测引擎
    
    设计原则：
    1. 信号生成和执行分离（信号在t日，执行在t+1日开盘）
    2. 考虑涨跌停无法成交的情况
    3. 双边交易成本（印花税+佣金）
    4. 支持等权和按分数加权
    
    使用示例:
        bt = VectorizedBacktest(predictions, panel)
        results = bt.run()
        bt.print_summary(results)
    """

    def __init__(self,
                 predictions: pd.Series,
                 panel: pd.DataFrame,
                 params: dict = None,
                 benchmark_returns: pd.Series = None):
        """
        Args:
            predictions: MultiIndex(date, stock_code) 预测得分
            panel: MultiIndex(date, stock_code) 日线面板（需含in_universe、at_limit）
            params: 回测参数（从config.PORTFOLIO_PARAMS）
            benchmark_returns: 基准日收益率 Series
        """
        self.predictions        = predictions
        self.panel              = panel
        self.params             = params or PORTFOLIO_PARAMS.copy()
        self.benchmark_returns  = benchmark_returns

    def run(self) -> dict:
        """执行回测，返回绩效字典"""
        logger.info("开始回测...")
        portfolio_returns = self._simulate_portfolio()
        metrics           = compute_performance_metrics(
            portfolio_returns,
            benchmark_returns=self.benchmark_returns,
            risk_free_rate=0.025 / 252  # 中国无风险利率约2.5%年化
        )
        logger.info("回测完成")
        return {
            "portfolio_returns": portfolio_returns,
            "metrics":           metrics,
        }

    def _simulate_portfolio(self) -> pd.Series:
        """
        核心回测逻辑
        Returns: 每日组合收益率 Series
        """
        params = self.params
        close  = self.panel["close"].unstack("stock_code")
        daily_ret = close.pct_change()

        # 涨跌停标记（当日信号不可执行）
        at_limit = None
        if "at_limit" in self.panel.columns:
            at_limit = self.panel["at_limit"].unstack("stock_code")

        # 获取调仓日
        from data.universe import get_rebalance_dates
        rebalance_dates = get_rebalance_dates(self.panel, params["rebalance_freq"])

        portfolio_returns = []
        current_holdings  = {}   # {stock_code: weight}
        prev_holdings     = {}

        trading_dates = daily_ret.index

        for i, dt in enumerate(trading_dates):
            # 当日组合收益（按昨日持仓）
            if current_holdings:
                day_ret = 0.0
                for code, wt in current_holdings.items():
                    if code in daily_ret.columns:
                        r = daily_ret.loc[dt, code]
                        if not np.isnan(r):
                            day_ret += wt * r
                portfolio_returns.append((dt, day_ret))
            else:
                portfolio_returns.append((dt, 0.0))

            # 调仓日：生成新持仓
            if dt in rebalance_dates:
                new_holdings = self._select_stocks(
                    dt, at_limit, params
                )
                if new_holdings:
                    # 计算换手率和交易成本
                    turnover = _compute_turnover(current_holdings, new_holdings)
                    cost     = turnover * params["transaction_cost"]
                    # 从当日收益中扣除交易成本
                    if portfolio_returns:
                        dt_idx, dt_ret = portfolio_returns[-1]
                        portfolio_returns[-1] = (dt_idx, dt_ret - cost)
                    prev_holdings    = current_holdings.copy()
                    current_holdings = new_holdings

        ret_series = pd.Series(
            dict(portfolio_returns), name="portfolio"
        ).sort_index()
        return ret_series

    def _select_stocks(self, signal_date, at_limit, params):
    # 获取预测分数
        try:
            scores = self.predictions.loc[signal_date]
        except KeyError:
            available = self.predictions.index.get_level_values("date")
            past = available[available <= signal_date]
            if len(past) == 0:
                return {}
            scores = self.predictions.loc[past[-1]]

        if isinstance(scores, pd.DataFrame):
            scores = scores.iloc[:, 0]

        # ── 修复：fillna(False) 再用布尔掩码 ──────────────────
        if at_limit is not None and signal_date in at_limit.index:
            day_limit = at_limit.loc[signal_date].fillna(False).astype(bool)
            limit_stocks = day_limit[day_limit].index
            scores = scores.drop(limit_stocks, errors="ignore")

        # 过滤 in_universe
        if "in_universe" in self.panel.columns:
            try:
                universe = (self.panel.loc[signal_date, "in_universe"]
                                .fillna(False).astype(bool))
                scores = scores.reindex(universe[universe].index).dropna()
            except KeyError:
                pass

        scores = scores.dropna()
        if len(scores) == 0:
            return {}

        n          = params["n_long"]
        top_stocks = scores.nlargest(min(n, len(scores))).index.tolist()

        if params["weight_scheme"] == "score":
            top_scores = scores[top_stocks]
            top_scores = top_scores - top_scores.min() + 1e-8
            return (top_scores / top_scores.sum()).to_dict()
        return {code: 1.0 / len(top_stocks) for code in top_stocks}


def _compute_turnover(old: dict, new: dict) -> float:
    """计算换手率（双边）"""
    all_codes = set(old) | set(new)
    turnover  = 0.0
    for code in all_codes:
        old_w = old.get(code, 0.0)
        new_w = new.get(code, 0.0)
        turnover += abs(new_w - old_w)
    return turnover / 2  # 单边换手率


# ════════════════════════════════════════════════════════════════════
# 二、绩效指标计算
# ════════════════════════════════════════════════════════════════════

def compute_performance_metrics(returns: pd.Series,
                                  benchmark_returns: pd.Series = None,
                                  risk_free_rate: float = 0.025 / 252,
                                  periods_per_year: int = 252) -> dict:
    """
    计算完整绩效指标
    
    Args:
        returns: 每日收益率 Series
        benchmark_returns: 基准每日收益率（用于计算超额收益）
        risk_free_rate: 日无风险利率
        periods_per_year: 年化因子（日频=252）
    
    Returns:
        dict 包含所有绩效指标
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return {}

    # ── 绝对绩效 ────────────────────────────────────────────────────
    cumret       = (1 + returns).cumprod()
    total_ret    = cumret.iloc[-1] - 1
    n_years      = len(returns) / periods_per_year
    ann_ret      = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol      = returns.std() * np.sqrt(periods_per_year)
    sharpe       = ((ann_ret - risk_free_rate * periods_per_year) /
                    ann_vol) if ann_vol > 1e-8 else 0
    max_dd       = _max_drawdown(cumret)
    calmar       = ann_ret / abs(max_dd) if max_dd < -1e-8 else 0
    sortino_vol  = returns[returns < 0].std() * np.sqrt(periods_per_year)
    sortino      = ((ann_ret - risk_free_rate * periods_per_year) /
                    sortino_vol) if sortino_vol > 1e-8 else 0
    win_rate     = (returns > 0).mean()
    skew         = returns.skew()
    kurt         = returns.kurtosis()

    metrics = {
        # 收益
        "累计收益率":   f"{total_ret:.2%}",
        "年化收益率":   f"{ann_ret:.2%}",
        # 风险
        "年化波动率":   f"{ann_vol:.2%}",
        "最大回撤":     f"{max_dd:.2%}",
        "最大回撤持续": _max_drawdown_duration(cumret),
        # 风险调整收益
        "夏普比率":     round(sharpe,  3),
        "索提诺比率":   round(sortino, 3),
        "卡玛比率":     round(calmar,  3),
        # 其他
        "日胜率":       f"{win_rate:.2%}",
        "偏度":         round(skew, 3),
        "峰度":         round(kurt, 3),
        "样本年数":     round(n_years, 1),
    }

    # ── 相对基准 ─────────────────────────────────────────────────────
    if benchmark_returns is not None:
        bench, port = benchmark_returns.align(returns, join="inner")
        excess_ret  = port - bench
        ann_excess  = excess_ret.mean() * periods_per_year
        te          = excess_ret.std() * np.sqrt(periods_per_year)
        info_ratio  = ann_excess / te if te > 1e-8 else 0
        beta        = _beta(port, bench)
        alpha_ann   = ann_ret - beta * (bench.mean() * periods_per_year)
        bench_cum   = (1 + bench).cumprod()
        bench_ret   = bench_cum.iloc[-1] - 1

        metrics.update({
            "基准累计收益": f"{bench_ret:.2%}",
            "超额收益(年化)": f"{ann_excess:.2%}",
            "跟踪误差":      f"{te:.2%}",
            "信息比率":      round(info_ratio, 3),
            "Beta":          round(beta,       3),
            "Alpha(年化)":   f"{alpha_ann:.2%}",
        })

    return metrics


def _max_drawdown(cumret: pd.Series) -> float:
    """最大回撤"""
    roll_max = cumret.cummax()
    drawdown = cumret / roll_max - 1
    return drawdown.min()


def _max_drawdown_duration(cumret: pd.Series) -> str:
    """最大回撤持续时间（天数）"""
    roll_max  = cumret.cummax()
    drawdown  = cumret / roll_max - 1
    in_dd     = drawdown < 0
    # 计算连续回撤天数
    groups    = (in_dd != in_dd.shift()).cumsum()
    dd_groups = in_dd[in_dd].groupby(groups[in_dd]).count()
    if dd_groups.empty:
        return "0天"
    return f"{dd_groups.max()}天"


def _beta(port: pd.Series, bench: pd.Series) -> float:
    """计算Beta"""
    cov = np.cov(port.values, bench.values)
    return cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-12 else 1.0


def compute_monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
    """
    生成月度收益率热力图数据
    Returns: DataFrame [年份 × 月份]
    """
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M")
    table = monthly.to_frame("ret")
    table["year"]  = monthly.index.year
    table["month"] = monthly.index.month
    pivot = table.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot["Annual"] = (1 + monthly).groupby(monthly.index.year).prod() - 1
    return pivot


# ════════════════════════════════════════════════════════════════════
# 三、Zipline 接入指引（注释说明，实际使用需额外配置）
# ════════════════════════════════════════════════════════════════════

ZIPLINE_SETUP_GUIDE = """
# ═══════ Zipline A股接入步骤 ═══════

# 1. 安装 zipline-reloaded
pip install zipline-reloaded

# 2. 创建自定义数据包（Custom Bundle）
# 在 ~/.zipline/extension.py 中注册：

from zipline.data.bundles import register
from backtest.zipline_bundle import a_share_bundle

register('a_share', a_share_bundle,
         calendar_name='XSHG')  # 上交所交易日历

# 3. 摄取数据
zipline ingest -b a_share

# 4. 运行策略
from zipline.api import order_percent, record, symbol, set_benchmark
from zipline import run_algorithm
import pandas as pd

def initialize(context):
    context.i = 0

def handle_data(context, data):
    # 从预测得分中读取当日信号
    # 实际接入时从预计算的predictions读取
    pass

results = run_algorithm(
    start=pd.Timestamp('2022-01-01', tz='utc'),
    end=pd.Timestamp('2024-01-01', tz='utc'),
    initialize=initialize,
    handle_data=handle_data,
    capital_base=1_000_000,
    bundle='a_share',
)
"""


def generate_zipline_bundle_template() -> str:
    """生成Zipline数据包模板代码"""
    return '''
"""
backtest/zipline_bundle.py — Zipline A股数据包
将下载好的AKShare数据转换为Zipline格式
"""

import pandas as pd
import numpy as np
from zipline.data.bundles import register
from zipline.utils.calendars import get_calendar

def a_share_bundle(environ, asset_db_writer, minute_bar_writer,
                   daily_bar_writer, adjustment_writer,
                   calendar, start_session, end_session,
                   cache, show_progress, output_dir):
    """
    Zipline bundle ingest function for A-share data
    """
    from config import DATA_DIR
    
    panel = pd.read_parquet(DATA_DIR / "raw" / "panel_daily.parquet")
    
    # 转换为Zipline期望的格式
    # 每只股票单独写入
    stocks = panel.index.get_level_values("stock_code").unique()
    
    def _data_generator():
        for i, code in enumerate(stocks):
            df = panel.xs(code, level="stock_code")[
                ["open", "high", "low", "close", "volume"]
            ].copy()
            df.index = df.index.tz_localize("Asia/Shanghai").tz_convert("UTC")
            yield i, df
    
    daily_bar_writer.write(_data_generator(), show_progress=show_progress)
    
    # 写入资产元数据
    asset_db_writer.write(
        equities=pd.DataFrame({
            "sid":    range(len(stocks)),
            "symbol": stocks,
            "exchange": ["XSHG" if s.startswith("6") else "XSHE" 
                         for s in stocks],
        })
    )
    
    # 写入分红/拆股调整（暂时跳过，使用后复权数据）
    adjustment_writer.write()
'''


# ─── 单独运行 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    pred_path  = DATA_DIR / "processed" / "predictions.parquet"
    panel_path = DATA_DIR / "processed" / "panel_with_universe.parquet"
    bench_path = DATA_DIR / "raw"       / "benchmark_daily.parquet"

    if not pred_path.exists():
        print("请先运行 model/trainer.py 生成预测")
        import sys; sys.exit(1)

    predictions = pd.read_parquet(pred_path)
    panel       = pd.read_parquet(panel_path)

    # 基准收益
    bench_ret = None
    if bench_path.exists():
        bench_df  = pd.read_parquet(bench_path)
        bench_ret = bench_df["close"].pct_change().dropna()

    # 运行回测
    bt      = VectorizedBacktest(predictions, panel,
                                  benchmark_returns=bench_ret)
    results = bt.run()

    # 打印绩效
    print("\n" + "="*50)
    print("         A股量化策略回测报告")
    print("="*50)
    for k, v in results["metrics"].items():
        print(f"  {k:<15}: {v}")
    print("="*50)

    # 月度收益表
    monthly_table = compute_monthly_returns(results["portfolio_returns"])
    print("\n月度收益率 (%):")
    print((monthly_table * 100).round(2).to_string())

    # 保存结果
    results["portfolio_returns"].to_csv(RESULT_DIR / "portfolio_returns.csv")
    monthly_table.to_csv(RESULT_DIR / "monthly_returns.csv")
    pd.DataFrame([results["metrics"]]).to_csv(RESULT_DIR / "metrics.csv")
    logger.info(f"结果已保存至 {RESULT_DIR}")

    # 打印Zipline使用说明
    print("\n" + "─"*50)
    print("如需使用Zipline回测，请参考以下配置：")
    print(ZIPLINE_SETUP_GUIDE)
