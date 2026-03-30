"""
scripts/run_pipeline.py — 全流程一键运行
数据下载 → 因子计算 → 模型训练 → 回测 → 报告
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, RESULT_DIR
from data.downloader import (download_index_constituents, download_all_stocks_info,
                              download_universe_daily, download_st_history,
                              download_benchmark, download_financial_indicators)
from data.universe  import build_universe, flag_limit_up_down
from factors.factors import build_default_registry
from model.trainer  import RollingLGBMTrainer, build_target, compute_ic_summary, compute_ic_series
from backtest.engine import VectorizedBacktest, compute_monthly_returns

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED = DATA_DIR / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
print("Final data will be saved to:", PROCESSED);

def step1_download():
    """Step 1: 下载原始数据"""
    logger.info("="*50)
    logger.info("STEP 1: 下载数据")
    logger.info("="*50)

    constituents = download_index_constituents()
    codes        = constituents["stock_code"].unique().tolist()
    logger.info(f"股票池: {len(codes)} 只")

    panel = download_universe_daily(codes)
    panel.to_parquet(DATA_DIR / "raw" / "panel_daily.parquet")

    st        = download_st_history()
    benchmark = download_benchmark()
    financial = download_financial_indicators(codes[:50])  # 先测试50只

    st.to_parquet(DATA_DIR / "raw" / "st_stocks.parquet")
    benchmark.to_parquet(DATA_DIR / "raw" / "benchmark_daily.parquet")
    financial.to_parquet(DATA_DIR / "raw" / "financial.parquet")

    logger.info("Step 1 完成")
    return panel, st, benchmark, financial


def step2_universe(panel, st):
    """Step 2: 构建股票池"""
    logger.info("="*50)
    logger.info("STEP 2: 构建股票池")
    logger.info("="*50)

    panel = build_universe(panel, st)
    panel = flag_limit_up_down(panel)
    panel.to_parquet(PROCESSED / "panel_with_universe.parquet")
    logger.info(f"Step 2 完成: {panel.shape}")
    return panel


def step3_factors(panel):
    """Step 3: 计算因子"""
    logger.info("="*50)
    logger.info("STEP 3: 计算因子")
    logger.info("="*50)

    if "in_universe" in panel.columns:
        universe_mask = panel["in_universe"].astype(bool)
    else:
        universe_mask = None
    registry      = build_default_registry(include_financial=False)
    factors_df    = registry.compute_all(panel, universe_mask)
    factors_df.to_parquet(PROCESSED / "factors.parquet")

    logger.info(f"Step 3 完成: {factors_df.shape[1]} 个因子")
    return factors_df


def step4_train(panel, factors_df):
    target  = build_target(panel, forward_periods=21)
    trainer = RollingLGBMTrainer()
    result  = trainer.fit_predict(factors_df, target)

    is_preds  = result["insample"]
    oos_preds = result["outofsample"]

    # IC 分开评估
    ic_is  = compute_ic_series(is_preds,  target)
    ic_oos = compute_ic_series(oos_preds, target)

    print("\n=== 样本内 IC ===")
    for k, v in compute_ic_summary(ic_is).items():
        print(f"  {k}: {v}")

    print("\n=== 样本外 IC（真正的评估）===")
    for k, v in compute_ic_summary(ic_oos).items():
        print(f"  {k}: {v}")

    # 保存两份预测
    is_preds.to_parquet(PROCESSED / "predictions_IS.parquet")
    oos_preds.to_parquet(PROCESSED / "predictions_OOS.parquet")
    result["full"].to_parquet(PROCESSED / "predictions.parquet")
    return result, trainer


def step5_backtest(preds, panel, benchmark):
    """Step 5: 回测"""
    logger.info("="*50)
    logger.info("STEP 5: 回测")
    logger.info("="*50)

    bench_ret = benchmark["close"].pct_change().dropna() if benchmark is not None else None
    bt        = VectorizedBacktest(preds, panel, benchmark_returns=bench_ret)
    results   = bt.run()

    logger.info("\n=== 回测结果 ===")
    for k, v in results["metrics"].items():
        logger.info(f"  {k}: {v}")

    results["portfolio_returns"].to_csv(RESULT_DIR / "portfolio_returns.csv")
    compute_monthly_returns(results["portfolio_returns"]).to_csv(
        RESULT_DIR / "monthly_returns.csv"
    )
    pd.DataFrame([results["metrics"]]).to_csv(RESULT_DIR / "metrics.csv")

    logger.info("Step 5 完成")
    return results


if __name__ == "__main__":
    # 检查是否已有缓存数据
    panel_path = DATA_DIR / "raw" / "panel_daily.parquet"

    if panel_path.exists():
        logger.info("发现缓存数据，跳过下载步骤")
        panel     = pd.read_parquet(panel_path)
        st        = pd.read_parquet(DATA_DIR / "raw" / "st_stocks.parquet") \
                    if (DATA_DIR / "raw" / "st_stocks.parquet").exists() \
                    else pd.DataFrame(columns=["stock_code"])
        benchmark = pd.read_parquet(DATA_DIR / "raw" / "benchmark_daily.parquet") \
                    if (DATA_DIR / "raw" / "benchmark_daily.parquet").exists() \
                    else None
        financial = pd.read_parquet(DATA_DIR / "raw" / "financial.parquet") \
                    if (DATA_DIR / "raw" / "financial.parquet").exists() \
                    else pd.DataFrame()
    else:
        panel, st, benchmark, financial = step1_download()

    panel      = step2_universe(panel, st)
    factors_df = step3_factors(panel)
    preds, trainer = step4_train(panel, factors_df)
    results    = step5_backtest(preds, panel, benchmark)

    logger.info("\n✅ 全流程完成！结果保存在 results/ 目录")
    logger.info("📊 打开 notebooks/ 中的Jupyter Notebook查看详细分析")
