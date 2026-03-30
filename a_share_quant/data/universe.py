"""
data/universe.py — 股票池构建与过滤
职责：在每个截面日生成合法的可交易股票池
关键：所有过滤条件必须在截面日当天已知，严禁look-ahead
可单独运行：python -m data.universe
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (DATA_DIR, START_DATE, END_DATE,
                    MIN_LISTED_DAYS, FILTER_ST, MIN_PRICE, MIN_MKT_CAP_PCTL)

logger = logging.getLogger(__name__)


def build_universe(panel: pd.DataFrame,
                   st_stocks: pd.DataFrame = None) -> pd.DataFrame:
    """
    在每个截面日，对股票池应用过滤规则，生成可交易标记
    
    Args:
        panel: MultiIndex(date, stock_code) 日线面板
        st_stocks: ST股票名单 DataFrame [stock_code]
    
    Returns:
        panel: 增加 'in_universe' bool列
    
    过滤规则（全部基于截面日已知信息，无look-ahead）：
    1. 上市不足1年 → 剔除
    2. ST / *ST → 剔除
    3. 股价 < 最低价格 → 剔除
    4. 市值最小5%分位（壳股）→ 剔除
    5. 当日停牌（成交量=0）→ 剔除
    6. 当日涨跌停 → 不剔除（但在信号执行时处理）
    """
    logger.info("构建股票池...")
    panel = panel.copy()
    panel["in_universe"] = True

    # ── 规则1：剔除上市不足1年 ──────────────────────────────────────
    # 用每只股票的最早出现日期作为上市日期的代理（有偏，但在免费数据下可接受）
    listing_dates = (panel.reset_index()
                         .groupby("stock_code")["date"]
                         .min()
                         .rename("listing_date"))
    panel = panel.join(listing_dates, on="stock_code")
    panel_idx = panel.reset_index()
    too_new = (panel_idx["date"] - panel_idx["listing_date"]).dt.days < MIN_LISTED_DAYS
    panel.loc[too_new.values, "in_universe"] = False
    panel = panel.drop(columns=["listing_date"])
    logger.info(f"  规则1(新股): 剔除 {too_new.sum()} 条")

    # ── 规则2：剔除ST股票 ────────────────────────────────────────────
    if FILTER_ST and st_stocks is not None and not st_stocks.empty:
        st_codes = set(st_stocks["stock_code"].tolist())
        st_mask = panel.reset_index()["stock_code"].isin(st_codes).values
        panel.loc[st_mask, "in_universe"] = False
        logger.info(f"  规则2(ST): 剔除 {st_mask.sum()} 条")
    else:
        # 用股票名称中的'ST'判断（仅当没有专门ST数据时）
        if "stock_name" in panel.columns:
            st_mask = panel["stock_name"].str.contains("ST", na=False)
            panel.loc[st_mask, "in_universe"] = False

    # ── 规则3：剔除低价股 ────────────────────────────────────────────
    if "close" in panel.columns:
        low_price = panel["close"] < MIN_PRICE
        panel.loc[low_price, "in_universe"] = False
        logger.info(f"  规则3(低价): 剔除 {low_price.sum()} 条")

    # ── 规则4：剔除市值最小5%（壳股）────────────────────────────────
    # 用收盘价 × 总股本 估算市值（免费数据限制）
    if "close" in panel.columns and "volume" in panel.columns:
        # 用成交量 / 换手率 估算流通股本，再算市值
        if "turnover" in panel.columns:
            panel["est_mkt_cap"] = _estimate_mkt_cap(panel)
            dates = panel.index.get_level_values("date").unique()
            for dt in dates:
                try:
                    day_data = panel.loc[dt, "est_mkt_cap"].dropna()
                    pctl = day_data.quantile(MIN_MKT_CAP_PCTL / 100)
                    small_cap = panel.loc[dt, "est_mkt_cap"] < pctl
                    panel.loc[(dt, slice(None)), "in_universe"] = (
                        panel.loc[(dt, slice(None)), "in_universe"] & ~small_cap
                    )
                except Exception:
                    pass

    # ── 规则5：剔除停牌日（成交量=0）────────────────────────────────
    if "volume" in panel.columns:
        suspended = panel["volume"] == 0
        panel.loc[suspended, "in_universe"] = False
        logger.info(f"  规则5(停牌): 剔除 {suspended.sum()} 条")

    in_universe_count = panel["in_universe"].sum()
    total = len(panel)
    logger.info(f"股票池构建完成: {in_universe_count}/{total} ({in_universe_count/total:.1%}) 在池内")
    return panel


def _estimate_mkt_cap(panel: pd.DataFrame) -> pd.Series:
    """
    用 成交量 / 换手率 估算流通股本，再 × 收盘价 估算市值
    换手率单位为 %，所以流通股本 = volume / (turnover/100)
    """
    turnover_frac = panel["turnover"] / 100
    float_shares  = panel["volume"] / turnover_frac.replace(0, np.nan)
    mkt_cap       = float_shares * panel["close"]
    return mkt_cap


def get_rebalance_dates(panel: pd.DataFrame, freq: str = "M") -> pd.DatetimeIndex:
    """
    生成调仓日期序列
    freq: 'M'=每月末, 'W'=每周末, 'Q'=每季末
    返回的日期是实际交易日
    """
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    
    if freq == "M":
        # 每月最后一个交易日
        date_series = pd.Series(all_dates, index=all_dates)
        rebalance = date_series.resample("ME").last().dropna()
    elif freq == "W":
        date_series = pd.Series(all_dates, index=all_dates)
        rebalance = date_series.resample("W").last().dropna()
    elif freq == "Q":
        date_series = pd.Series(all_dates, index=all_dates)
        rebalance = date_series.resample("QE").last().dropna()
    else:
        raise ValueError(f"不支持的调仓频率: {freq}")
    
    return pd.DatetimeIndex(rebalance.values)


def get_universe_on_date(panel: pd.DataFrame, date: pd.Timestamp) -> list:
    """获取指定日期的股票池（已过滤）"""
    try:
        day = panel.loc[date]
        if "in_universe" in day.columns:
            return day[day["in_universe"]].index.tolist()
        return day.index.tolist()
    except KeyError:
        return []


def flag_limit_up_down(panel: pd.DataFrame,
                        threshold: float = 0.099) -> pd.DataFrame:
    """
    标记涨跌停状态
    A股涨跌停为±10%（科创板±20%），不能在涨跌停日开仓
    ⚠️ 这里用当日涨跌幅，仅用于执行过滤，不影响信号生成
    """
    if "pct_change" not in panel.columns:
        return panel
    panel = panel.copy()
    panel["limit_up"]   = panel["pct_change"] >= threshold * 100
    panel["limit_down"] = panel["pct_change"] <= -threshold * 100
    panel["at_limit"]   = panel["limit_up"] | panel["limit_down"]
    return panel


# ─── 单独运行测试 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    
    panel_path = DATA_DIR / "raw" / "panel_daily.parquet"
    if not panel_path.exists():
        print("请先运行 data/downloader.py 下载数据")
    else:
        panel = pd.read_parquet(panel_path)
        print(f"原始面板: {panel.shape}")
        
        from data.downloader import download_st_history
        st = download_st_history()
        
        panel = build_universe(panel, st)
        panel = flag_limit_up_down(panel)
        
        # 查看每个截面日的股票数量
        daily_count = (panel.groupby(level="date")["in_universe"]
                            .sum()
                            .rename("universe_size"))
        print("\n每月末股票池大小:")
        print(daily_count.resample("ME").last().tail(12))
        
        panel.to_parquet(DATA_DIR / "processed" / "panel_with_universe.parquet")
        print("\n已保存处理后的面板")
