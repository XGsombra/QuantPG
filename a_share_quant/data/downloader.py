"""
data/downloader.py — AKShare数据下载器
职责：从AKShare拉取原始数据，缓存到本地parquet
可单独运行：python -m data.downloader
"""

import time
import logging
from pathlib import Path

import akshare as ak
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, START_DATE, END_DATE, DOWNLOAD_SLEEP, CACHE_ENABLED

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ─── 通用缓存装饰器 ─────────────────────────────────────────────────

def cached(filename: str):
    """如果缓存存在且启用，直接读取parquet；否则运行函数并保存"""
    def decorator(fn):
        def wrapper(*args, force_refresh=False, **kwargs):
            path = RAW_DIR / filename
            if CACHE_ENABLED and path.exists() and not force_refresh:
                logger.info(f"从缓存读取: {path}")
                return pd.read_parquet(path)
            result = fn(*args, **kwargs)
            if result is not None and not result.empty:
                result.to_parquet(path)
                logger.info(f"已保存: {path}")
            return result
        return wrapper
    return decorator


# ─── 股票池（指数成分股历史） ────────────────────────────────────────

@cached("index_constituents.parquet")
def download_index_constituents(index_code: str = "000300") -> pd.DataFrame:
    """
    下载指数历史成分股
    返回: DataFrame [date, stock_code, weight]
    注意: AKShare的成分股API返回当前成分，历史需要按月拼接
    """
    logger.info(f"下载指数 {index_code} 成分股...")
    try:
        # 当前成分股（含历史权重变动）
        df = ak.index_stock_cons_weight_csindex(symbol=index_code)
        df = df.rename(columns={
            "成分券代码": "stock_code",
            "成分券名称": "stock_name",
            "日期": "date",
            "权重": "weight",
        })
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        logger.warning(f"成分股权重API失败，使用简单成分股列表: {e}")
        df = ak.index_stock_cons(symbol=index_code)
        df = df.rename(columns={"品种代码": "stock_code", "品种名称": "stock_name"})
        df["date"] = pd.Timestamp.today().normalize()
        df["weight"] = 1.0 / len(df)
        return df


@cached("all_stocks_info.parquet")
def download_all_stocks_info() -> pd.DataFrame:
    """
    下载A股所有股票基本信息（含退市股）
    用于Universe构建和ST过滤
    """
    logger.info("下载A股全量股票信息...")
    df = ak.stock_info_a_code_name()
    df.columns = ["stock_code", "stock_name"]
    time.sleep(DOWNLOAD_SLEEP)
    return df


# ─── 日线行情 ────────────────────────────────────────────────────────

def download_stock_daily(stock_code: str,
                          start: str = START_DATE,
                          end: str = END_DATE,
                          adjust: str = "hfq") -> pd.DataFrame:
    """
    下载单只股票后复权日线数据
    adjust: hfq=后复权(用于收益率计算), qfq=前复权, "": 不复权
    返回: DataFrame [date, open, high, low, close, volume, amount, turnover]
    """
    cache_path = RAW_DIR / f"daily_{stock_code}_{adjust}.parquet"
    if CACHE_ENABLED and cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust=adjust,
        )
        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
            "成交额": "amount", "换手率": "turnover",
            "涨跌幅": "pct_change", "振幅": "amplitude",
        })
        df["date"] = pd.to_datetime(df["date"])
        df["stock_code"] = stock_code
        df = df.set_index("date").sort_index()

        if CACHE_ENABLED:
            df.to_parquet(cache_path)

        time.sleep(DOWNLOAD_SLEEP)
        return df

    except Exception as e:
        logger.warning(f"下载 {stock_code} 失败: {e}")
        return pd.DataFrame()


def download_universe_daily(stock_codes: list,
                             start: str = START_DATE,
                             end: str = END_DATE) -> pd.DataFrame:
    """
    批量下载股票池日线数据，返回面板数据
    返回: MultiIndex DataFrame (date, stock_code)
    """
    all_dfs = []
    total = len(stock_codes)
    for i, code in enumerate(stock_codes):
        if (i + 1) % 50 == 0:
            logger.info(f"进度: {i+1}/{total}")
        df = download_stock_daily(code, start, end)
        if not df.empty:
            df["stock_code"] = code
            all_dfs.append(df.reset_index())

    if not all_dfs:
        return pd.DataFrame()

    panel = pd.concat(all_dfs, ignore_index=True)
    panel = panel.set_index(["date", "stock_code"]).sort_index()
    return panel


# ─── 财务数据 ────────────────────────────────────────────────────────
# 关键：财务数据必须用披露日期（announce_date），绝对不能用报告期！
# 这是避免look-ahead bias的核心

@cached("financial_indicators.parquet")
def download_financial_indicators(stock_codes: list) -> pd.DataFrame:
    """
    下载股票财务指标（ROE、ROA、PE、PB等）
    ⚠️ 返回数据包含 report_date 和 announce_date
       下游必须用 announce_date + lag 构建信号，严禁使用 report_date
    """
    all_dfs = []
    for i, code in enumerate(stock_codes):
        if (i + 1) % 20 == 0:
            logger.info(f"财务数据进度: {i+1}/{len(stock_codes)}")
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2014")
            if df.empty:
                continue
            df["stock_code"] = code
            # 规范列名
            df = df.rename(columns={
                "日期": "report_date",
                "净资产收益率": "roe",
                "总资产净利率": "roa",
                "市盈率": "pe_ratio",
                "市净率": "pb_ratio",
                "每股收益": "eps",
                "营业收入同比增长率": "revenue_growth",
                "净利润同比增长率": "net_profit_growth",
            })
            df["report_date"] = pd.to_datetime(df["report_date"])
            # 估算披露日期：年报4月30日前，三季报10月31日前，中报8月31日前，一季报4月30日前
            df["announce_date"] = df["report_date"].apply(_estimate_announce_date)
            all_dfs.append(df)
            time.sleep(DOWNLOAD_SLEEP)
        except Exception as e:
            logger.warning(f"财务数据 {code} 失败: {e}")

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def _estimate_announce_date(report_date: pd.Timestamp) -> pd.Timestamp:
    """
    根据报告期估算最晚披露日期（保守估计，避免look-ahead bias）
    Q1(3月31日) → 4月30日
    Q2(6月30日) → 8月31日
    Q3(9月30日) → 10月31日
    Q4(12月31日) → 次年4月30日
    """
    month = report_date.month
    year  = report_date.year
    if month == 3:
        return pd.Timestamp(year, 4, 30)
    elif month == 6:
        return pd.Timestamp(year, 8, 31)
    elif month == 9:
        return pd.Timestamp(year, 10, 31)
    else:  # month == 12
        return pd.Timestamp(year + 1, 4, 30)


# ─── ST状态数据 ──────────────────────────────────────────────────────

@cached("st_stocks.parquet")
def download_st_history() -> pd.DataFrame:
    """
    下载历史ST股票名单
    返回: DataFrame [stock_code, start_date, end_date]
    """
    logger.info("下载ST股票历史...")
    try:
        df = ak.stock_zh_a_st_em()
        df = df.rename(columns={
            "代码": "stock_code",
            "名称": "stock_name",
        })
        return df
    except Exception as e:
        logger.warning(f"ST数据下载失败: {e}")
        return pd.DataFrame(columns=["stock_code", "stock_name"])


# ─── 指数基准数据 ────────────────────────────────────────────────────

@cached("benchmark_daily.parquet")
def download_benchmark(index_code: str = "000300",
                        start: str = START_DATE,
                        end: str = END_DATE) -> pd.DataFrame:
    """下载基准指数日线数据（沪深300等）"""
    logger.info(f"下载基准 {index_code}...")
    df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
    df = df.rename(columns={"date": "date", "close": "close",
                             "open": "open", "high": "high",
                             "low": "low", "volume": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[start:end]
    return df


# ─── 资金流向数据（A股特色信号）────────────────────────────────────

def download_money_flow(stock_code: str) -> pd.DataFrame:
    """
    下载个股资金流向（主力、散户买卖方向）
    A股特有数据，可能对散户情绪有预测力
    """
    cache_path = RAW_DIR / f"moneyflow_{stock_code}.parquet"
    if CACHE_ENABLED and cache_path.exists():
        return pd.read_parquet(cache_path)
    try:
        df = ak.stock_individual_fund_flow(stock=stock_code, market="sh"
                                           if stock_code.startswith("6") else "sz")
        df = df.rename(columns={
            "日期": "date",
            "主力净流入-净额": "main_net_inflow",
            "小单净流入-净额": "small_net_inflow",
            "中单净流入-净额": "mid_net_inflow",
            "大单净流入-净额": "large_net_inflow",
            "超大单净流入-净额": "xlarge_net_inflow",
        })
        df["date"] = pd.to_datetime(df["date"])
        df["stock_code"] = stock_code
        df = df.set_index("date").sort_index()
        if CACHE_ENABLED:
            df.to_parquet(cache_path)
        time.sleep(DOWNLOAD_SLEEP)
        return df
    except Exception as e:
        logger.warning(f"资金流向 {stock_code} 失败: {e}")
        return pd.DataFrame()


# ─── 入口：单独运行下载全部数据 ─────────────────────────────────────

if __name__ == "__main__":
    from data.universe import build_universe
    logger.info("=== 开始下载全量数据 ===")

    # 1. 股票信息
    all_info = download_all_stocks_info()
    logger.info(f"A股总数: {len(all_info)}")

    # 2. 指数成分股
    constituents = download_index_constituents()
    logger.info(f"指数成分股: {len(constituents)}")

    # 3. 基准
    benchmark = download_benchmark()
    logger.info(f"基准数据: {len(benchmark)} 个交易日")

    # 4. 股票日线（用指数成分股）
    codes = constituents["stock_code"].unique().tolist()
    logger.info(f"开始下载 {len(codes)} 只股票日线...")
    panel = download_universe_daily(codes)
    panel.to_parquet(DATA_DIR / "raw" / "panel_daily.parquet")
    logger.info(f"日线面板: {panel.shape}")

    # 5. 财务数据
    financial = download_financial_indicators(codes)
    logger.info(f"财务数据: {len(financial)} 条记录")

    # 6. ST数据
    st = download_st_history()
    logger.info(f"ST股票: {len(st)}")

    logger.info("=== 数据下载完成 ===")
