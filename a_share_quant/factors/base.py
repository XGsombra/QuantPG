"""
factors/base.py — 因子基类
所有因子继承此类，统一接口，方便添加新因子
"""

from abc import ABC, abstractmethod
from pathlib import Path
import logging

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class BaseFactor(ABC):
    """
    因子基类。
    子类只需实现 compute() 方法，其余标准化、去极值、缺失处理由基类完成。
    
    使用示例:
        class MyFactor(BaseFactor):
            name = "my_factor"
            def compute(self, panel): ...
        
        factor = MyFactor()
        result = factor.run(panel)       # 返回标准化后的因子值 Series
    """

    name: str = "base_factor"    # 子类必须定义
    description: str = ""        # 因子说明

    # ── 标准化参数（子类可覆盖）──────────────────────────────────────
    winsorize_pct: float = 0.01  # 去极值：上下各1%截断
    normalize:     bool  = True  # 是否截面标准化（z-score）
    demean:        bool  = True  # 是否截面去均值

    @abstractmethod
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        计算原始因子值
        
        Args:
            panel: MultiIndex(date, stock_code) DataFrame，包含日线数据
        
        Returns:
            Series with MultiIndex(date, stock_code)，原始因子值
            ⚠️ 必须保证：因子值在日期t，只使用t日或t日之前的数据
        """
        raise NotImplementedError

    def run(self, panel: pd.DataFrame,
            universe_mask: pd.Series = None) -> pd.Series:
        """
        完整因子计算流程：compute → 过滤 → 去极值 → 截面标准化
        
        Args:
            panel: 日线面板 MultiIndex(date, stock_code)
            universe_mask: bool Series，True=在股票池内。若为None则不过滤
        
        Returns:
            标准化后的因子值 Series，MultiIndex(date, stock_code)
        """
        logger.info(f"计算因子: {self.name}")
        
        # 1. 计算原始因子
        raw = self.compute(panel)
        
        # 2. 只保留股票池内的股票
        if universe_mask is not None:
            aligned_mask = universe_mask.reindex(raw.index).fillna(False)
            raw = raw[aligned_mask]
        
        # 3. 截面去极值（Winsorize）
        raw = self._winsorize(raw)
        
        # 4. 截面标准化
        if self.normalize:
            raw = self._cross_section_zscore(raw)
        
        raw.name = self.name
        return raw

    # ── 工具方法 ─────────────────────────────────────────────────────

    @staticmethod
    def _winsorize(factor: pd.Series) -> pd.Series:
        """截面去极值：每个截面日单独做"""
        def _win_date(s):
            lo = s.quantile(0.01)
            hi = s.quantile(0.99)
            return s.clip(lo, hi)
        return factor.groupby(level="date").transform(_win_date)

    @staticmethod
    def _cross_section_zscore(factor: pd.Series) -> pd.Series:
        """截面z-score标准化"""
        def _zscore(s):
            mu  = s.mean()
            std = s.std()
            if std < 1e-8:
                return s - mu
            return (s - mu) / std
        return factor.groupby(level="date").transform(_zscore)

    @staticmethod
    def _cross_section_rank(factor: pd.Series) -> pd.Series:
        """截面排名（0-1分位数）"""
        return factor.groupby(level="date").transform(
            lambda s: s.rank(pct=True)
        )

    @staticmethod
    def safe_log(x: pd.Series) -> pd.Series:
        """安全log（处理负值和零）"""
        return np.log(x.clip(lower=1e-8))

    @staticmethod
    def lag(series: pd.Series, periods: int = 1) -> pd.Series:
        """
        在时间维度上lag（每只股票单独lag）
        ⚠️ 这是避免look-ahead bias的核心工具
        """
        return series.groupby(level="stock_code").shift(periods)

    @staticmethod
    def rolling_apply(series: pd.Series,
                       window: int,
                       func,
                       min_periods: int = None) -> pd.Series:
        """每只股票单独做滚动计算（按stock_code分组）"""
        if min_periods is None:
            min_periods = window // 2
        return (series
                .groupby(level="stock_code")
                .transform(lambda s: s.rolling(window, min_periods=min_periods)
                                      .apply(func, raw=True)))


class FactorRegistry:
    """
    因子注册表：统一管理所有因子，方便添加和批量计算
    
    使用示例:
        registry = FactorRegistry()
        registry.register(MomentumFactor())
        registry.register(ReversalFactor())
        
        all_factors = registry.compute_all(panel, universe_mask)
    """

    def __init__(self):
        self._factors: dict[str, BaseFactor] = {}

    def register(self, factor: BaseFactor) -> "FactorRegistry":
        """注册一个因子（支持链式调用）"""
        self._factors[factor.name] = factor
        logger.info(f"注册因子: {factor.name}")
        return self

    def register_many(self, factors: list) -> "FactorRegistry":
        for f in factors:
            self.register(f)
        return self

    def unregister(self, name: str) -> None:
        self._factors.pop(name, None)

    def list_factors(self) -> list:
        return list(self._factors.keys())

    def compute_all(self, panel: pd.DataFrame,
                    universe_mask: pd.Series = None) -> pd.DataFrame:
        """
        批量计算所有已注册因子
        返回: DataFrame，列 = 因子名，Index = MultiIndex(date, stock_code)
        """
        results = {}
        for name, factor in self._factors.items():
            try:
                results[name] = factor.run(panel, universe_mask)
            except Exception as e:
                logger.error(f"因子 {name} 计算失败: {e}")
                results[name] = pd.Series(dtype=float, name=name)

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results)

    def compute_one(self, name: str, panel: pd.DataFrame,
                    universe_mask: pd.Series = None) -> pd.Series:
        """计算单个因子"""
        if name not in self._factors:
            raise ValueError(f"因子 {name} 未注册，已注册: {self.list_factors()}")
        return self._factors[name].run(panel, universe_mask)
