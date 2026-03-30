"""
model/trainer.py — LightGBM 模型训练与预测
职责：
  - 滚动窗口训练（避免未来数据泄露）
  - 特征重要性分析
  - IC / ICIR 评估
  - 预测信号生成
可单独运行：python -m model.trainer
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (DATA_DIR, LGBM_PARAMS, TRAIN_WINDOW_MONTHS,
                    RETRAIN_FREQ_MONTHS, TRAIN_END, TEST_START)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ─── 目标变量构建 ─────────────────────────────────────────────────────

def build_target(panel: pd.DataFrame,
                 forward_periods: int = 21,
                 method: str = "excess") -> pd.Series:
    """
    构建预测目标：未来N日超额收益
    
    ⚠️ look-ahead bias 处理：
       在日期t构建因子时，目标变量是 t+1 到 t+forward_periods 的收益
       训练时用 t 的因子预测 t 的目标（目标已经是未来收益）
       实盘时 t 日因子 → 预测 t+1 起的未来收益 → t+1 日执行
    
    Args:
        panel: MultiIndex(date, stock_code) 面板
        forward_periods: 未来多少个交易日
        method: "excess" = 超额收益（相对截面均值）, "raw" = 原始收益
    
    Returns:
        Series: 未来forward_periods日超额收益，MultiIndex(date, stock_code)
    """
    close  = panel["close"].unstack("stock_code")
    # 未来N日原始收益（已经是前向的，使用时注意shift）
    fwd_ret = close.shift(-forward_periods) / close - 1

    if method == "excess":
        # 截面去均值：相对于同期所有股票的超额收益
        fwd_ret = fwd_ret.sub(fwd_ret.mean(axis=1), axis=0)

    target = fwd_ret.stack(future_stack=True).rename("target")
    return target


def align_features_target(features: pd.DataFrame,
                            target: pd.Series) -> tuple:
    """
    对齐特征和目标，去除缺失值
    Returns: (X, y) aligned DataFrames
    """
    # 统一 index 层级顺序为 (date, stock_code)
    def _sort_index(obj):
        if obj.index.names != ["date", "stock_code"]:
            level_map = {name: i for i, name in enumerate(obj.index.names)}
            if "date" in level_map and "stock_code" in level_map:
                obj = obj.reorder_levels(["date", "stock_code"])
        return obj.sort_index()

    features = _sort_index(features)
    target   = _sort_index(target.rename("target"))

    # 用 concat 替代 join，完全避免笛卡尔积
    combined = pd.concat([features, target], axis=1, join="inner").dropna()

    X = combined[features.columns]
    y = combined["target"]

    print(f"对齐后样本数: {len(X):,}（应在 1万~10万 之间）")
    return X, y


# ─── 滚动窗口训练器 ───────────────────────────────────────────────────

class RollingLGBMTrainer:
    """
    滚动窗口LightGBM训练器
    
    核心设计：
    - 每隔 retrain_freq 个月重新训练一次模型
    - 训练集：当前时间点的过去 train_window 个月
    - 预测集：下一个 retrain_freq 个月
    - 严格保证没有未来数据进入训练集
    
    使用示例:
        trainer = RollingLGBMTrainer()
        predictions = trainer.fit_predict(features, target)
        trainer.plot_feature_importance()
    """

    def __init__(self,
                 params: dict = None,
                 train_window_months: int = TRAIN_WINDOW_MONTHS,
                 retrain_freq_months: int = RETRAIN_FREQ_MONTHS):
        self.params              = params or LGBM_PARAMS.copy()
        self.train_window_months = train_window_months
        self.retrain_freq_months = retrain_freq_months
        self.models_             = []   # 保存每个窗口的模型
        self.feature_names_      = []
        self.all_importances_    = []

    def fit_predict(self, features, target,
                    train_end=TRAIN_END) -> pd.Series:
        """
        只做样本内滚动预测，用于模型开发阶段
        """
        X, y = align_features_target(features, target)
        self.feature_names_ = X.columns.tolist()

        dates     = X.index.get_level_values("date").unique().sort_values()
        train_end = pd.Timestamp(train_end)
        is_dates  = dates[dates <= train_end]

        print(f"样本内: {is_dates[0].date()} ~ {is_dates[-1].date()}")
        preds = self._rolling_predict(X, y, is_dates)
        preds.name = "pred_score"
        return preds  # 只返回 IS 预测，Series

    def _rolling_predict(self, X, y, dates) -> pd.Series:
        """样本内滚动预测（供 IS 诊断用）"""
        monthly = dates[dates.to_series().dt.is_month_end]
        retrain = monthly[::self.retrain_freq_months]
        all_preds = []

        for i, dt in enumerate(retrain):
            train_start = dt - pd.DateOffset(months=self.train_window_months)
            train_mask  = ((X.index.get_level_values("date") >= train_start) &
                        (X.index.get_level_values("date") <  dt))
            if train_mask.sum() < 500:
                continue

            pred_end  = retrain[i+1] if i+1 < len(retrain) else dates[-1]
            pred_mask = ((X.index.get_level_values("date") >= dt) &
                        (X.index.get_level_values("date") <  pred_end))
            if pred_mask.sum() == 0:
                continue

            model = self._train_single(X[train_mask], y[train_mask], dt)
            preds = pd.Series(
                model.predict(model._scaler.transform(X[pred_mask])),
                index=X[pred_mask].index,
                name="pred_score"
            )
            all_preds.append(preds)

        return pd.concat(all_preds).sort_index() if all_preds else pd.Series(dtype=float)

    def _train_single(self, X_train: pd.DataFrame,
                       y_train: pd.Series,
                       date: pd.Timestamp) -> lgb.LGBMRegressor:
        """训练单个窗口的LightGBM模型"""
        # 使用 RobustScaler 对特征标准化（对异常值鲁棒）
        scaler  = RobustScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = lgb.LGBMRegressor(**self.params)
        # 使用早停防止过拟合（用最后20%训练数据作为验证集）
        n_val = max(100, int(len(X_train) * 0.15))
        X_val, y_val = X_scaled[-n_val:], y_train.iloc[-n_val:]
        X_tr,  y_tr  = X_scaled[:-n_val], y_train.iloc[:-n_val]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )

        # 保存特征重要性
        imp = pd.Series(model.feature_importances_,
                        index=self.feature_names_,
                        name=str(date.date()))
        self.all_importances_.append(imp)

        # 保存scaler到model（用于后续预测）
        model._scaler = scaler
        return model

    def get_feature_importance(self) -> pd.DataFrame:
        """返回所有训练窗口的平均特征重要性"""
        if not self.all_importances_:
            return pd.DataFrame()
        imp_df = pd.DataFrame(self.all_importances_).T
        imp_df["mean"] = imp_df.mean(axis=1)
        imp_df["std"]  = imp_df.std(axis=1)
        return imp_df.sort_values("mean", ascending=False)
    
    def evaluate_oos(self, features, target,
                  train_end=TRAIN_END,
                  test_start=TEST_START) -> pd.Series:
        """
        ⚠️  只在你确认完成模型开发后调用一次
        调用前在代码里留注释记录日期，之后不再修改模型
        """
        confirm = input(
            "\n警告：你即将查看样本外数据。\n"
            "确认已完成所有因子选择和参数调整？(yes/no): "
        )
        if confirm.strip().lower() != "yes":
            print("已取消。请完成模型开发后再调用此函数。")
            return pd.Series(dtype=float)

        X, y    = align_features_target(features, target)
        X_train = X[X.index.get_level_values("date") <= pd.Timestamp(train_end)]
        y_train = y[y.index.get_level_values("date") <= pd.Timestamp(train_end)]
        X_oos   = X[X.index.get_level_values("date") >= pd.Timestamp(test_start)]

        model     = self._train_single(X_train, y_train, pd.Timestamp(train_end))
        oos_preds = pd.Series(
            model.predict(model._scaler.transform(X_oos)),
            index=X_oos.index, name="pred_score"
        )
        print(f"样本外预测完成: {test_start} ~ {X_oos.index.get_level_values('date').max().date()}")
        print("⚠️  请不要再修改模型后重新调用此函数。")
        return oos_preds


# ─── 因子评价：IC / ICIR ─────────────────────────────────────────────

def compute_ic_series(predictions: pd.Series,
                       target: pd.Series,
                       method: str = "rank") -> pd.Series:
    """
    计算每月IC（Information Coefficient）
    IC = 因子值与下期收益的横截面相关系数
    
    Args:
        method: "pearson" or "rank"（Rank IC更鲁棒，推荐）
    
    Returns:
        monthly IC Series
    """
    combined = pd.concat(
        [predictions.rename("score"), target.rename("target")],
        axis=1, join="inner"
    ).dropna()

    def _ic_date(df):
        if len(df) < 10:
            return np.nan
        if method == "rank":
            return df["score"].rank().corr(df["target"].rank())
        return df["score"].corr(df["target"])

    ic = combined.groupby(level="date").apply(_ic_date)
    return ic


def compute_ic_summary(ic: pd.Series) -> dict:
    """
    计算IC统计摘要
    Returns: dict with IC均值、标准差、ICIR、t统计量等
    """
    ic_clean = ic.dropna()
    n = len(ic_clean)
    if n == 0:
        return {}

    ic_mean = ic_clean.mean()
    ic_std  = ic_clean.std()
    icir    = ic_mean / ic_std if ic_std > 1e-8 else 0
    t_stat  = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 1e-8 else 0
    ic_pos  = (ic_clean > 0).mean()

    return {
        "IC均值":     round(ic_mean, 4),
        "IC标准差":   round(ic_std,  4),
        "ICIR":       round(icir,    3),
        "t统计量":    round(t_stat,  2),
        "IC正比例":   round(ic_pos,  3),
        "样本数":     n,
    }


def compute_quintile_returns(predictions: pd.Series,
                              forward_returns: pd.Series,
                              n_groups: int = 5) -> pd.DataFrame:
    combined = pd.concat(
        [predictions.rename("score"), forward_returns.rename("fwd_ret")],
        axis=1, join="inner"
    ).dropna()

    def _group_ret(df):
        # ── 修复1：股票数量必须足够分组 ──
        if len(df) < n_groups * 3:
            return pd.Series(dtype=float)
        try:
            groups = pd.qcut(df["score"], n_groups,
                             labels=[f"Q{i+1}" for i in range(n_groups)],
                             duplicates="drop")
        except ValueError:
            # ── 修复2：分位数边界重复时，改用 rank 分组 ──
            try:
                groups = pd.qcut(df["score"].rank(method="first"),
                                 n_groups,
                                 labels=[f"Q{i+1}" for i in range(n_groups)])
            except ValueError:
                return pd.Series(dtype=float)

        result = df.groupby(groups, observed=True)["fwd_ret"].mean()
        # ── 修复3：确保返回完整的5组，缺失的填 NaN ──
        full_idx = [f"Q{i+1}" for i in range(n_groups)]
        return result.reindex(full_idx)

    group_rets = combined.groupby(level="date").apply(_group_ret)
    return group_rets


# ─── 单独运行 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    factor_path = DATA_DIR / "processed" / "factors.parquet"
    panel_path  = DATA_DIR / "processed" / "panel_with_universe.parquet"

    if not factor_path.exists() or not panel_path.exists():
        print("请先运行 data/universe.py 和 factors/factors.py")
        import sys; sys.exit(1)

    features = pd.read_parquet(factor_path)
    panel    = pd.read_parquet(panel_path)
    logger.info(f"特征矩阵: {features.shape}")

    # 构建目标变量
    target = build_target(panel, forward_periods=21)

    # ── 样本内训练（开发阶段）──────────────────────────
    trainer  = RollingLGBMTrainer()
    is_preds = trainer.fit_predict(features, target)  # 现在直接返回 Series

    # IC评估（只用样本内）
    ic_series = compute_ic_series(is_preds, target)
    print("\n=== 样本内 IC 统计 ===")
    for k, v in compute_ic_summary(ic_series).items():
        print(f"  {k}: {v}")

    # 分组收益
    group_rets = compute_quintile_returns(is_preds, target)
    print("\n=== 分组平均月收益（样本内）===")
    print(group_rets.mean().round(4))

    # 保存
    is_preds.to_frame("pred_score").to_parquet(
        DATA_DIR / "processed" / "predictions.parquet")
    ic_series.to_csv(DATA_DIR / "processed" / "ic_series.csv")
    trainer.get_feature_importance().to_csv(
        DATA_DIR / "processed" / "feature_importance.csv")

    print("\n✅ 样本内训练完成，结果已保存")
    print("⚠️  OOS评估请在完成所有调参后调用 trainer.evaluate_oos()")
