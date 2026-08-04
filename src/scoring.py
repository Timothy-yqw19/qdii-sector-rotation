"""
打分层：信号 → 维度分 → 板块合成分 → 轮动分差 → 仓位。

为什么是“维度内均值 + 维度间等权”而不是 20%/15%/10%：
  多个动量类信号本质在测同一个东西，若拉平等权会让“动量”这一维度被重复计票。
  先在维度内取均值消除重复，再在维度间等权，等于给每个**独立信息源**同样的话语权。
  精确到 5% 的权重是虚假精度——真实数据支撑不了那种分辨率。
  权重的稳健性交给 backtest.weight_sensitivity() 用随机权重抽样来证明。
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402


# ==========================================================================
# 维度分与合成分
# ==========================================================================

def dimension_scores(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    {sector: DataFrame(index=date, columns=DIMENSIONS)}
    维度分 = 该维度内所有已对齐信号的均值（跳过缺失，至少要有1个非缺失才出分）。
    """
    out = {}
    for sector, df in panel.items():
        dims = pd.DataFrame(index=df.index)
        for dim in C.DIMENSIONS:
            keys = [s.key for s in C.signals_in(dim) if s.key in df.columns]
            dims[dim] = df[keys].mean(axis=1, skipna=True) if keys else np.nan
        out[sector] = dims
    return out


def dimension_spread(dim_scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    各维度自身的多空分差（多头腿 - 空头腿）。
    这才是权重该作用的对象：我们要决策的是相对配置，不是绝对分。
    """
    return dim_scores[C.LONG_LEG] - dim_scores[C.SHORT_LEG]


def composite_scores(
    dim_scores: dict[str, pd.DataFrame],
    weights: dict[str, float] | pd.DataFrame | None = None,
    smooth_days: int | None = None,
) -> pd.DataFrame:
    """
    维度加权合成 → 板块合成分（z 单位）。

    weights 可以是：
      - dict            静态权重
      - DataFrame       时变权重(index=date, columns=dims)，用于逆波动/IC加权
      - None            用 config 里的默认
    """
    weights = C.SCORING.dimension_weights if weights is None else weights
    smooth_days = C.SCORING.score_smooth_days if smooth_days is None else smooth_days

    out = {}
    for sector, dims in dim_scores.items():
        if isinstance(weights, pd.DataFrame):
            w = weights.reindex(dims.index).reindex(columns=dims.columns).ffill()
        else:
            w = pd.DataFrame(
                [[weights.get(d, 0.0) for d in dims.columns]] * len(dims),
                index=dims.index, columns=dims.columns,
            )
        # 用可用维度重新归一化权重，避免早期维度缺失时分数被系统性压低
        wsum = dims.notna().mul(w).sum(axis=1).replace(0, np.nan)
        score = dims.mul(w).sum(axis=1, skipna=True) / wsum
        if smooth_days and smooth_days > 1:
            score = score.rolling(smooth_days, min_periods=1).mean()
        out[sector] = score
    return pd.DataFrame(out)


# ==========================================================================
# 权重方案
# ==========================================================================

def equal_weights(index: pd.Index) -> pd.DataFrame:
    n = len(C.DIMENSIONS)
    return pd.DataFrame(1.0 / n, index=index, columns=C.DIMENSIONS)


def inverse_vol_weights(
    dim_spread: pd.DataFrame, window: int | None = None
) -> pd.DataFrame:
    """
    逆波动加权（风险平价思路）。

    动机：等权是**名义**等权，不是风险等权。动量维度的分差波动天然比估值维度大，
    等权的结果是动量实际主导了整个信号。按 1/σ 加权后，各维度对最终分差的
    风险贡献才大致相等。

    注意 σ 用滚动窗口估计，只用截止当日信息。
    """
    window = window or C.SCORING.vol_weight_window
    vol = dim_spread.rolling(window, min_periods=window // 3).std()
    inv = 1.0 / vol.replace(0, np.nan)
    return inv.div(inv.sum(axis=1), axis=0)


def ic_weights(
    dim_spread: pd.DataFrame,
    rel_ret: pd.Series,
    window: int | None = None,
    horizon: int | None = None,
    shrinkage: float | None = None,
    floor: float | None = None,
    exec_lag: int | None = None,
) -> pd.DataFrame:
    """
    滚动 IC 加权：按各维度近期预测力分配权重。

    【防前视是这里最容易翻车的地方，说清楚】
    t 日的权重只能用**已经完全实现**的前瞻收益来估。
    s 日信号对应的前瞻收益覆盖 [s+lag, s+lag+h-1]，要到 s+lag+h-1 才知道结果。
    所以把滚动 IC 序列整体后移 (lag+h-1) 天：t 日用到的最新一对样本，
    其收益恰好在 t 日之前就已实现。少移一天都是偷看未来。

    【防过拟合的两道保险】
      1. 负 IC 截断为 0 —— 不因为一段噪音就去做反向押注；
      2. 向等权收缩 shrinkage（默认 0.5）—— 业界常规做法，
         纯 IC 加权在样本外经常还不如等权。
    """
    window = window or C.SCORING.ic_weight_window
    horizon = horizon or C.SCORING.ic_weight_horizon
    shrinkage = C.SCORING.ic_weight_shrinkage if shrinkage is None else shrinkage
    floor = C.SCORING.ic_weight_floor if floor is None else floor
    exec_lag = C.BACKTEST.execution_lag if exec_lag is None else exec_lag

    # 与回测端口径一致的前瞻收益
    fwd = rel_ret.rolling(horizon).sum().shift(-(horizon + exec_lag - 1))
    fwd = fwd.reindex(dim_spread.index)

    lag_to_realize = horizon + exec_lag - 1
    raw = {}
    for d in dim_spread.columns:
        ic = dim_spread[d].rolling(window, min_periods=window // 3).corr(fwd)
        raw[d] = ic.shift(lag_to_realize)      # ← 防前视的关键一行
    ic_df = pd.DataFrame(raw)

    pos = ic_df.clip(lower=0.0) + floor
    w_ic = pos.div(pos.sum(axis=1), axis=0)
    w_eq = equal_weights(dim_spread.index)
    w = shrinkage * w_ic + (1 - shrinkage) * w_eq
    # IC 估计本身很吵，权重再平滑一个月，避免权重自己制造换手
    w = w.rolling(21, min_periods=1).mean()
    return w.fillna(w_eq).div(w.sum(axis=1), axis=0).fillna(1.0 / len(C.DIMENSIONS))


def resolve_weights(
    dim_scores: dict[str, pd.DataFrame],
    rel_ret: pd.Series | None = None,
    scheme: str | None = None,
    active_dims: list[str] | None = None,
) -> pd.DataFrame:
    """
    按方案名产出时变权重表。ic_weighted 需要传 rel_ret。

    active_dims：只保留指定维度（其余权重置0后重新归一化）。
    用于"砍掉被证伪的维度"这类实验——逐维度IC拆解发现某维度是纯噪音时，
    应该做的是把它删掉重跑，而不是继续让它稀释有效信号。
    """
    scheme = scheme or C.SCORING.weight_scheme
    spread = dimension_spread(dim_scores)

    if scheme == "equal":
        w = equal_weights(spread.index)
    elif scheme == "inverse_vol":
        w = inverse_vol_weights(spread)
    elif scheme == "ic_weighted":
        if rel_ret is None:
            raise ValueError("ic_weighted 需要传入 rel_ret（板块相对日收益）")
        w = ic_weights(spread, rel_ret)
    else:
        raise ValueError(
            f"未知权重方案: {scheme}（可选 equal / inverse_vol / ic_weighted）")

    if active_dims:
        unknown = set(active_dims) - set(C.DIMENSIONS)
        if unknown:
            raise ValueError(f"未知维度 {unknown}，可选: {C.DIMENSIONS}")
        mask = pd.Series({d: (1.0 if d in active_dims else 0.0) for d in w.columns})
        w = w.mul(mask, axis=1)
        w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    return w


def rotation_spread(comp: pd.DataFrame) -> pd.Series:
    """轮动分差 = 多头腿合成分 - 空头腿合成分。>0 看多 TECH 相对 CONSUMER。"""
    return comp[C.LONG_LEG] - comp[C.SHORT_LEG]


# ==========================================================================
# 展示用 0-100 分
# ==========================================================================

def to_display_score(z: pd.Series | float) -> pd.Series | float:
    """
    把 z 分数映射到 0-100 便于沟通：Φ(z)×100，即“历史分布中的百分位”。
    注意这只是**展示层**变换，回测全程用 z，避免非线性变换影响统计量。
    """
    if isinstance(z, pd.Series):
        return z.apply(lambda x: np.nan if pd.isna(x) else 100 * _norm_cdf(x))
    return np.nan if pd.isna(z) else 100 * _norm_cdf(z)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ==========================================================================
# 仓位映射（含迟滞带）
# ==========================================================================

def raw_position(spread: pd.Series, scale: float | None = None) -> pd.Series:
    """分差 → [-1, 1] 目标仓位。tanh 天然限幅，避免极端分数导致杠杆失控。"""
    scale = C.BACKTEST.position_scale if scale is None else scale
    return np.tanh(spread * scale)


def apply_hysteresis(target: pd.Series, band: float | None = None) -> pd.Series:
    """
    迟滞带：目标仓位与现有仓位差异小于 band 时不动手。

    这是日频模型的关键工程细节——底层慢信号一周才更新一次，
    快信号的日内噪音会造成大量无意义换手。迟滞带把换手压到有信息含量的变动上。
    """
    band = C.BACKTEST.hysteresis_band if band is None else band
    held, out = 0.0, []
    for v in target.to_numpy():
        if np.isnan(v):
            out.append(np.nan)
            continue
        if abs(v - held) > band:
            held = float(v)
        out.append(held)
    return pd.Series(out, index=target.index)


def apply_rebalance_freq(pos: pd.Series, freq: str | None) -> pd.Series:
    """
    限制调仓频率。freq=None/'D' 为日频；'W' 周频；'M' 月频。

    为什么需要这个：如果某维度的 IC 随期限单调上升（5日 < 20日 < 60日），
    那它本质是个慢信号，日频调仓只会把慢信号的价值消耗在换手成本上。
    月频调仓不是"偷懒"，而是让调仓频率匹配信息更新频率。
    """
    if not freq or freq.upper() == "D":
        return pos
    marks = pos.groupby(pos.index.to_period(freq.upper())).transform(
        lambda s: s.index == s.index.max()
    )
    held = pos.where(marks).ffill()
    return held.fillna(0.0)


def build_positions(
    spread: pd.Series,
    band: float | None = None,
    scale: float | None = None,
    rebalance: str | None = None,
) -> pd.Series:
    pos = apply_hysteresis(raw_position(spread, scale), band)
    return apply_rebalance_freq(pos, rebalance or C.BACKTEST.rebalance_freq)


# ==========================================================================
# 一站式
# ==========================================================================

def run_scoring(
    panel: dict[str, pd.DataFrame],
    weights: dict[str, float] | pd.DataFrame | None = None,
    scheme: str | None = None,
    rel_ret: pd.Series | None = None,
    active_dims: list[str] | None = None,
    rebalance: str | None = None,
) -> dict:
    """
    weights 显式传入时优先；否则按 scheme 解析（默认取 config.SCORING.weight_scheme）。
    """
    dims = dimension_scores(panel)
    active_dims = active_dims or C.ACTIVE_DIMENSIONS
    if weights is None:
        weights = resolve_weights(dims, rel_ret=rel_ret, scheme=scheme,
                                  active_dims=active_dims)
    comp = composite_scores(dims, weights)
    spread = rotation_spread(comp)
    return {
        "dimension_scores": dims,
        "dimension_spread": dimension_spread(dims),
        "weights": weights,
        "scheme": scheme or C.SCORING.weight_scheme,
        "composite": comp,
        "spread": spread,
        "active_dims": active_dims or C.DIMENSIONS,
        "rebalance": rebalance or C.BACKTEST.rebalance_freq or "D",
        "display": comp.apply(to_display_score),
        "positions": build_positions(spread, rebalance=rebalance),
    }
