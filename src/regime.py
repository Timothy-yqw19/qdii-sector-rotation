"""
Regime（风险状态）识别。

【设计要点】
1. **必须因果**：压力度量用的是已施加发布滞后的宏观序列 + 滚动分位数，
   任何时点只用截止当日的信息。regime 判定本身也会进防前视测试。
2. **必须迟滞**：用两个阈值（进入 0.70 / 退出 0.50）而不是一个。
   单阈值会让状态在边界反复横跳，制造大量无信息含量的换手——
   和仓位迟滞带是同一个道理。
3. **必须简单**：只分两态。样本里每年只有约 4.7 次独立押注，
   多态模型没有足够的观测去支撑。

【为什么用信用利差 + VIX 而不是别的】
两者都是市场**实时定价**的风险偏好，不需要等发布、不会被修订；
而 GDP/失业率这类宏观状态变量发布滞后长、修订大，用来定义 regime 会引入
"当时并不知道自己处在衰退中"的问题（NBER 认定衰退往往滞后一年以上）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
import src.signals as sig  # noqa: E402


def stress_index(
    macro_raw: pd.DataFrame, trading_index: pd.DatetimeIndex
) -> pd.Series:
    """
    市场压力指数 ∈ [0,1]：各压力序列水平值的滚动分位数取平均。

    注意走的是 signals.build_macro_signals 的 'level' 变换，
    因此**同样施加了发布滞后**——不会用到当天还没发布的数据。
    """
    probes = [
        C.MacroSeries(ms.fred_id, ms.name, ms.publication_lag_days, "level",
                      ms.expected_min_years)
        for ms in C.MACRO_SERIES if ms.fred_id in C.REGIME.stress_series
    ]
    if not probes:
        return pd.Series(np.nan, index=trading_index, name="stress")

    levels = sig.build_macro_signals(macro_raw, trading_index, series=probes)
    pct = pd.DataFrame({
        col: sig.rolling_percentile(levels[col], C.REGIME.pctile_window)
        for col in levels.columns
    })
    return pct.mean(axis=1, skipna=True).rename("stress")


def classify(stress: pd.Series) -> pd.Series:
    """
    压力指数 → {risk_on, risk_off}，带迟滞。

    上穿 enter_risk_off 进入 risk-off；只有跌回 exit_risk_off 以下才退出。
    中间地带维持上一状态。预热期（分位数还没值）标为 risk_on（中性默认）。
    """
    enter, exit_ = C.REGIME.enter_risk_off, C.REGIME.exit_risk_off
    state = "risk_on"
    out = []
    for v in stress.to_numpy():
        if not np.isnan(v):
            if state == "risk_on" and v >= enter:
                state = "risk_off"
            elif state == "risk_off" and v <= exit_:
                state = "risk_on"
        out.append(state)
    return pd.Series(out, index=stress.index, name="regime")


def build_regime(
    macro_raw: pd.DataFrame, trading_index: pd.DatetimeIndex
) -> pd.Series:
    return classify(stress_index(macro_raw, trading_index))


def summarize(regime: pd.Series) -> pd.DataFrame:
    """状态占比与持续时长，用来确认这个划分是否合理（别切出一堆一天就翻转的碎片）。"""
    blocks, cur, n = [], regime.iloc[0], 0
    for r in regime:
        if r == cur:
            n += 1
        else:
            blocks.append((cur, n))
            cur, n = r, 1
    blocks.append((cur, n))

    rows = []
    for state in ("risk_on", "risk_off"):
        lens = [n for s, n in blocks if s == state]
        rows.append({
            "状态": C.REGIME_LABELS[state],
            "占样本比例": f"{(regime == state).mean():.1%}",
            "区间段数": len(lens),
            "平均持续(交易日)": int(np.mean(lens)) if lens else 0,
            "最长(交易日)": int(max(lens)) if lens else 0,
        })
    return pd.DataFrame(rows).set_index("状态")
