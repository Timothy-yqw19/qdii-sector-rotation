"""
信号层 —— 整个项目的防前视核心。

三条铁律：
1. **标准化只用截止当日的信息**：滚动 z-score / 滚动分位数，绝不用全样本 min-max。
   全样本标准化会把未来的分布信息泄漏进历史信号，是量化回测最常见的致命错误。
2. **慢频宏观按真实发布滞后对齐**：先把观测日索引整体推迟 publication_lag_days，
   再 reindex 到交易日 forward-fill，最后才做差分/变化率变换。
   这样 t 日看到的宏观值，一定是 t 日现实中已经发布了的。
3. **方向对齐在最后一步**：z-score 保持原始经济含义，
   由 config 中的 directions 乘上去，得到“越高越利好该板块”的信号。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402


# ==========================================================================
# 基础工具：滚动标准化（防前视）
# ==========================================================================

def rolling_zscore(
    s: pd.Series,
    window: int | None = None,
    min_periods: int | None = None,
    clip: float | None = None,
) -> pd.Series:
    """
    滚动 z-score。t 时刻的均值/标准差只由 [t-window+1, t] 的数据算出。

    注意 rolling 默认包含当期值——这是正确的，因为 t 日的信号值在 t 日收盘后已知。
    """
    window = window or C.SCORING.zscore_window
    min_periods = min_periods or C.SCORING.zscore_min_periods
    clip = C.SCORING.zscore_clip if clip is None else clip

    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    z = (s - mu) / sd.replace(0, np.nan)
    return z.clip(-clip, clip) if clip else z


def rolling_percentile(s: pd.Series, window: int | None = None) -> pd.Series:
    """滚动分位数（0-1）。同样只用截止当日的历史窗口。"""
    window = window or C.SCORING.pctile_window
    min_periods = max(60, window // 4)
    # rolling.rank(pct=True) 给出当期值在窗口内的百分位排名
    return s.rolling(window, min_periods=min_periods).rank(pct=True)


# ==========================================================================
# 量价信号
# ==========================================================================

def _wide(prices: pd.DataFrame, col: str) -> pd.DataFrame:
    return prices.pivot(index="date", columns="ticker", values=col).sort_index()


def build_price_signals(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    对每个板块，用 primary + proxies 各自算信号后等权平均（降低单一标的噪音）。

    返回 {sector: DataFrame(index=date, columns=signal_key)}，值为**原始信号**
    （未标准化、未乘方向）。
    """
    close = _wide(prices, "close")
    volume = _wide(prices, "volume")
    bench = close[C.BENCHMARK]

    out: dict[str, pd.DataFrame] = {}
    for sector, meta in C.SECTORS.items():
        tickers = [meta["primary"]] + [p for p in meta["proxies"] if p in close.columns]
        per_ticker: list[pd.DataFrame] = []

        for t in tickers:
            px, vol = close[t], volume[t]
            rel = px / bench                       # 相对基准的价格比
            rel_ret = rel.pct_change()

            d = pd.DataFrame(index=close.index)
            d["rel_mom_20"] = rel / rel.shift(20) - 1
            d["rel_mom_60"] = rel / rel.shift(60) - 1
            realized_vol = rel_ret.rolling(60).std() * np.sqrt(252)
            d["risk_adj_mom_60"] = d["rel_mom_60"] / realized_vol.replace(0, np.nan)

            # 资金流代理：sign(收益)×成交额 的净额占总成交额比例
            dollar_vol = (px * vol).replace(0, np.nan)
            signed = np.sign(px.pct_change()) * dollar_vol
            d["money_flow_20"] = (
                signed.rolling(20).sum() / dollar_vol.rolling(20).sum()
            )

            # 估值代理：相对价格比的滚动分位数
            d["rel_price_pctile_756"] = rolling_percentile(rel)
            ma200 = px.rolling(200).mean()
            d["dist_ma200_pctile"] = rolling_percentile(px / ma200 - 1)

            per_ticker.append(d)

        out[sector] = sum(per_ticker) / len(per_ticker)
    return out


def apply_real_pe(
    price_signals: dict[str, pd.DataFrame], pe_df: pd.DataFrame | None
) -> dict[str, pd.DataFrame]:
    """若用户提供了真实 PE 历史，用真实 PE 的滚动分位数替换价格比代理。"""
    if pe_df is None:
        return price_signals
    pe_wide = pe_df.pivot(index="date", columns="ticker", values="pe").sort_index()
    for sector, meta in C.SECTORS.items():
        t = meta["primary"]
        if t not in pe_wide.columns:
            continue
        idx = price_signals[sector].index
        pe = pe_wide[t].reindex(idx).ffill()
        price_signals[sector]["rel_price_pctile_756"] = rolling_percentile(pe)
    return price_signals


# ==========================================================================
# 宏观信号（含发布滞后）
# ==========================================================================

_TRANSFORMS = {
    "level": lambda s: s,
    "diff_20d": lambda s: s.diff(20),
    "diff_60d": lambda s: s.diff(60),
    "pct_chg_60d": lambda s: s.pct_change(60),
}


def build_macro_signals(
    macro_raw: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    series: list[C.MacroSeries] | None = None,
) -> pd.DataFrame:
    """
    把原始 FRED 序列转成交易日频率的信号，严格施加发布滞后。

    步骤（顺序不可交换）：
      观测日索引 +lag → reindex 到交易日 ffill → 差分/变化率变换
    """
    out = pd.DataFrame(index=trading_index)
    for s in (series or C.MACRO_SERIES):
        if s.fred_id not in macro_raw.columns:
            print(f"  [warn] 缺少宏观序列 {s.fred_id}，跳过")
            continue
        raw = macro_raw[s.fred_id].dropna()
        # 关键一步：把观测日推迟到“真实可获得日”
        available = raw.copy()
        available.index = available.index + pd.Timedelta(days=s.publication_lag_days)
        aligned = available.reindex(
            available.index.union(trading_index)
        ).ffill().reindex(trading_index)
        out[f"{s.fred_id}_{s.transform}"] = _TRANSFORMS[s.transform](aligned)
    return out


# ==========================================================================
# 组装信号面板
# ==========================================================================

def build_signal_panel(
    prices: pd.DataFrame,
    macro_raw: pd.DataFrame,
    pe_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    产出 {sector: DataFrame(index=date, columns=signal_key)}，
    值为 **已标准化、已按方向对齐** 的信号（越高越利好该板块）。
    """
    trading_index = _wide(prices, "close").index
    px_sig = apply_real_pe(build_price_signals(prices), pe_df)
    mac_sig = build_macro_signals(macro_raw, trading_index)

    panel: dict[str, pd.DataFrame] = {}
    for sector in C.SECTORS:
        raw = pd.concat([px_sig[sector], mac_sig], axis=1)
        aligned = pd.DataFrame(index=trading_index)
        for sig in C.SIGNALS:
            if sig.key not in raw.columns:
                continue
            z = rolling_zscore(raw[sig.key])
            aligned[sig.key] = z * sig.directions.get(sector, 0.0)
        panel[sector] = aligned
    return panel


def raw_signal_panel(
    prices: pd.DataFrame, macro_raw: pd.DataFrame, pe_df: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    """未标准化的原始信号，仪表盘上展示“真实数值”时用。"""
    trading_index = _wide(prices, "close").index
    px_sig = apply_real_pe(build_price_signals(prices), pe_df)
    mac_sig = build_macro_signals(macro_raw, trading_index)
    return {
        sector: pd.concat([px_sig[sector], mac_sig], axis=1)
        for sector in C.SECTORS
    }
