"""
生成报告与 README 用的核心图表 → charts/*.png

    python make_charts.py

刻意只画五张图，每张回答一个具体问题：
  01 净值        —— 策略相对基准表现如何？
  02 分层        —— 信号在哪一段有效？
  03 子样本      —— 是不是单一 regime 的产物？
  04 t值衰减     —— 显著性经得起自相关修正吗？
  05 维度筛选    —— 为什么砍掉两个维度？
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config as C  # noqa: E402
from src import backtest as bt  # noqa: E402
from src import data_fetcher as dfetch  # noqa: E402
from src import scoring  # noqa: E402
from src import signals as sig  # noqa: E402

OUT = os.path.join(ROOT, "charts")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
NAVY, RED, GREY, GREEN = "#1f3864", "#c0392b", "#7f8c8d", "#27ae60"


def _save(fig, name: str) -> None:
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {name}")


def main() -> None:
    prices, macro = dfetch.load_cache(ROOT)
    dfetch.assert_cache_usable(prices)
    panel = sig.build_signal_panel(prices, macro, dfetch.load_optional_pe(ROOT))
    legs = bt.leg_returns(prices)
    rel = bt.relative_return(legs)
    scores = scoring.run_scoring(panel, rel_ret=rel)          # 最终模型
    res = bt.run_strategy(scores["positions"], legs)

    # ---------- 01 净值 ----------
    r = res["returns"].dropna()
    eq = (1 + r).cumprod()
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(eq.index, eq["long_only_net"], color=NAVY, lw=1.6,
            label="Model tilt (net of costs)")
    a1.plot(eq.index, eq["benchmark_5050"], color=GREY, lw=1.3, ls="--",
            label="50/50 benchmark")
    a1.plot(eq.index, eq["static_long_leg"], color=RED, lw=1.1, ls=":",
            label="Static 100% QQQ (ex-post benchmark)")
    a1.set_yscale("log")
    a1.set_ylabel("Cumulative growth (log)")
    a1.set_title("Long-only tilt vs benchmarks — beats 50/50, loses to hindsight-optimal QQQ",
                 fontsize=11, loc="left")
    a1.legend(frameon=False, fontsize=9)

    cum_ex = (1 + r["excess_vs_bench"]).cumprod()
    a2.plot(cum_ex.index, cum_ex, color=GREEN, lw=1.4)
    a2.axhline(1.0, color=GREY, lw=0.8)
    a2.set_ylabel("Excess vs 50/50")
    a2.set_title(f"Information ratio {res['stats']['excess_vs_bench']['sharpe']:.2f}, "
                 f"tracking error {res['stats']['excess_vs_bench']['ann_vol']:.1%}",
                 fontsize=9.5, loc="left", color=GREY)
    _save(fig, "01_equity_curve.png")

    # ---------- 02 分层 ----------
    bk = bt.bucket_analysis(scores["spread"], rel, horizon=20)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = [RED if v < 0 else NAVY for v in bk["mean_fwd_ret"]]
    bars = ax.bar(range(1, len(bk) + 1), bk["mean_fwd_ret"] * 100, color=colors, width=0.62)
    for i, (b, h) in enumerate(zip(bars, bk["hit_rate_positive"])):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + (0.04 if b.get_height() >= 0 else -0.10),
                f"{h:.0%}", ha="center", fontsize=8.5, color=GREY)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(1, len(bk) + 1))
    ax.set_xlabel("Signal quintile (low → high)")
    ax.set_ylabel("Mean 20d forward relative return (%)")
    ax.set_title("Value is concentrated in the top quintile — an asymmetric signal\n"
                 "(labels = share of positive outcomes)", fontsize=11, loc="left")
    _save(fig, "02_quintiles.png")

    # ---------- 03 子样本 ----------
    sub = bt.subperiod_analysis(scores["spread"], prices)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = [p.replace(" → ", "\n→ ") for p in sub.index]
    a1.bar(range(len(sub)), sub["IC_20d"], color=NAVY, width=0.55)
    a1.axhline(0, color="black", lw=0.8)
    a1.set_xticks(range(len(sub)))
    a1.set_xticklabels(labels, fontsize=7.5)
    a1.set_ylabel("Spearman IC (20d)")
    a1.set_title("IC positive in all three sub-samples", fontsize=10, loc="left")

    a2.bar(range(len(sub)), sub["rel_drift_ann"] * 100, color=GREY, width=0.55)
    a2.axhline(0, color="black", lw=0.8)
    a2.set_xticks(range(len(sub)))
    a2.set_xticklabels(labels, fontsize=7.5)
    a2.set_ylabel("Tech-vs-Consumer drift (%/yr)")
    a2.set_title("...including the sub-period where tech underperformed",
                 fontsize=10, loc="left")
    _save(fig, "03_subperiods.png")

    # ---------- 04 t 值衰减 ----------
    nw = {h: bt.newey_west_test(scores["spread"], rel, h) for h in (5, 20)}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (h, d), col in zip(nw.items(), (GREY, NAVY)):
        if not d:
            continue
        xs = ["naive"] + [k.split("(")[0] for k in d["t_by_lag"]]
        ys = [d["t_naive"]] + list(d["t_by_lag"].values())
        ax.plot(xs, ys, marker="o", color=col, lw=1.6, label=f"{h}-day horizon")
        ax.annotate(f"{ys[-1]:.2f}", (len(xs) - 1, ys[-1]),
                    textcoords="offset points", xytext=(6, 0), fontsize=9, color=col)
    ax.axhline(2.0, color=RED, ls="--", lw=1, label="t = 2")
    ax.set_ylabel("t-statistic")
    ax.set_xlabel("Newey-West lag length")
    ax.set_title("t-stat stabilises above 3 even with one year of HAC lags\n"
                 "(the naive t is the number to throw away)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "04_newey_west.png")

    # ---------- 05 维度筛选 ----------
    dim_ic = bt.dimension_ic(scores["dimension_spread"], rel)
    dim_st = bt.single_dimension_strategies(scores["dimension_spread"], prices)
    names_en = {"动量与持仓": "Momentum &\npositioning",
                "宏观流动性与风险偏好": "Macro liquidity\n& risk appetite",
                "相对估值(均值回归)": "Relative\nvaluation"}
    idx = [names_en.get(i, i) for i in dim_ic.index]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, series, title, ylab in [
        (a1, dim_ic["IC_20d"], "IC (20d)", "Spearman IC"),
        (a2, dim_st["sharpe_ls_net"], "Standalone Sharpe (net)", "Sharpe"),
        (a3, dim_st["ann_turnover"], "Annual turnover", "×"),
    ]:
        vals = series.to_numpy()
        cols = [RED if v < 0 else NAVY for v in vals] if ax is not a3 \
            else [RED if v > 10 else NAVY for v in vals]
        ax.bar(range(len(vals)), vals, color=cols, width=0.55)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(idx, fontsize=7.5)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel(ylab, fontsize=9)
    fig.suptitle("Two of three dimensions were rejected by their own diagnostics",
                 fontsize=11.5, x=0.09, ha="left", y=1.02)
    _save(fig, "05_dimension_screening.png")

    print(f"\n{5} charts → charts/")


if __name__ == "__main__":
    main()
