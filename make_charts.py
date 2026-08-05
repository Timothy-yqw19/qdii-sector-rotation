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

    make_oos_charts()
    print("\ncharts → charts/")


# ==========================================================================
# 样本外检验图（需先跑 oos_test.py --fetch 生成 data/macro_long.csv）
# ==========================================================================

def make_oos_charts() -> None:
    import config as Cfg
    macro_path = os.path.join(ROOT, "data", "macro_long.csv")
    if not os.path.exists(macro_path):
        print("  [skip] 缺 data/macro_long.csv，跳过样本外图（先跑 ./run.sh oos-fetch）")
        return

    import oos_test as oos
    macro_long = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    prices, _ = dfetch.load_cache(ROOT)
    if prices["date"].min() > pd.Timestamp("2000-06-01"):
        print("  [skip] 价格未回溯到1999年，跳过样本外图")
        return

    # ---------- 06 分段 IC 断点 ----------
    with oos.oos_pair_context("TECH", "STAPLES"):
        sub = prices[prices["ticker"].isin({"XLK", "XLP", "SPY"})]
        panel = sig.build_signal_panel(sub, macro_long, macro_series=Cfg.OOS_SERIES)
        legs = bt.leg_returns(sub)
        rel = bt.relative_return(legs)
        sc = scoring.run_scoring(panel, rel_ret=rel, active_dims=["macro_liquidity"])
    spread = sc["spread"].dropna()
    fwd = bt.forward_relative_return(rel, 20)

    segs = [("2000-01", "2003-12"), ("2004-01", "2007-12"),
            ("2008-01", "2013-12"), ("2014-01", "2019-12")]
    ics = []
    for lo, hi in segs:
        m = (spread.index >= lo) & (spread.index <= hi)
        df = pd.concat([spread[m].rename("s"), fwd.rename("f")], axis=1).dropna()
        ics.append(bt.spearman(df["s"], df["f"]) if len(df) > 200 else np.nan)

    fig, ax = plt.subplots(figsize=(8, 4))
    cols = [GREY, GREY, NAVY, NAVY]
    ax.bar(range(4), ics, color=cols, width=0.55)
    ax.axvline(1.5, color=RED, ls="--", lw=1.4)
    # 图内一律用英文：容器里没有中文字体，混中文会渲染成方框
    ax.text(1.55, max(ics) * 0.92, "  2008: QE era begins", fontsize=9.5, color=RED)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{a}\n~{b}" for a, b in segs], fontsize=8.5)
    ax.set_ylabel("Spearman IC (20d)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("A clean structural break at 2008\n"
                 "The same 4-signal model is 3–5× stronger after QE begins",
                 fontsize=11, loc="left")
    _save(fig, "06_regime_break.png")

    # ---------- 07 样本外 vs 对照段逐对 ----------
    path = os.path.join(ROOT, C.OUTPUT_DIR, "oos_results.csv")
    if os.path.exists(path):
        r = pd.read_csv(path)
        piv = r.pivot(index="板块对", columns="时段", values="sharpe_net")
        oos_col = [c for c in piv.columns if "样本外" in c][0]
        ins_col = [c for c in piv.columns if "对照" in c][0]
        piv = piv.sort_values(ins_col)
        names_en = {"半导体 vs 必需消费": "Semis vs Staples",
                    "可选消费 vs 必需消费": "Disc. vs Staples",
                    "工业 vs 必需消费": "Indust. vs Staples",
                    "科技 vs 公用事业": "Tech vs Utilities",
                    "科技 vs 医疗": "Tech vs Health",
                    "科技 vs 可选消费": "Tech vs Disc.",
                    "科技 vs 必需消费": "Tech vs Staples",
                    "金融 vs 公用事业": "Fin. vs Utilities"}
        y = np.arange(len(piv))
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.barh(y + 0.2, piv[ins_col], height=0.38, color=NAVY,
                label="2008–2019 (in-sample era)")
        ax.barh(y - 0.2, piv[oos_col], height=0.38, color=RED,
                label="2000–2007 (out-of-sample)")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([names_en.get(i, i) for i in piv.index], fontsize=8.5)
        ax.set_xlabel("Net Sharpe")
        ax.set_title("Out-of-sample: every single pair turns negative\n"
                     "8/8 positive in-sample → 0/8 positive out-of-sample",
                     fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        _save(fig, "07_oos_by_pair.png")

    # ---------- 08 科网泡沫期：模型说什么 vs 实际发生什么 ----------
    m = (spread.index >= "2000-01-01") & (spread.index <= "2004-06-30")
    sp, rr = spread[m], rel[(rel.index >= "2000-01-01") & (rel.index <= "2004-06-30")]
    cum = (1 + rr.fillna(0)).cumprod()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.3]})
    a1.fill_between(sp.index, 0, sp, where=sp > 0, color=NAVY, alpha=0.75,
                    label="Model says: overweight tech")
    a1.fill_between(sp.index, 0, sp, where=sp <= 0, color=GREY, alpha=0.6,
                    label="Model says: underweight tech")
    a1.axhline(0, color="black", lw=0.8)
    a1.set_ylabel("Signal spread (z)")
    a1.legend(frameon=False, fontsize=8.5, loc="upper right")
    a1.set_title("Dot-com bust: the model was long tech through a 50% collapse",
                 fontsize=11, loc="left")

    a2.plot(cum.index, cum, color=RED, lw=1.6)
    a2.axhline(1.0, color=GREY, lw=0.8)
    a2.set_ylabel("Tech vs Staples (cumulative)")
    a2.set_title("Fed cut 6.5% → 1%; the rate signal read it as a tailwind. "
                 "It was a response to the crash.", fontsize=9.5, loc="left",
                 color=GREY)
    _save(fig, "08_dotcom_failure.png")


if __name__ == "__main__":
    main()
