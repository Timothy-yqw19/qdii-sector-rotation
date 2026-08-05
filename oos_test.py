"""
样本外检验：把模型往前推到 2000–2007，一段从未被用于任何设计决策的历史。

    python oos_test.py --fetch     # 首次运行：把价格拉到1998年、抓长历史宏观序列
    python oos_test.py             # 用缓存跑

【为什么这是比"多加板块对"更硬的检验】
多板块对的信号两两相关中位数 0.82、第一主成分占 77%，
等效独立对数只有约 1.6 个——8 个对本质是同一笔交易换了几件衣服。
换时间段则不同：2000–2007 与 2008–2026 在时间上完全不重叠，
且包含科网泡沫破裂与 2004–2006 加息周期这两个全新的宏观环境。
本项目的所有设计决策（砍哪两个维度、用哪些信号、月频、迟滞带）
都是在 2008 年之后的数据上做的，因此 2000–2007 是真正干净的样本外。

【必须接受的妥协，以及为此做的控制】
5 个宏观信号里 3 个在 2007 年前不存在，只能用长历史替代：
    实际利率 → DGS10 名义10年美债      （1962年起）
    广义美元 → DTWEXM 主要货币美元指数  （1973–2020）
    联储扩表 → 丢弃（无合适的长历史替代）
关键控制：**样本外段与对照段跑的是同一个 4 信号简化模型**。
否则"样本外更差"会分不清是模型失效还是信号变少。
DTWEXM 于 2020 年停更，故对照段取 2008–2019。
"""

from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config as C  # noqa: E402
from src import backtest as bt  # noqa: E402
from src import data_fetcher as dfetch  # noqa: E402
from src import scoring  # noqa: E402
from src import signals as sig  # noqa: E402

MACRO_OOS_CACHE = "data/macro_long.csv"


@contextlib.contextmanager
def oos_pair_context(long_s: str, short_s: str):
    """切换到指定板块对，并把信号集换成 4 个长历史信号（方向仍由属性推导）。"""
    saved = (C.SECTORS, C.LONG_LEG, C.SHORT_LEG, C.SIGNALS, C.SIGNALS_BY_KEY)
    try:
        C.SECTORS = {s: dict(C.SECTOR_TICKERS[s]) for s in (long_s, short_s)}
        C.LONG_LEG, C.SHORT_LEG = long_s, short_s
        C.SIGNALS = [
            C.Signal(key=key, name=key, dimension="macro_liquidity",
                     directions={sec: C.derive_directions_oos(sec)[key]
                                 for sec in (long_s, short_s)},
                     rationale="长历史替代序列")
            for key in C.OOS_SIGNAL_ATTR_MAP
        ]
        C.SIGNALS_BY_KEY = {s.key: s for s in C.SIGNALS}
        yield
    finally:
        (C.SECTORS, C.LONG_LEG, C.SHORT_LEG, C.SIGNALS, C.SIGNALS_BY_KEY) = saved


def fetch_long_history() -> None:
    print("拉取 1998 年起的价格（Tiingo）...")
    prices, meta = dfetch.fetch_prices(
        start=C.OOS_PRICE_START, tickers=C.MULTIPAIR_TICKERS, pause=0.4)
    dfetch.save_prices(prices, root=ROOT, meta=meta)

    print("\n拉取长历史宏观序列 ...")
    macro = dfetch.fetch_macro(series=C.OOS_SERIES)
    macro.to_csv(os.path.join(ROOT, MACRO_OOS_CACHE))
    print(f"  → {MACRO_OOS_CACHE}")
    print(dfetch.report_macro_coverage(macro).to_string(index=False))


def evaluate(spread: pd.Series, legs: pd.DataFrame, lo: str, hi: str) -> dict:
    """在指定时间窗内评估。窗内单独算 IC 与策略表现。"""
    rel = bt.relative_return(legs)
    m = (spread.index >= lo) & (spread.index <= hi)
    sp = spread[m]
    if sp.notna().sum() < 300:
        return {}
    fwd = bt.forward_relative_return(rel, 20)
    df = pd.concat([sp.rename("s"), fwd.rename("f")], axis=1).dropna()
    if len(df) < 200:
        return {}
    pos = scoring.build_positions(sp)
    res = bt.run_strategy(pos, legs.loc[m])
    nw = bt.newey_west_test(sp, rel[m], 20)
    return {
        "IC_20d": bt.spearman(df["s"], df["f"]),
        "hit": float((np.sign(df["s"]) == np.sign(df["f"])).mean()),
        "t_NW_1y": nw.get("t_nw_1y", np.nan),
        "sharpe_net": res["stats"]["long_short_net"].get("sharpe", np.nan),
        "n_days": len(df),
    }


def main() -> None:
    if "--fetch" in sys.argv:
        fetch_long_history()

    path = os.path.join(ROOT, MACRO_OOS_CACHE)
    if not os.path.exists(path):
        print("缺少长历史宏观缓存。先跑：python oos_test.py --fetch")
        return
    macro = pd.read_csv(path, index_col=0, parse_dates=True)
    prices, _ = dfetch.load_cache(ROOT)

    start = prices["date"].min()
    print(f"\n价格数据起点: {start.date()}")
    if start > pd.Timestamp("2000-06-01"):
        print("  [!] 价格没有回溯到 1999 年，样本外段会很短。"
              "先跑：python oos_test.py --fetch")

    rows = []
    for lo, sh in C.SECTOR_PAIRS:
        need = {C.SECTOR_TICKERS[lo]["primary"],
                C.SECTOR_TICKERS[sh]["primary"], C.BENCHMARK}
        if not need <= set(prices["ticker"].unique()):
            continue
        with oos_pair_context(lo, sh):
            sub = prices[prices["ticker"].isin(need)]
            panel = sig.build_signal_panel(sub, macro, macro_series=C.OOS_SERIES)
            legs = bt.leg_returns(sub)
            sc = scoring.run_scoring(panel, rel_ret=bt.relative_return(legs),
                                     active_dims=["macro_liquidity"])
            label = (f"{C.SECTOR_TICKERS[lo]['label']} vs "
                     f"{C.SECTOR_TICKERS[sh]['label']}")
            for pname, (a, b) in C.OOS_PERIODS.items():
                r = evaluate(sc["spread"], legs, a, b)
                if r:
                    rows.append({"板块对": label, "时段": pname, **r})

    if not rows:
        print("没有可评估的组合——多半是价格历史不够长。先跑 --fetch。")
        return

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 88}\n逐对 × 逐时段（4信号简化模型，两段口径完全一致）\n{'=' * 88}")
    piv = df.pivot(index="板块对", columns="时段", values="IC_20d")
    piv.columns = [f"IC {c}" for c in piv.columns]
    sh_piv = df.pivot(index="板块对", columns="时段", values="sharpe_net")
    for c in sh_piv.columns:
        piv[f"夏普 {c}"] = sh_piv[c]
    print(piv.round(3).to_string())

    print(f"\n{'=' * 88}\n汇总\n{'=' * 88}")
    for pname in C.OOS_PERIODS:
        s = df[df["时段"] == pname]
        if s.empty:
            continue
        ic, shp = s["IC_20d"], s["sharpe_net"]
        print(f"{pname}:  IC 中位数 {ic.median():+.3f}  "
              f"({(ic > 0).sum()}/{len(ic)} 对为正)  |  "
              f"净夏普中位数 {shp.median():+.3f}  "
              f"({(shp > 0).sum()}/{len(shp)} 对为正)  |  "
              f"平均样本 {s['n_days'].mean():.0f} 天")

    oos = df[df["时段"].str.contains("样本外")]["IC_20d"]
    ins = df[df["时段"].str.contains("对照")]["IC_20d"]
    if len(oos) and len(ins):
        print(f"\n样本外 IC 中位数 / 对照段 = {oos.median() / ins.median():.0%}" if
              ins.median() != 0 else "")
        print("""
判读标准（事先声明）：
  · 样本外 IC 中位数为正、且多数板块对同号  → 模型有跨期普适性，这是最好的结果
  · 样本外 IC 明显小于对照段但仍为正        → 存在过拟合成分，但核心逻辑站得住
  · 样本外 IC ≈ 0 或为负                    → 之前的结果主要是2008年后特定环境的产物，
                                             必须在结论里明确限定适用区间""")

    outdir = os.path.join(ROOT, C.OUTPUT_DIR)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(f"{outdir}/oos_results.csv", index=False)
    print(f"\n落盘 → {C.OUTPUT_DIR}/oos_results.csv")


if __name__ == "__main__":
    main()
