"""
把模型的历史读数摊开，用来找规律 / 做案例复盘。

    python explain_history.py              # 近24个月末读数 + 历史极端档案例
    python explain_history.py --months 48  # 看更长
    python explain_history.py --weekly     # 近半年改成每周一行

输出同时落盘到 output/history_monthly.csv 与 output/episodes_q5.csv，
方便自己在 Excel 里翻。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config as C  # noqa: E402
from src import backtest as bt  # noqa: E402
from src import data_fetcher as dfetch  # noqa: E402
from src import regime as rg  # noqa: E402
from src import scoring  # noqa: E402
from src import signals as sig  # noqa: E402

SHORT = {
    "DFII10_diff_60d": "实际利率",
    "DTWEXBGS_pct_chg_60d": "美元",
    "BAA10Y_diff_20d": "信用利差",
    "VIXCLS_diff_20d": "VIX",
    "WALCL_pct_chg_60d": "联储扩表",
}


def build() -> dict:
    prices, macro = dfetch.load_cache(ROOT)
    dfetch.assert_cache_usable(prices)
    panel = sig.build_signal_panel(prices, macro, dfetch.load_optional_pe(ROOT))
    legs = bt.leg_returns(prices)
    rel = bt.relative_return(legs)
    scores = scoring.run_scoring(panel, rel_ret=rel)
    idx = prices.pivot(index="date", columns="ticker", values="close").index
    regime = rg.build_regime(macro, idx)

    spread = scores["spread"]
    # 分档用**滚动历史**分位数，避免用未来分布给过去分档（否则解读会自带后见之明）
    quint = spread.rolling(756, min_periods=252).apply(
        lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False)

    fwd20 = bt.forward_relative_return(rel, 20)
    fwd60 = bt.forward_relative_return(rel, 60)

    tbl = pd.DataFrame({
        "科技分": scores["display"]["TECH"].round(1),
        "消费分": scores["display"]["CONSUMER"].round(1),
        "分差": spread.round(2),
        "档位": (quint * 5).apply(lambda x: np.nan if pd.isna(x) else min(int(x) + 1, 5)),
        "状态": regime.reindex(spread.index).map(
            {"risk_on": "平静", "risk_off": "压力"}),
    })
    for k, name in SHORT.items():
        if k in panel["TECH"].columns:
            tbl[name] = panel["TECH"][k].round(2)
    tbl["未来20日"] = (fwd20 * 100).round(2)
    tbl["未来60日"] = (fwd60 * 100).round(2)
    return {"tbl": tbl, "spread": spread, "quint": quint,
            "fwd20": fwd20, "fwd60": fwd60, "regime": regime}


def episodes(d: dict, target: int = 5, min_gap: int = 40) -> pd.DataFrame:
    """
    找出历史上分差进入某一档的**独立事件**（间隔小于 min_gap 天的算同一段）。
    这是复盘的正确单位——连续 60 天都在第5档是**一次**押注，不是 60 次。
    """
    q = d["quint"].dropna()
    hits = q[(q * 5).apply(lambda x: min(int(x) + 1, 5)) == target].index
    if len(hits) == 0:
        return pd.DataFrame()

    rows, start, prev = [], hits[0], hits[0]
    for t in hits[1:]:
        if (t - prev).days > min_gap:
            rows.append((start, prev))
            start = t
        prev = t
    rows.append((start, prev))

    out = []
    for s, e in rows:
        seg = d["spread"].loc[s:e]
        out.append({
            "起": s.date(), "止": e.date(),
            "持续(交易日)": len(seg),
            "峰值分差": round(seg.max() if target >= 4 else seg.min(), 2),
            "状态": "压力" if (d["regime"].loc[s:e] == "risk_off").mean() > 0.5 else "平静",
            "入场后20日(%)": round(d["fwd20"].get(s, np.nan) * 100, 2),
            "入场后60日(%)": round(d["fwd60"].get(s, np.nan) * 100, 2),
        })
    return pd.DataFrame(out)


def main() -> None:
    argv = sys.argv[1:]
    months = int(next((a.split("=")[1] for a in argv if a.startswith("--months=")), 24))
    d = build()
    tbl = d["tbl"]

    outdir = os.path.join(ROOT, C.OUTPUT_DIR)
    os.makedirs(outdir, exist_ok=True)

    if "--weekly" in argv:
        show = tbl.resample("W-FRI").last().dropna(subset=["分差"]).tail(26)
        title = "近26周（每周五）"
    else:
        show = tbl.resample("ME").last().dropna(subset=["分差"]).tail(months)
        title = f"近{months}个月（每月末）"

    pd.set_option("display.width", 200)
    print(f"\n{'=' * 100}\n{title}读数\n{'=' * 100}")
    print("（未来20日/60日 = 该日之后科技相对消费的实际表现，最近几行为空是因为还没走完）\n")
    print(show.to_string())

    tbl.resample("ME").last().to_csv(f"{outdir}/history_monthly.csv")

    for tgt, label in [(5, "第5档（超配科技信号）"), (1, "第1档（最看空科技）")]:
        ep = episodes(d, target=tgt)
        print(f"\n{'=' * 100}\n历史上进入{label}的独立事件\n{'=' * 100}")
        if ep.empty:
            print("  无")
            continue
        print(ep.to_string(index=False))
        col = "入场后20日(%)"
        v = ep[col].dropna()
        if len(v):
            print(f"\n  共 {len(ep)} 次事件｜入场后20日平均 {v.mean():+.2f}%、"
                  f"中位数 {v.median():+.2f}%、胜率 {(v > 0).mean():.0%}")
            for st in ("压力", "平静"):
                s = ep[ep["状态"] == st][col].dropna()
                if len(s):
                    print(f"    {st}期 {len(s)} 次：平均 {s.mean():+.2f}%、"
                          f"胜率 {(s > 0).mean():.0%}")
        ep.to_csv(f"{outdir}/episodes_q{tgt}.csv", index=False)

    print(f"\n已落盘 → {C.OUTPUT_DIR}/history_monthly.csv, episodes_q5.csv, episodes_q1.csv")


if __name__ == "__main__":
    main()
