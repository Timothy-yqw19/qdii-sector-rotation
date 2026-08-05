"""
多板块对检验：这个宏观信号只在科技vs消费上成立，还是有普遍性？

    python multipair.py                # 跑全部板块对
    python multipair.py --fetch        # 先补抓新增标的
    python multipair.py --pairs=TECH-STAPLES,SEMIS-STAPLES

【为什么做这个】
单对样本只有 23 个第5档独立事件，撑不起任何二次切分。
扩到 8 对可以把事件数提升到 100+，同时回答一个更硬的问题：
跨板块对是否一致有效。**如果只在原始那一对上成立，那大概率是挖出来的。**

【方向先验怎么来】
不再手写。每个板块在 config.SECTOR_ATTRS 里有三个属性分
（久期 / 海外收入 / 贝塔），信号方向由 config.SIGNAL_ATTR_MAP 自动推导。
加板块只需填三个数，不必碰任何逻辑代码——这也让"先验是否合理"变得可审。
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
import explain_history as eh  # noqa: E402
from src import backtest as bt  # noqa: E402
from src import data_fetcher as dfetch  # noqa: E402
from src import regime as rg  # noqa: E402
from src import scoring  # noqa: E402
from src import signals as sig  # noqa: E402


@contextlib.contextmanager
def pair_context(long_s: str, short_s: str):
    """
    临时把全局配置切换成指定板块对。

    为什么用上下文管理器而不是给每个函数加参数：
    signals/scoring/backtest 三个模块都在调用时读取 config 的模块级属性，
    改成参数传递要动五个文件、十几个函数签名，
    而这里只需要保证"进去改、出来恢复"，风险低得多。
    """
    saved = (C.SECTORS, C.LONG_LEG, C.SHORT_LEG, C.SIGNALS, C.SIGNALS_BY_KEY)
    try:
        C.SECTORS = {
            s: {**C.SECTOR_TICKERS[s], "label": C.SECTOR_TICKERS[s]["label"]}
            for s in (long_s, short_s)
        }
        C.LONG_LEG, C.SHORT_LEG = long_s, short_s
        # 只保留宏观维度信号，并把方向换成由属性推导的版本
        C.SIGNALS = [
            C.Signal(key=s.key, name=s.name, dimension=s.dimension,
                     directions={sec: C.derive_directions(sec)[s.key]
                                 for sec in (long_s, short_s)},
                     rationale=s.rationale)
            for s in saved[3]
            if s.dimension == "macro_liquidity" and s.key in C.SIGNAL_ATTR_MAP
        ]
        C.SIGNALS_BY_KEY = {s.key: s for s in C.SIGNALS}
        yield
    finally:
        (C.SECTORS, C.LONG_LEG, C.SHORT_LEG, C.SIGNALS, C.SIGNALS_BY_KEY) = saved


def run_pair(long_s: str, short_s: str, prices: pd.DataFrame,
             macro: pd.DataFrame) -> dict | None:
    need = {C.SECTOR_TICKERS[long_s]["primary"],
            C.SECTOR_TICKERS[short_s]["primary"], C.BENCHMARK}
    have = set(prices["ticker"].unique())
    if not need <= have:
        print(f"  [skip] {long_s}-{short_s}: 缺标的 {sorted(need - have)}")
        return None

    with pair_context(long_s, short_s):
        sub = prices[prices["ticker"].isin(
            need | {p for s in C.SECTORS.values() for p in s["proxies"]})]
        panel = sig.build_signal_panel(sub, macro)
        legs = bt.leg_returns(sub)
        rel = bt.relative_return(legs)
        sc = scoring.run_scoring(panel, rel_ret=rel,
                                 active_dims=["macro_liquidity"])
        res = bt.run_strategy(sc["positions"], legs)
        nw = bt.newey_west_test(sc["spread"], rel, 20)
        fwd = bt.forward_relative_return(rel, 20)
        df = pd.concat([sc["spread"].rename("s"), fwd.rename("f")], axis=1).dropna()

        # 事件表（第5档），用滚动分位数分档，无后见之明
        q = sc["spread"].rolling(756, min_periods=252).apply(
            lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False)
        d = {"spread": sc["spread"], "quint": q, "fwd20": fwd,
             "fwd60": bt.forward_relative_return(rel, 60),
             "regime": rg.build_regime(macro, sc["spread"].index)}
        ep = eh.episodes(d, target=5)
        if not ep.empty:
            ep.insert(0, "pair", f"{long_s}-{short_s}")

        return {
            "pair": f"{C.SECTOR_TICKERS[long_s]['label']} vs "
                    f"{C.SECTOR_TICKERS[short_s]['label']}",
            "key": f"{long_s}-{short_s}",
            "IC_20d": bt.spearman(df["s"], df["f"]),
            "hit": float((np.sign(df["s"]) == np.sign(df["f"])).mean()),
            "t_NW_1y": nw.get("t_nw_1y", np.nan),
            "sharpe_net": res["stats"]["long_short_net"].get("sharpe", np.nan),
            "IR_vs_bench": res["stats"]["excess_vs_bench"].get("sharpe", np.nan),
            "turnover": res["stats"]["long_short_net"].get("ann_turnover", np.nan),
            "episodes": ep,
        }


def main() -> None:
    argv = sys.argv[1:]

    if "--fetch" in argv:
        print("补抓多板块对所需标的 ...")
        prices, meta = dfetch.fetch_prices(tickers=C.MULTIPAIR_TICKERS, pause=0.4)
        dfetch.save_prices(prices, root=ROOT, meta=meta)
        try:
            macro = dfetch.fetch_macro()
            dfetch.save_macro(macro, root=ROOT)
        except RuntimeError as e:
            print(f"  [降级] 宏观沿用缓存：{str(e).splitlines()[0]}")
    prices, macro = dfetch.load_cache(ROOT)

    pairs = C.SECTOR_PAIRS
    sel = next((a.split("=")[1] for a in argv if a.startswith("--pairs=")), None)
    if sel:
        want = set(sel.split(","))
        pairs = [p for p in pairs if f"{p[0]}-{p[1]}" in want]

    rows, all_ep = [], []
    for lo, sh in pairs:
        r = run_pair(lo, sh, prices, macro)
        if r is None:
            continue
        ep = r.pop("episodes")
        if ep is not None and not ep.empty:
            all_ep.append(ep)
        rows.append(r)

    if not rows:
        print("\n没有可用的板块对。多半是缺标的——先跑：python multipair.py --fetch")
        return

    tbl = pd.DataFrame(rows).set_index("pair").drop(columns=["key"])
    print(f"\n{'=' * 92}\n各板块对独立检验（仅宏观维度，月频）\n{'=' * 92}")
    print(tbl.round(3).to_string())

    ic = tbl["IC_20d"].dropna()
    sh = tbl["sharpe_net"].dropna()
    print(f"\n跨对汇总：IC 中位数 {ic.median():+.3f}，"
          f"{(ic > 0).sum()}/{len(ic)} 对为正；"
          f"净夏普中位数 {sh.median():+.3f}，{(sh > 0).sum()}/{len(sh)} 对为正")
    print("判读：若绝大多数对同号，说明这是个跨板块的宏观效应，而非单对上的偶然；"
          "若只有原始那一对亮眼，那基本可以判定是挖出来的。")

    if all_ep:
        pool = pd.concat(all_ep, ignore_index=True)
        v = pool["入场后20日(%)"].dropna()
        print(f"\n{'=' * 92}\n合并事件池\n{'=' * 92}")
        print(f"共 {len(pool)} 个第5档独立事件（单对时只有 23 个）")
        print(f"入场后20日：平均 {v.mean():+.2f}%、中位数 {v.median():+.2f}%、"
              f"胜率 {(v > 0).mean():.0%}")
        for st in ("压力", "平静"):
            s = pool[pool["状态"] == st]["入场后20日(%)"].dropna()
            if len(s):
                print(f"  {st}期 {len(s)} 次：平均 {s.mean():+.2f}%、"
                      f"胜率 {(s > 0).mean():.0%}")
        outdir = os.path.join(ROOT, C.OUTPUT_DIR)
        os.makedirs(outdir, exist_ok=True)
        pool.to_csv(f"{outdir}/episodes_pooled.csv", index=False)
        tbl.to_csv(f"{outdir}/multipair_summary.csv")
        print(f"\n落盘 → {C.OUTPUT_DIR}/episodes_pooled.csv, multipair_summary.csv")
        print("现在可以在这个更大的样本上重跑 find_patterns 的那套假设检验了。")


if __name__ == "__main__":
    main()
