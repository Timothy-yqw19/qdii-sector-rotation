"""
在历史事件上检验**事先声明的**假设。

    python find_patterns.py

【为什么要事先声明假设】
第5档只有 23 个独立事件。在这么小的样本上反复切分，
总能切出一个"看起来很强"的分组——那是多重检验陷阱，不是规律。

所以本脚本只测下面 6 条假设，而且每条都：
  · 有事先的经济学理由（不是看了数据才想出来的）
  · 报告分组样本量（少于 8 个就当轶事看，别当规律）
  · 报告随机情况下能得到同样结果的概率（置换检验）

判读原则：
  n < 8         → 轶事，不足以支撑任何规则
  p > 0.10      → 与噪音无法区分
  效应小于 1%   → 即使显著也不值得为它增加模型复杂度
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config as C  # noqa: E402
import explain_history as eh  # noqa: E402
from src import signals as sig  # noqa: E402

MACRO_KEYS = [s.key for s in C.signals_in("macro_liquidity")]
RNG = np.random.default_rng(42)


def enrich(ep: pd.DataFrame, panel: dict) -> pd.DataFrame:
    """给每个事件补上入场当日的信号特征。"""
    tech = panel["TECH"]
    rows = []
    for _, r in ep.iterrows():
        d = pd.Timestamp(r["起"])
        if d not in tech.index:
            nearest = tech.index[tech.index <= d]
            if len(nearest) == 0:
                continue
            d = nearest[-1]
        vals = tech.loc[d, [k for k in MACRO_KEYS if k in tech.columns]].dropna()
        pos = int((vals > 0).sum())
        rows.append({
            **r.to_dict(),
            "同向个数": pos,                       # 5个信号里有几个利好科技
            "一致度": abs(pos - len(vals) / 2) / (len(vals) / 2),  # 0=完全打架, 1=完全一致
            "实际利率": vals.get("DFII10_diff_60d", np.nan),
            "联储扩表": vals.get("WALCL_pct_chg_60d", np.nan),
            "信用利差": vals.get("BAA10Y_diff_20d", np.nan),
        })
    return pd.DataFrame(rows)


def permutation_p(a: np.ndarray, b: np.ndarray, n: int = 20000) -> float:
    """两组均值差的置换检验 p 值（双尾）。小样本下比 t 检验稳。"""
    if len(a) < 2 or len(b) < 2:
        return np.nan
    obs = abs(np.mean(a) - np.mean(b))
    pool = np.concatenate([a, b])
    k = len(a)
    cnt = 0
    for _ in range(n):
        RNG.shuffle(pool)
        if abs(np.mean(pool[:k]) - np.mean(pool[k:])) >= obs:
            cnt += 1
    return cnt / n


def test(name: str, ep: pd.DataFrame, mask: pd.Series,
         lab_true: str, lab_false: str, col: str = "入场后20日(%)") -> dict:
    a = ep.loc[mask, col].dropna().to_numpy()
    b = ep.loc[~mask, col].dropna().to_numpy()
    p = permutation_p(a, b)
    n_min = min(len(a), len(b))
    diff = np.mean(a) - np.mean(b) if len(a) and len(b) else np.nan

    if n_min < 8:
        verdict = f"轶事(n={n_min})"
    elif p < 0.10 and abs(diff) > 1.0:
        verdict = "★ 可能有信息"
    else:
        verdict = "与噪音难区分"

    def fmt(x):
        return f"{np.mean(x):+.2f}% n={len(x)} 胜{np.mean(x > 0):.0%}" if len(x) else "-"

    return {
        "假设": name,
        "分组A": lab_true, "A结果": fmt(a),
        "分组B": lab_false, "B结果": fmt(b),
        "差异": f"{diff:+.2f}%" if not np.isnan(diff) else "-",
        "p值": f"{p:.3f}" if not np.isnan(p) else "-",
        "判读": verdict,
    }


def main() -> None:
    d = eh.build()
    from src import data_fetcher as dfetch
    prices, macro = dfetch.load_cache(ROOT)
    panel = sig.build_signal_panel(prices, macro, dfetch.load_optional_pe(ROOT))

    ep = enrich(eh.episodes(d, target=5), panel)
    print(f"\n第5档独立事件 {len(ep)} 个，"
          f"整体入场后20日平均 {ep['入场后20日(%)'].mean():+.2f}%、"
          f"胜率 {(ep['入场后20日(%)'] > 0).mean():.0%}\n")

    results = [
        # H1 压力期传导更强（regime诊断已提示，此处在事件层面复核）
        test("H1 压力期 vs 平静期", ep, ep["状态"] == "压力", "压力期", "平静期"),
        # H2 信号一致时可信度更高（分散化逻辑的直接推论）
        test("H2 信号高度一致(≥4/5同向)", ep, ep["同向个数"] >= 4, "一致", "打架"),
        # H3 信号越极端越可靠（分层测试的延伸）
        test("H3 峰值分差 > 中位数", ep, ep["峰值分差"] > ep["峰值分差"].median(),
             "更极端", "较温和"),
        # H4 短事件多为噪音，长事件是真趋势
        test("H4 持续 > 40 个交易日", ep, ep["持续(交易日)"] > 40, "长事件", "短事件"),
        # H5 联储扩表是最强单信号，它同向时应更好
        test("H5 入场时联储扩表为正", ep, ep["联储扩表"] > 0, "扩表利好", "扩表不利"),
        # H6 实际利率是最弱单信号，预期无差别（作为阴性对照）
        test("H6 入场时实际利率为正(阴性对照)", ep, ep["实际利率"] > 0, "利率利好", "利率不利"),
    ]
    print(pd.DataFrame(results).to_string(index=False))

    print(f"""
{'=' * 96}
怎么读这张表
{'=' * 96}
· 一共测了 {len(results)} 条假设。在 p<0.10 的门槛下，即使全是噪音，
  也预期会出现 {len(results) * 0.1:.1f} 条"显著"的结果。所以看到一两条"显著"不要激动。
· H6 是**阴性对照**：实际利率是最弱的单信号，理论上不该有区分度。
  如果它反而"显著"，说明这套切分方法本身在制造假信号，其余结论也要打折。
· 真正值得采纳的条件（三者同时满足）：两组 n 都 ≥ 8、p < 0.10、
  效应 > 1%，并且**你能讲出一个事前就成立的经济学理由**。
· 只满足统计条件、讲不出理由的，一律当噪音处理。
""")

    ep.to_csv(os.path.join(ROOT, C.OUTPUT_DIR, "episodes_q5_enriched.csv"), index=False)
    print(f"带信号特征的事件表 → {C.OUTPUT_DIR}/episodes_q5_enriched.csv")


if __name__ == "__main__":
    main()
