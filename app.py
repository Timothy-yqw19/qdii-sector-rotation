"""
Streamlit 仪表盘。

    streamlit run app.py

页面结构刻意按“研究员汇报顺序”排：
  当前观点 → 观点是怎么来的（维度/信号拆解） → 历史上它准不准（回测）
  → 未回测的实时增强层（明确标注） → 方法论与已知局限
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config as C  # noqa: E402
from src import backtest as bt  # noqa: E402
from src import data_fetcher as dfetch  # noqa: E402
from src import scoring  # noqa: E402
from src import signals as sig  # noqa: E402

st.set_page_config(page_title="全球宏观-板块轮动打分系统", layout="wide")


# ==========================================================================
# 数据加载与计算（缓存）
# ==========================================================================

@st.cache_data(show_spinner="加载数据与计算信号 ...")
def load_all():
    prices, macro = dfetch.load_cache(root=ROOT)
    dfetch.assert_cache_usable(prices)
    meta = dfetch.load_meta(root=ROOT)
    pe = dfetch.load_optional_pe(root=ROOT)
    panel = sig.build_signal_panel(prices, macro, pe)
    raw = sig.raw_signal_panel(prices, macro, pe)
    return prices, macro, panel, raw, meta


@st.cache_data(show_spinner="打分 ...")
def score_with(_panel, prices, scheme: str):
    rel = bt.relative_return(bt.leg_returns(prices))
    return scoring.run_scoring(_panel, scheme=scheme, rel_ret=rel)


@st.cache_data(show_spinner="跑回测 ...")
def run_bt(_scores, prices, with_sens: bool):
    return bt.run_full_backtest(_scores, prices, run_sensitivity=with_sens)


@st.cache_data(show_spinner="对比三种权重方案 ...")
def compare_schemes(_panel, prices):
    return bt.compare_weight_schemes(_panel, prices)


try:
    prices, macro, panel, raw_panel, meta = load_all()
except FileNotFoundError:
    st.error("找不到 data/ 缓存。先跑一次：`python run_pipeline.py --fetch`"
             "（无网络时用 `--offline` 生成合成数据验证界面）")
    st.stop()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("参数")
    scheme = st.selectbox(
        "维度权重方案",
        ["equal", "inverse_vol", "ic_weighted"],
        index=["equal", "inverse_vol", "ic_weighted"].index(C.SCORING.weight_scheme),
        format_func=lambda s: {
            "equal": "等权（基线）",
            "inverse_vol": "逆波动（风险平价）",
            "ic_weighted": "滚动IC加权（收缩后）",
        }[s],
    )
    with_sens = st.checkbox("跑权重敏感性分析（较慢）", value=False)
    show_compare = st.checkbox("对比三种权重方案", value=False)
    band = st.slider("迟滞带 (抑制换手)", 0.0, 0.6, C.BACKTEST.hysteresis_band, 0.05)
    cost = st.slider("单边成本 (bps)", 0.0, 20.0, C.BACKTEST.cost_bps, 1.0)
    st.caption("调这两个参数看结论是否稳健——如果一加成本就没了，"
               "说明信号强度撑不起日频换手。")
    st.divider()
    st.caption(f"价格源：{meta.get('price_source', '未知')}｜"
               f"{'已分红复权' if meta.get('dividend_adjusted') else '未复权'}")

scores = score_with(panel, prices, scheme)

_bias = dfetch.dividend_bias_note(meta)
if _bias:
    st.warning(f"**数据口径提示**：{_bias}")

# 侧栏参数会改变仓位与成本，重算这两块
scores = dict(scores)
scores["positions"] = scoring.build_positions(scores["spread"], band=band)
results = run_bt(scores, prices, with_sens)
legs = results["legs"]
results["strategy"] = bt.run_strategy(scores["positions"], legs, cost_bps=cost)

LONG_LABEL = C.SECTORS[C.LONG_LEG]["label"]
SHORT_LABEL = C.SECTORS[C.SHORT_LEG]["label"]

# ==========================================================================
# 0. 标题
# ==========================================================================
st.title("全球宏观-板块轮动打分系统")
st.caption(
    f"美股板块相对强弱 · 多 {LONG_LABEL}({C.SECTORS[C.LONG_LEG]['primary']}) "
    f"/ 空 {SHORT_LABEL}({C.SECTORS[C.SHORT_LEG]['primary']}) · "
    f"信号T日收盘生成，T+1收盘执行，T+2起计收益"
)

# ==========================================================================
# 1. 当前观点
# ==========================================================================
st.header("1 · 当前观点")

disp = scores["display"].dropna()
spread = scores["spread"].dropna()
pos = scores["positions"].dropna()
asof = disp.index[-1].date()

c1, c2, c3, c4 = st.columns(4)
prev = disp.iloc[-22] if len(disp) > 22 else disp.iloc[0]
c1.metric(f"{LONG_LABEL}", f"{disp[C.LONG_LEG].iloc[-1]:.0f} / 100",
          f"{disp[C.LONG_LEG].iloc[-1] - prev[C.LONG_LEG]:+.1f} (20日)")
c2.metric(f"{SHORT_LABEL}", f"{disp[C.SHORT_LEG].iloc[-1]:.0f} / 100",
          f"{disp[C.SHORT_LEG].iloc[-1] - prev[C.SHORT_LEG]:+.1f} (20日)")
c3.metric("轮动分差 (z)", f"{spread.iloc[-1]:+.2f}")
c4.metric("建议倾斜仓位", f"{pos.iloc[-1]:+.2f}",
          help="+1 = 全部配置多头腿；-1 = 全部配置空头腿；0 = 50/50中性")

tilt = pos.iloc[-1]
if tilt > 0.3:
    st.success(f"**超配 {LONG_LABEL}** —— 分差 {spread.iloc[-1]:+.2f}z（截至 {asof}）")
elif tilt < -0.3:
    st.success(f"**超配 {SHORT_LABEL}** —— 分差 {spread.iloc[-1]:+.2f}z（截至 {asof}）")
else:
    st.info(f"**中性** —— 分差 {spread.iloc[-1]:+.2f}z 未突破迟滞带（截至 {asof}）")

# ==========================================================================
# 2. 观点拆解
# ==========================================================================
st.header("2 · 这个观点是怎么来的")

dim_now = pd.DataFrame({
    C.DIMENSION_LABELS[d]: {
        C.SECTORS[s]["label"]: scores["dimension_scores"][s][d].dropna().iloc[-1]
        if scores["dimension_scores"][s][d].notna().any() else np.nan
        for s in C.SECTORS
    } for d in C.DIMENSIONS
}).T
dim_now["分差(多-空)"] = dim_now[LONG_LABEL] - dim_now[SHORT_LABEL]

left, right = st.columns([1, 1])
with left:
    st.subheader("维度分 (z单位)")
    st.dataframe(dim_now.round(2), use_container_width=True)
    st.bar_chart(dim_now["分差(多-空)"])
    st.caption("维度内取均值消除同类信号重复计票。"
               "精确到5%的权重是虚假精度——稳健性由第3节的方案对比与敏感性证明。")

    w = scores["weights"]
    if isinstance(w, pd.DataFrame) and w.notna().any().any():
        st.subheader(f"维度权重（方案: {scheme}）")
        wl = w.dropna(how="all").rename(columns=C.DIMENSION_LABELS)
        cur = wl.iloc[-1]
        st.dataframe(cur.rename("当前权重").to_frame().T.round(3),
                     use_container_width=True)
        if scheme != "equal":
            st.line_chart(wl)
            st.caption("IC加权已向等权收缩且滞后 "
                       f"{C.SCORING.ic_weight_horizon + C.BACKTEST.execution_lag - 1} "
                       "个交易日，只用已实现收益估计——防前视测试覆盖了这一点。")

with right:
    st.subheader("信号明细")
    rows = []
    for s in C.SIGNALS:
        r = {"信号": s.name, "维度": C.DIMENSION_LABELS[s.dimension]}
        for sector, meta in C.SECTORS.items():
            if s.key in raw_panel[sector].columns:
                rv = raw_panel[sector][s.key].dropna()
                zv = panel[sector][s.key].dropna()
                r[f"{meta['label']}·原值"] = rv.iloc[-1] if len(rv) else np.nan
                r[f"{meta['label']}·对齐z"] = zv.iloc[-1] if len(zv) else np.nan
        rows.append(r)
    st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, height=380)
    st.caption("“对齐z”= 滚动z-score × 方向先验，已经是“越高越利好该板块”的口径。")

with st.expander("每个信号的方向先验及其经济学理由（面试必问）"):
    for s in C.SIGNALS:
        dirs = " / ".join(
            f"{C.SECTORS[k]['label']}: {v:+.1f}" for k, v in s.directions.items()
        )
        st.markdown(f"**{s.name}** （{C.DIMENSION_LABELS[s.dimension]}） — {dirs}  \n{s.rationale}")

# ==========================================================================
# 3. 回测
# ==========================================================================
st.header("3 · 历史上它准不准")

st.subheader("3.1 信息系数 (IC)")
st.dataframe(results["ic"].round(4), use_container_width=True)
st.caption("重叠窗口会严重高估统计显著性，故单列非重叠采样下的 IC 与 t 值。"
           "|t| < 2 就该老实承认预测力不显著。")

st.subheader("3.2 滚动 IC —— 信号有没有衰减")
ric = results["rolling_ic_20d"].dropna()
if not ric.empty:
    st.line_chart(ric.rename("252日滚动IC (20日前瞻)"))
    st.caption("如果近几年系统性下移，说明这个alpha正在被市场消化——"
               "这是必须主动交代的，而不是等面试官问出来。")

st.subheader("3.3 分层测试 —— 什么时候最有效")
h = st.radio("前瞻窗口", list(C.BACKTEST.ic_horizons), horizontal=True,
             format_func=lambda x: f"{x}日")
bk = results["buckets"][h]
if not bk.empty:
    b1, b2 = st.columns([1, 1])
    b1.dataframe(bk.round(4), use_container_width=True)
    b2.bar_chart(bk["mean_fwd_ret"].rename("各档平均前瞻相对收益"))
    st.caption("若只有最高/最低档显著，这就是一个**极端仓位确认器**而非日常择时器——"
               "该这么定位，就这么讲。")

st.subheader("3.4 净值与成本")
rets = results["strategy"]["returns"]
show = st.multiselect(
    "曲线", list(rets.columns),
    default=["long_only_net", "benchmark_5050", "long_short_net"],
)
if show:
    st.line_chart((1 + rets[show].fillna(0)).cumprod())
stats = pd.DataFrame(results["strategy"]["stats"]).T
st.dataframe(
    stats[[c for c in ["ann_return", "ann_vol", "sharpe", "max_drawdown",
                       "hit_rate_daily", "ann_turnover"] if c in stats.columns]].round(3),
    use_container_width=True,
)
st.caption(f"当前设定：单边 {cost:.0f}bps、迟滞带 {band:.2f}。"
           "long_only_net 是公募QDII实际可执行的形态（50/50基准上做倾斜），"
           "long_short_net 才是信号本身价值的干净度量。")

if show_compare:
    st.subheader("3.5 权重方案对比")
    cmp = compare_schemes(panel, prices)
    st.dataframe(cmp.round(4), use_container_width=True)
    st.caption(
        "同一套信号、同一套回测口径，只换权重方案。若 ic_weighted 相对 equal "
        "的提升在成本后所剩无几，就该老实用等权——学术与业界的共识都是"
        "等权是极难被稳定打败的基线。这张表本身就是对『你权重怎么定的』最好的回答。"
    )

if with_sens and "sensitivity" in results:
    st.subheader("3.6 权重敏感性")
    sens, sm = results["sensitivity"], results["sensitivity_summary"]
    s1, s2, s3 = st.columns(3)
    s1.metric("IC > 0 的权重占比", f"{sm['IC_positive_share']:.0%}")
    s2.metric("IC 中位数", f"{sm['IC_median']:.4f}")
    s3.metric("净夏普 > 0 的权重占比", f"{sm['sharpe_net_positive_share']:.0%}")
    hist = np.histogram(sens["IC_spearman"].dropna(), bins=30)
    st.bar_chart(pd.Series(hist[0], index=np.round(hist[1][:-1], 4)).rename("IC分布"))
    st.caption(f"{len(sens)} 组 Dirichlet 随机权重下的结果分布。"
               "结论若只在少数权重下成立，那就是拟合出来的。")

# ==========================================================================
# 4. 历史分数 vs 实际相对收益
# ==========================================================================
st.header("4 · 分数与实际相对表现")
lookback = st.selectbox("回看区间", ["1年", "3年", "5年", "全样本"], index=1)
days = {"1年": 252, "3年": 756, "5年": 1260, "全样本": len(spread)}[lookback]
sub = spread.iloc[-days:]
cum_rel = (1 + results["relative_return"].reindex(sub.index).fillna(0)).cumprod()
st.line_chart(pd.DataFrame({
    "轮动分差 (z, 左轴概念)": sub,
    f"{LONG_LABEL} 相对 {SHORT_LABEL} 累计净值": cum_rel / cum_rel.iloc[0],
}))
st.caption("肉眼看拐点是否领先——但别把眼球拟合当验证，结论以第3节统计量为准。")

# ==========================================================================
# 5. 实时增强层
# ==========================================================================
st.header("5 · 实时增强层（未进回测）")
st.warning(
    "以下信号**不参与仓位计算**。原因：免费源上情绪与盈利修正拿不到足够长的、"
    "无回填偏差的历史，硬做回测只会得到一个假 IC。它们的定位是对当前打分做"
    "**独立确认或证伪**——若模型看多科技而情绪与盈利修正同时转弱，那就是降低仓位的理由。"
)
if st.button("拉取实时情绪与盈利修正"):
    from src import live_layer
    with st.spinner("抓取新闻/社媒与盈利数据 ..."):
        st.dataframe(live_layer.run_live_layer(), use_container_width=True)

# ==========================================================================
# 6. 方法论与局限
# ==========================================================================
st.header("6 · 方法论与已知局限")
with st.expander("防前视是怎么做的", expanded=False):
    st.markdown(f"""
- **滚动标准化**：z-score 用 {C.SCORING.zscore_window} 交易日滚动窗口，
  分位数用 {C.SCORING.pctile_window} 日窗口，全程只用截止当日的数据。
  绝不使用全样本 min-max —— 那会把未来的分布信息泄漏进历史信号。
- **宏观发布滞后**：每个 FRED 序列按真实发布节奏推迟（DFII10 +1日，
  DTWEXBGS +4日，HY OAS +1日，WALCL +9日），先滞后、再对齐交易日、最后做变换。
- **执行滞后**：`position.shift({C.BACKTEST.execution_lag})`，
  即 T 日收盘出信号、T+1 收盘成交、T+2 起计收益。
- **IC 口径一致**：前瞻收益区间与仓位执行区间严格对齐，不存在时序错配。
""")
with st.expander("这个模型不能做什么", expanded=False):
    st.markdown("""
- **日频调仓 ≠ 日频信息**。快层（动量、资金流）每天真更新；宏观层一周才动一次；
  估值层更慢。所以日频调仓吃的主要是快层，迟滞带的作用就是把慢层的噪音换手挡掉。
- **方向先验是固定的**。“宽松利好成长”这类关系在不同宏观状态下会翻转
  （例如衰退期的降息是坏消息）。当前版本没有做 regime 切换，这是已知局限。
- **信号维度并不完全正交**。动量与资金流代理相关性偏高，实际独立信息源少于信号个数。
- **估值维度是价格拉伸度代理**，不是真实 forward PE。
  放入 `data/pe_history.csv` 后会自动切换成真实PE分位数。
- **回测不含 ETF 冲击成本与借券成本**（多空版），long_only 版更贴近可执行现实。
""")
st.caption("本工具为研究用途，不构成投资建议。")
