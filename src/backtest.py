"""
回测层 —— 这是整个项目的灵魂，也是面试时唯一能扛住追问的部分。

目标不是证明模型多神，而是诚实回答四个问题：
  1. 分差对未来板块相对收益到底有没有预测力？(IC / 胜率)
  2. 预测力在什么状态下最强？(分层测试)
  3. 扣掉换手成本之后还剩多少？(净值曲线 / 换手率)
  4. 结论依赖我拍的那组权重吗？(权重敏感性)

时序对齐约定（execution_lag=2）：
  t 日收盘算出信号 → t+1 日收盘执行 → 从 t+2 日开始赚取收益。
  对应 position.shift(2)。设成 1 就是“用收盘价算完立刻以收盘价成交”的乐观假设。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
import src.scoring as scoring  # noqa: E402

TRADING_DAYS = 252


def spearman(a: pd.Series, b: pd.Series) -> float:
    """
    秩相关 = 对秩做 Pearson。自己实现而不用 pandas 的 method='spearman'，
    因为后者依赖 scipy，减少一个硬依赖。
    """
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    return float(df["a"].rank().corr(df["b"].rank()))


# ==========================================================================
# 收益序列
# ==========================================================================

def leg_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """两条腿的日收益（用各板块 primary 标的）。"""
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    return pd.DataFrame({
        sector: close[meta["primary"]].pct_change()
        for sector, meta in C.SECTORS.items()
    })


def relative_return(legs: pd.DataFrame) -> pd.Series:
    """多头腿 - 空头腿 的日相对收益。"""
    return legs[C.LONG_LEG] - legs[C.SHORT_LEG]


def forward_relative_return(
    rel_ret: pd.Series, horizon: int, exec_lag: int | None = None
) -> pd.Series:
    """
    t 日对应的前瞻相对收益 = [t+exec_lag, t+exec_lag+horizon-1] 区间的累计相对收益。
    与策略端的 position.shift(exec_lag) 严格一致，不存在时序错配。
    """
    exec_lag = C.BACKTEST.execution_lag if exec_lag is None else exec_lag
    return rel_ret.rolling(horizon).sum().shift(-(horizon + exec_lag - 1))


# ==========================================================================
# 1. 信息系数 (IC)
# ==========================================================================

def information_coefficient(
    spread: pd.Series, rel_ret: pd.Series, horizons: tuple[int, ...] | None = None
) -> pd.DataFrame:
    """
    对每个前瞻窗口报告：
      - pearson / spearman IC（全样本，重叠窗口）
      - 非重叠采样下的 IC 与 t 统计量（重叠窗口会严重高估显著性，必须分开报）
      - 方向胜率
    """
    horizons = horizons or C.BACKTEST.ic_horizons
    rows = []
    for h in horizons:
        fwd = forward_relative_return(rel_ret, h)
        df = pd.concat([spread.rename("s"), fwd.rename("f")], axis=1).dropna()
        if len(df) < 100:
            continue

        # 非重叠采样：每 h 天取一个观测，消除重叠窗口造成的自相关
        nonoverlap = df.iloc[::h]
        ic_no = spearman(nonoverlap["s"], nonoverlap["f"])
        n = len(nonoverlap)
        t_stat = ic_no * np.sqrt(max(n - 2, 1) / max(1 - ic_no**2, 1e-9))

        hit = float((np.sign(df["s"]) == np.sign(df["f"])).mean())

        rows.append({
            "horizon": h,
            "IC_pearson": df["s"].corr(df["f"]),
            "IC_spearman": spearman(df["s"], df["f"]),
            "IC_nonoverlap": ic_no,
            "n_nonoverlap": n,
            "t_stat": t_stat,
            "hit_rate": hit,
            "n_obs": len(df),
        })
    return pd.DataFrame(rows).set_index("horizon")


def dimension_ic(
    dim_spread: pd.DataFrame,
    rel_ret: pd.Series,
    horizons: tuple[int, ...] = (5, 20, 60),
) -> pd.DataFrame:
    """
    【最重要的诊断】逐维度拆 IC。

    合成分整体 IC≈0 有两种完全不同的解释：
      (a) 每个维度都没用 → 这个方向到此为止；
      (b) 有维度有用、但被没用的维度稀释掉了 → 该做的是砍掉噪音维度，而不是放弃。

    文献给了明确的先验：行业动量稳健（Moskowitz & Grinblatt 1999），
    宏观周期择时扣成本后消失（Molchanov & Stangl 2024）。
    这张表就是用自己的数据去检验这个先验对不对。
    """
    rows = []
    for dim in dim_spread.columns:
        row = {"dimension": dim}
        for h in horizons:
            fwd = forward_relative_return(rel_ret, h)
            df = pd.concat([dim_spread[dim].rename("s"), fwd.rename("f")],
                           axis=1).dropna()
            if len(df) < 100:
                row[f"IC_{h}d"] = np.nan
                row[f"hit_{h}d"] = np.nan
                continue
            row[f"IC_{h}d"] = spearman(df["s"], df["f"])
            row[f"hit_{h}d"] = float((np.sign(df["s"]) == np.sign(df["f"])).mean())
            if h == horizons[-1]:
                row["n_obs"] = len(df)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("dimension")
    out.index = [C.DIMENSION_LABELS.get(d, d) for d in out.index]
    return out


def single_dimension_strategies(
    dim_spread: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """每个维度**单独**作为信号跑一遍策略，看谁在贡献、谁在拖后腿。"""
    legs = leg_returns(prices)
    rows = []
    for dim in dim_spread.columns:
        pos = scoring.build_positions(dim_spread[dim])
        st = run_strategy(pos, legs)["stats"]
        rows.append({
            "dimension": C.DIMENSION_LABELS.get(dim, dim),
            "sharpe_ls_net": st["long_short_net"].get("sharpe", np.nan),
            "ann_ret_net": st["long_short_net"].get("ann_return", np.nan),
            "maxdd": st["long_short_net"].get("max_drawdown", np.nan),
            "ann_turnover": st["long_short_net"].get("ann_turnover", np.nan),
        })
    return pd.DataFrame(rows).set_index("dimension")


def newey_west_test(
    spread: pd.Series, rel_ret: pd.Series, horizon: int = 20,
    exec_lag: int | None = None,
) -> dict:
    """
    【把"t值虚高"从一句话变成一个数字】

    问题：慢变量信号 + 重叠前瞻窗口 → 残差严重自相关，
    朴素 t 值会把有效样本数当成名义样本数，显著性被系统性夸大。

    做法：对 fwd ~ a + b·spread 做 OLS，用 Newey-West(HAC) 修正标准误，
    滞后阶数取 horizon + exec_lag（重叠窗口造成的自相关跨度）。
    同时用信号自身的一阶自相关 ρ 估计有效样本数：
        N_eff ≈ N × (1-ρ)/(1+ρ)
    ρ 越接近1（越慢的变量），N_eff 缩水越狠。

    报告 t_naive 与 t_nw 的差距本身就是最好的诚实证明。
    """
    exec_lag = C.BACKTEST.execution_lag if exec_lag is None else exec_lag
    fwd = forward_relative_return(rel_ret, horizon, exec_lag)
    df = pd.concat([spread.rename("x"), fwd.rename("y")], axis=1).dropna()
    if len(df) < 200:
        return {}

    x = (df["x"] - df["x"].mean()).to_numpy()
    y = df["y"].to_numpy()
    n = len(x)

    sxx = float((x * x).sum())
    beta = float((x * y).sum() / sxx)
    resid = y - y.mean() - beta * x
    u = x * resid
    se_naive = float(np.sqrt(resid.var(ddof=2) / sxx))

    def _t_at_lag(L: int) -> float:
        """Newey-West: S = γ0 + 2Σ(1 - l/(L+1))γl，作用在 score = x·resid 上。"""
        L = min(L, n - 2)
        s = float((u * u).sum())
        for lag in range(1, L + 1):
            w = 1.0 - lag / (L + 1.0)
            s += 2.0 * w * float((u[lag:] * u[:-lag]).sum())
        se = float(np.sqrt(max(s, 1e-18)) / sxx)
        return beta / se if se > 0 else np.nan

    # 【为什么要报多个滞后阶数】
    # 教科书常用 L = 前瞻窗口长度，但那只盖住了"重叠窗口"造成的自相关。
    # 若信号本身是超慢变量（ρ1≈0.997，自相关半衰期以月计），
    # 残差自相关的跨度远超前瞻窗口，L 取小了就是修正不足、t 值仍然虚高。
    # 与其挑一个自己喜欢的 L，不如把 t 随 L 的衰减曲线整个摆出来，让人自己判断。
    lags = {
        f"L={horizon + exec_lag}(前瞻窗口)": horizon + exec_lag,
        "L=63(季度)": 63,
        "L=126(半年)": 126,
        "L=252(一年)": 252,
    }
    t_by_lag = {name: _t_at_lag(L) for name, L in lags.items()}

    rho = float(pd.Series(x).autocorr(lag=1) or 0.0)
    # 信号自相关的半衰期（交易日）：比 ρ1 本身更好懂
    half_life = float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else np.nan
    # AR(1) 有效样本数。ρ→1 时该公式会机械崩塌给出个位数，
    # 只能当"数量级下界"看，不能当真实自由度。
    n_eff_ar1 = n * (1 - rho) / (1 + rho) if rho < 1 else np.nan
    # 更贴近直觉的独立押注数：信号方向变号次数
    sign = np.sign(spread.dropna().to_numpy())
    flips = int((np.diff(sign) != 0).sum())

    return {
        "beta": beta,
        "t_naive": beta / se_naive if se_naive > 0 else np.nan,
        "t_by_lag": t_by_lag,
        "t_newey_west": t_by_lag[f"L={horizon + exec_lag}(前瞻窗口)"],
        "t_nw_1y": t_by_lag["L=252(一年)"],
        "n_obs": n,
        "rho1_signal": rho,
        "half_life_days": half_life,
        "n_effective_ar1": n_eff_ar1,
        "n_sign_flips": flips,
        "years": n / TRADING_DAYS,
    }


def signal_ic(
    panel: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (20, 60),
    n_periods: int = 3,
) -> pd.DataFrame:
    """
    【最后一道拆解】把维度再拆到**单个信号**。

    要回答的问题：某个维度有效，是四个信号共同贡献，
    还是其实只有一个变量在干活、另外三个搭便车？

    这个区别对定性至关重要：
      - 多信号共同有效 → 这是一个"宏观流动性框架"
      - 只有一个信号有效 → 这其实是"某单一变量择时"，
        依然有价值，但必须换个名字讲，否则就是夸大。
    同时给出逐子样本 IC，检查单信号层面的稳健性。
    """
    rel = relative_return(leg_returns(prices))
    long_p, short_p = panel[C.LONG_LEG], panel[C.SHORT_LEG]
    valid_idx = long_p.dropna(how="all").index
    edges = pd.date_range(valid_idx.min(), valid_idx.max(), periods=n_periods + 1)

    rows = []
    for s in C.SIGNALS:
        if s.key not in long_p.columns or s.key not in short_p.columns:
            continue
        spread = (long_p[s.key] - short_p[s.key]).dropna()
        if spread.empty:
            continue
        row = {"信号": s.name, "维度": C.DIMENSION_LABELS.get(s.dimension, s.dimension)}
        for h in horizons:
            fwd = forward_relative_return(rel, h)
            df = pd.concat([spread.rename("s"), fwd.rename("f")], axis=1).dropna()
            row[f"IC_{h}d"] = spearman(df["s"], df["f"]) if len(df) >= 100 else np.nan
        # 逐子样本，看单信号是不是也跨 regime 稳健
        fwd20 = forward_relative_return(rel, 20)
        for i in range(n_periods):
            lo, hi = edges[i], edges[i + 1]
            m = (spread.index >= lo) & (spread.index < hi)
            df = pd.concat([spread[m].rename("s"), fwd20.rename("f")], axis=1).dropna()
            row[f"P{i + 1}"] = spearman(df["s"], df["f"]) if len(df) >= 100 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("信号")


def subperiod_analysis(
    spread: pd.Series,
    prices: pd.DataFrame,
    n_periods: int = 3,
    horizon: int = 20,
) -> pd.DataFrame:
    """
    【单一 regime 伪装成 alpha 的照妖镜】

    问题背景：2008–2026 只有一个主导宏观叙事——实际利率长期下行、科技碾压消费。
    一个"低实际利率→超配科技"的信号在全样本上必然好看，
    但它可能只是**一次持续十几年的单边押注**，而不是择时能力。

    做法：等分子样本，逐段报告 IC 与净夏普。
    - 若各段符号一致且都为正 → 信号跨 regime 稳健，可信度大增。
    - 若只有某一段极强、其余接近0甚至为负 → 全样本 IC 是被那一段拉起来的，
      这就是伪 alpha，必须如实说明。
    """
    legs = leg_returns(prices)
    rel = relative_return(legs)
    valid = spread.dropna().index
    if len(valid) < n_periods * 252:
        return pd.DataFrame()

    edges = pd.date_range(valid.min(), valid.max(), periods=n_periods + 1)
    fwd = forward_relative_return(rel, horizon)

    rows = []
    for i in range(n_periods):
        lo, hi = edges[i], edges[i + 1]
        mask = (spread.index >= lo) & (spread.index < hi)
        sub_spread = spread[mask]
        df = pd.concat([sub_spread.rename("s"), fwd.rename("f")], axis=1).dropna()
        if len(df) < 100:
            continue
        pos = scoring.build_positions(sub_spread)
        sub_legs = legs.loc[mask]
        res = run_strategy(pos, sub_legs)
        rows.append({
            "period": f"{lo.date()} → {hi.date()}",
            f"IC_{horizon}d": spearman(df["s"], df["f"]),
            "hit_rate": float((np.sign(df["s"]) == np.sign(df["f"])).mean()),
            "sharpe_ls_net": res["stats"]["long_short_net"].get("sharpe", np.nan),
            "rel_drift_ann": float(sub_legs.mean().iloc[0] * TRADING_DAYS
                                   - sub_legs.mean().iloc[1] * TRADING_DAYS),
            "n_obs": len(df),
        })
    return pd.DataFrame(rows).set_index("period")


def dimension_subperiod_ic(
    dim_spread: pd.DataFrame,
    prices: pd.DataFrame,
    n_periods: int = 3,
    horizon: int = 20,
) -> pd.DataFrame:
    """逐维度 × 逐子样本的 IC。哪个维度是真稳健、哪个是靠某一段撑起来的，一眼看穿。"""
    rel = relative_return(leg_returns(prices))
    fwd = forward_relative_return(rel, horizon)
    valid = dim_spread.dropna(how="all").index
    if len(valid) < n_periods * 252:
        return pd.DataFrame()
    edges = pd.date_range(valid.min(), valid.max(), periods=n_periods + 1)

    out = {}
    for dim in dim_spread.columns:
        col = {}
        for i in range(n_periods):
            lo, hi = edges[i], edges[i + 1]
            mask = (dim_spread.index >= lo) & (dim_spread.index < hi)
            df = pd.concat([dim_spread.loc[mask, dim].rename("s"),
                            fwd.rename("f")], axis=1).dropna()
            col[f"{lo.date()}→{hi.date()}"] = (
                spearman(df["s"], df["f"]) if len(df) >= 100 else np.nan
            )
        out[C.DIMENSION_LABELS.get(dim, dim)] = col
    return pd.DataFrame(out).T


def rolling_ic(spread: pd.Series, rel_ret: pd.Series, horizon: int = 20,
               window: int = 252) -> pd.Series:
    """滚动 IC，用来看预测力是否随时间衰减（面试常问：这是不是已经失效了？）"""
    fwd = forward_relative_return(rel_ret, horizon)
    df = pd.concat([spread.rename("s"), fwd.rename("f")], axis=1)
    return df["s"].rolling(window, min_periods=window // 2).corr(df["f"])


# ==========================================================================
# 2. 分层测试
# ==========================================================================

def bucket_analysis(
    spread: pd.Series, rel_ret: pd.Series, horizon: int = 20, n_buckets: int = 5
) -> pd.DataFrame:
    """
    按分差分档，看各档的平均前瞻相对收益与胜率。
    重点看两端：如果只有极端档有效，那这就是个“极端仓位确认器”而非日常择时器。
    """
    fwd = forward_relative_return(rel_ret, horizon)
    df = pd.concat([spread.rename("s"), fwd.rename("f")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    df["bucket"] = pd.qcut(df["s"], n_buckets, labels=False, duplicates="drop")
    g = df.groupby("bucket")
    res = pd.DataFrame({
        "spread_min": g["s"].min(),
        "spread_max": g["s"].max(),
        "mean_fwd_ret": g["f"].mean(),
        "median_fwd_ret": g["f"].median(),
        "hit_rate_positive": g["f"].apply(lambda x: float((x > 0).mean())),
        "n_obs": g.size(),
    })
    res.index.name = f"bucket(低→高), horizon={horizon}d"
    return res


# ==========================================================================
# 3. 策略回测（含成本）
# ==========================================================================

def performance_stats(ret: pd.Series, positions: pd.Series | None = None) -> dict:
    r = ret.dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    ann_ret = eq.iloc[-1] ** (TRADING_DAYS / len(r)) - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    dd = (eq / eq.cummax() - 1).min()
    stats = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "max_drawdown": dd,
        "hit_rate_daily": float((r > 0).mean()),
        "n_days": len(r),
    }
    if positions is not None:
        turn = positions.diff().abs().dropna()
        stats["ann_turnover"] = float(turn.sum() / len(turn) * TRADING_DAYS)
    return stats


def run_strategy(
    positions: pd.Series,
    legs: pd.DataFrame,
    exec_lag: int | None = None,
    cost_bps: float | None = None,
    long_only_tilt: bool = True,
) -> dict:
    """
    两个版本一起跑：
      - long_short: 纯多空相对策略，最干净地衡量信号本身的价值
      - long_only : 50/50 基准上做倾斜（公募QDII实际可执行的形态），
                    与静态 50/50 基准对比看超额

    成本：换手一个单位分差需要同时调整两条腿，故成本 = turnover × bps × 2。
    """
    exec_lag = C.BACKTEST.execution_lag if exec_lag is None else exec_lag
    cost_bps = C.BACKTEST.cost_bps if cost_bps is None else cost_bps

    rel = relative_return(legs)
    pos = positions.shift(exec_lag)
    turnover = pos.diff().abs().fillna(0.0)
    cost = turnover * (cost_bps / 1e4) * 2

    ls_gross = pos * rel
    ls_net = ls_gross - cost

    # 多头倾斜版：w_long = 0.5 + 0.5×pos ∈ [0,1]
    w_long = (0.5 + 0.5 * pos).clip(0, 1)
    lo_gross = w_long * legs[C.LONG_LEG] + (1 - w_long) * legs[C.SHORT_LEG]
    lo_net = lo_gross - cost * 0.5   # 倾斜幅度是多空版的一半
    bench = 0.5 * legs[C.LONG_LEG] + 0.5 * legs[C.SHORT_LEG]

    # 诚实对照组：这段样本里科技相对消费本身就有很强的结构性上行漂移。
    # 只跟 50/50 比会让"常年偏多科技"的 beta 冒充 alpha，
    # 所以必须把"什么都不做、一直满仓多头腿"也摆出来。
    static_long = legs[C.LONG_LEG]

    return {
        "returns": pd.DataFrame({
            "long_short_gross": ls_gross,
            "long_short_net": ls_net,
            "long_only_net": lo_net,
            "benchmark_5050": bench,
            "static_long_leg": static_long,
            "excess_vs_bench": lo_net - bench,
            "excess_vs_static_long": lo_net - static_long,
        }),
        "positions_executed": pos,
        "turnover": turnover,
        "stats": {
            "long_short_gross": performance_stats(ls_gross, pos),
            "long_short_net": performance_stats(ls_net, pos),
            "long_only_net": performance_stats(lo_net, pos),
            "benchmark_5050": performance_stats(bench),
            "static_long_leg": performance_stats(static_long),
            "excess_vs_bench": performance_stats(lo_net - bench, pos),
            "excess_vs_static_long": performance_stats(lo_net - static_long, pos),
        },
    }


# ==========================================================================
# 4. 权重敏感性
# ==========================================================================

def weight_sensitivity(
    dim_scores: dict[str, pd.DataFrame],
    legs: pd.DataFrame,
    n_draws: int | None = None,
    horizon: int = 20,
) -> pd.DataFrame:
    """
    从 Dirichlet 分布随机抽取维度权重，重跑打分与回测。
    如果结论（IC 符号、净夏普）在绝大多数权重下都成立，
    就证明它不是某一组精心挑选的权重堆出来的。
    """
    n_draws = n_draws or C.BACKTEST.n_weight_draws
    rng = np.random.default_rng(C.BACKTEST.random_seed)
    rel = relative_return(legs)
    dims = C.DIMENSIONS

    rows = []
    for _ in range(n_draws):
        w = rng.dirichlet(np.ones(len(dims)))
        weights = dict(zip(dims, w))
        comp = scoring.composite_scores(dim_scores, weights)
        spread = scoring.rotation_spread(comp)
        pos = scoring.build_positions(spread)

        fwd = forward_relative_return(rel, horizon)
        df = pd.concat([spread.rename("s"), fwd.rename("f")], axis=1).dropna()
        ic = spearman(df["s"], df["f"]) if len(df) > 100 else np.nan

        res = run_strategy(pos, legs)
        rows.append({
            **{f"w_{d}": weights[d] for d in dims},
            "IC_spearman": ic,
            "sharpe_net": res["stats"]["long_short_net"].get("sharpe", np.nan),
            "excess_sharpe": res["stats"]["excess_vs_bench"].get("sharpe", np.nan),
        })
    return pd.DataFrame(rows)


def regime_conditional_ic(
    panel: dict[str, pd.DataFrame],
    regime: pd.Series,
    prices: pd.DataFrame,
    horizon: int = 20,
) -> pd.DataFrame:
    """
    【检验固定方向先验的假设是否成立】

    对每个信号，分别在 risk-on / risk-off 下算 IC。
    - 若两态 IC 接近 → 固定先验没问题，regime 切换是多余的复杂度；
    - 若某信号在两态下**符号相反或幅度悬殊** → 固定先验确实丢了信息，
      regime 切换有依据。

    注意这是**诊断**，不是拟合：看到差异不等于该按差异去调参数，
    因为 risk-off 样本量小、容易是噪音。诊断结果只用来判断
    事先声明的那套乘数方向对不对。
    """
    rel = relative_return(leg_returns(prices))
    fwd = forward_relative_return(rel, horizon)
    long_p, short_p = panel[C.LONG_LEG], panel[C.SHORT_LEG]
    reg = regime.reindex(long_p.index).ffill()

    rows = []
    for s in C.SIGNALS:
        if s.key not in long_p.columns:
            continue
        spread = (long_p[s.key] - short_p[s.key])
        row = {"信号": s.name, "维度": C.DIMENSION_LABELS.get(s.dimension, s.dimension)}
        for state in ("risk_on", "risk_off"):
            m = reg == state
            df = pd.concat([spread[m].rename("s"), fwd.rename("f")], axis=1).dropna()
            row[f"IC_{state}"] = spearman(df["s"], df["f"]) if len(df) >= 100 else np.nan
            row[f"n_{state}"] = len(df)
        row["预设risk_off乘数"] = C.REGIME_MULTIPLIERS.get(s.key, 1.0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("信号")


def compare_regime_variants(
    prices: pd.DataFrame,
    macro_raw: pd.DataFrame,
    regime: pd.Series,
    pe_df: pd.DataFrame | None = None,
    horizon: int = 20,
) -> pd.DataFrame:
    """
    三个变体同口径对比，决定 regime 切换该不该进最终模型：
      baseline       固定方向先验（当前模型）
      regime_priors  方向先验按 risk-on/off 切换
      regime_scaled  方向先验不变，但 risk-off 时整体降敞口

    判定标准要事先说清楚：改善必须**同时**体现在 IC 与扣成本夏普上，
    且幅度要明显超过噪音水平，否则按奥卡姆剃刀保留基线。
    """
    import src.regime as rg  # noqa: PLC0415
    import src.signals as sg  # noqa: PLC0415

    legs = leg_returns(prices)
    rel = relative_return(legs)
    fwd = forward_relative_return(rel, horizon)
    reg = regime.reindex(prices["date"].unique()).ffill()

    def _row(name: str, scores: dict, pos: pd.Series) -> dict:
        res = run_strategy(pos, legs)
        df = pd.concat([scores["spread"].rename("s"), fwd.rename("f")],
                       axis=1).dropna()
        return {
            "variant": name,
            f"IC_{horizon}d": spearman(df["s"], df["f"]),
            "hit_rate": float((np.sign(df["s"]) == np.sign(df["f"])).mean()),
            "sharpe_net": res["stats"]["long_short_net"].get("sharpe", np.nan),
            "IR_vs_bench": res["stats"]["excess_vs_bench"].get("sharpe", np.nan),
            "maxdd": res["stats"]["long_short_net"].get("max_drawdown", np.nan),
            "turnover": res["stats"]["long_short_net"].get("ann_turnover", np.nan),
        }

    rows = []
    base_panel = sg.build_signal_panel(prices, macro_raw, pe_df)
    base = scoring.run_scoring(base_panel, rel_ret=rel)
    rows.append(_row("baseline", base, base["positions"]))

    reg_panel = sg.build_signal_panel(prices, macro_raw, pe_df, regime=regime)
    reg_scores = scoring.run_scoring(reg_panel, rel_ret=rel)
    rows.append(_row("regime_priors", reg_scores, reg_scores["positions"]))

    # 降敞口变体：信号不变，只在 risk-off 时把仓位按比例缩小
    scale = pd.Series(1.0, index=base["positions"].index)
    r = regime.reindex(scale.index).ffill().fillna("risk_on")
    scale[r == "risk_off"] = 0.5
    rows.append(_row("regime_scaled(0.5x)", base, base["positions"] * scale))

    return pd.DataFrame(rows).set_index("variant")


def compare_weight_schemes(
    panel: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    schemes: tuple[str, ...] = ("equal", "inverse_vol", "ic_weighted"),
    horizon: int = 20,
) -> pd.DataFrame:
    """
    三种权重方案横向对比。

    这张表是面试时回答"你的权重怎么定的"最有力的答案：
    不是我定的，是三种方案摆在这里、用同一套口径比出来的。
    如果 IC 加权只比等权好一点点，那就老实说等权已经够用——
    学术界的普遍结论也是：等权是极难被稳定打败的基线。
    """
    legs = leg_returns(prices)
    rel = relative_return(legs)
    rows = []
    for scheme in schemes:
        sc = scoring.run_scoring(panel, scheme=scheme, rel_ret=rel)
        res = run_strategy(sc["positions"], legs)
        fwd = forward_relative_return(rel, horizon)
        df = pd.concat([sc["spread"].rename("s"), fwd.rename("f")], axis=1).dropna()
        st_ls = res["stats"]["long_short_net"]
        st_ex = res["stats"]["excess_vs_bench"]
        rows.append({
            "scheme": scheme,
            f"IC_{horizon}d": spearman(df["s"], df["f"]),
            "hit_rate": float((np.sign(df["s"]) == np.sign(df["f"])).mean()),
            "sharpe_ls_net": st_ls.get("sharpe", np.nan),
            "ann_ret_ls_net": st_ls.get("ann_return", np.nan),
            "maxdd_ls_net": st_ls.get("max_drawdown", np.nan),
            "sharpe_excess": st_ex.get("sharpe", np.nan),
            "ann_turnover": st_ls.get("ann_turnover", np.nan),
        })
    return pd.DataFrame(rows).set_index("scheme")


def summarize_sensitivity(sens: pd.DataFrame) -> dict:
    return {
        "IC_positive_share": float((sens["IC_spearman"] > 0).mean()),
        "IC_median": float(sens["IC_spearman"].median()),
        "IC_p05": float(sens["IC_spearman"].quantile(0.05)),
        "IC_p95": float(sens["IC_spearman"].quantile(0.95)),
        "sharpe_net_median": float(sens["sharpe_net"].median()),
        "sharpe_net_positive_share": float((sens["sharpe_net"] > 0).mean()),
    }


# ==========================================================================
# 5. 一站式
# ==========================================================================

def run_full_backtest(scores: dict, prices: pd.DataFrame,
                      run_sensitivity: bool = True,
                      panel: dict[str, pd.DataFrame] | None = None,
                      compare_schemes_flag: bool = True) -> dict:
    legs = leg_returns(prices)
    rel = relative_return(legs)
    spread = scores["spread"]

    out = {
        "ic": information_coefficient(spread, rel),
        "dimension_ic": dimension_ic(scores["dimension_spread"], rel),
        "dimension_strategies": single_dimension_strategies(
            scores["dimension_spread"], prices),
        "nw_test": {h: newey_west_test(spread, rel, h)
                    for h in C.BACKTEST.ic_horizons},
        "subperiods": subperiod_analysis(spread, prices),
        "dimension_subperiod_ic": dimension_subperiod_ic(
            scores["dimension_spread"], prices),
        "rolling_ic_20d": rolling_ic(spread, rel, horizon=20),
        "buckets": {h: bucket_analysis(spread, rel, horizon=h)
                    for h in C.BACKTEST.ic_horizons},
        "strategy": run_strategy(scores["positions"], legs),
        "legs": legs,
        "relative_return": rel,
    }
    if panel is not None:
        out["signal_ic"] = signal_ic(panel, prices)
        if compare_schemes_flag:
            out["scheme_comparison"] = compare_weight_schemes(panel, prices)
    if run_sensitivity:
        sens = weight_sensitivity(scores["dimension_scores"], legs)
        out["sensitivity"] = sens
        out["sensitivity_summary"] = summarize_sensitivity(sens)
    return out
