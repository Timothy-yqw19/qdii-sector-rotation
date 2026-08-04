"""
防前视自查 —— 这套测试本身就是面试材料。

面试官问“你怎么保证没有前视偏差？”，最好的答案不是“我很小心”，
而是“我写了截断不变性测试：把数据截断到任意历史日期重跑，
截断前的信号值必须逐点相同。如果有任何一处用了未来信息，这个测试必然失败。”

直接跑：  python tests/test_no_lookahead.py
或 pytest： pytest tests/ -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as C  # noqa: E402
from src import backtest as bt  # noqa: E402
from src import data_fetcher as dfetch  # noqa: E402
from src import scoring  # noqa: E402
from src import signals as sig  # noqa: E402

PRICES, MACRO = dfetch.make_synthetic_data(start="2012-01-01", end="2024-12-31", seed=11)


# ==========================================================================
def test_truncation_invariance():
    """
    【核心测试】把数据截断到 2021-06-30 重跑整条信号链，
    截断日之前的每一个信号值都必须与全样本结果逐点相同。
    任何形式的前视（全样本标准化、未来数据 ffill、错误 shift）都会让它失败。
    """
    cut = pd.Timestamp("2021-06-30")
    full = sig.build_signal_panel(PRICES, MACRO)
    p_cut = PRICES[PRICES["date"] <= cut]
    m_cut = MACRO[MACRO.index <= cut]
    trunc = sig.build_signal_panel(p_cut, m_cut)

    for sector in C.SECTORS:
        a = full[sector].loc[:cut]
        b = trunc[sector].loc[:cut]
        common = a.index.intersection(b.index)
        for col in a.columns:
            x, y = a.loc[common, col], b.loc[common, col]
            both = x.notna() & y.notna()
            assert (x.isna() == y.isna()).all(), f"{sector}/{col} 缺失位置不一致"
            assert np.allclose(x[both], y[both], atol=1e-10), \
                f"{sector}/{col} 截断前后不一致 → 存在前视偏差"
    print("✓ 截断不变性：信号链未使用未来信息")


# ==========================================================================
def test_rolling_zscore_is_causal():
    """篡改未来的值，不能改变过去的 z-score。"""
    s = pd.Series(np.random.default_rng(0).normal(size=1500),
                  index=pd.bdate_range("2015-01-01", periods=1500))
    base = sig.rolling_zscore(s)
    tampered = s.copy()
    tampered.iloc[1200:] += 100.0          # 未来注入巨大冲击
    after = sig.rolling_zscore(tampered)
    assert np.allclose(base.iloc[:1200].dropna(), after.iloc[:1200].dropna(), atol=1e-12)
    print("✓ 滚动 z-score 严格因果")


def test_rolling_percentile_is_causal():
    s = pd.Series(np.random.default_rng(1).normal(size=1500).cumsum(),
                  index=pd.bdate_range("2015-01-01", periods=1500))
    base = sig.rolling_percentile(s)
    tampered = s.copy()
    tampered.iloc[1000:] += 500.0
    after = sig.rolling_percentile(tampered)
    assert np.allclose(base.iloc[:1000].dropna(), after.iloc[:1000].dropna(), atol=1e-12)
    print("✓ 滚动分位数严格因果")


# ==========================================================================
def test_macro_publication_lag():
    """
    在某个观测日打一个脉冲，检查它最早出现在信号里的日期 ≥ 观测日 + 发布滞后。
    这一条专防“用了发布日当天数据”这种隐蔽错误。
    """
    trading_index = PRICES.pivot(index="date", columns="ticker", values="close").index
    spike_date = pd.Timestamp("2019-03-06")

    for ms in C.MACRO_SERIES:
        # 用 level 变换做脉冲测试最干净（差分会把脉冲扩散到整个窗口，
        # 掩盖“最早出现日”这个我们真正要检验的量）。滞后逻辑与变换无关。
        probe = C.MacroSeries(ms.fred_id, ms.name, ms.publication_lag_days, "level")
        col = f"{ms.fred_id}_level"

        m = MACRO.copy()
        # 用**阶跃**而非单点脉冲：单点脉冲在“日频序列 + 日历日滞后”下
        # 可能正好落到周末而被下一个观测覆盖，导致测试自身失效。
        # 阶跃一定会传导，且首次变化日仍必须 ≥ 起始观测日 + 发布滞后。
        obs_dates = MACRO[ms.fred_id].dropna().index
        sd = obs_dates[obs_dates >= spike_date][0]
        m.loc[m.index >= sd, ms.fred_id] = (
            m.loc[m.index >= sd, ms.fred_id] + float(np.nanstd(m[ms.fred_id])) * 10
        )

        base = sig.build_macro_signals(MACRO, trading_index, series=[probe])[col]
        out = sig.build_macro_signals(m, trading_index, series=[probe])[col]
        diff = (out - base).abs()
        moved = diff[diff > 1e-9]
        assert not moved.empty, f"{ms.fred_id} 脉冲完全没有传导，测试无效"
        first = moved.index.min()
        assert first >= sd + pd.Timedelta(days=ms.publication_lag_days), \
            f"{ms.fred_id} 脉冲提前泄漏：{first.date()} 早于 " \
            f"{sd.date()} + {ms.publication_lag_days}d"

    # 通用检查：所有宏观信号在任意日期都不早于其观测日 + lag 才变化
    for ms in C.MACRO_SERIES:
        raw = MACRO[ms.fred_id].dropna()
        avail = raw.copy()
        avail.index = avail.index + pd.Timedelta(days=ms.publication_lag_days)
        assert (avail.index > raw.index).all(), f"{ms.fred_id} 未施加发布滞后"
    print("✓ 宏观数据发布滞后已正确施加")


# ==========================================================================
def test_forward_return_alignment():
    """
    手工核对前瞻收益的区间：t 日信号对应 [t+exec_lag, t+exec_lag+horizon-1]。
    与策略端 position.shift(exec_lag) 必须严格一致。
    """
    idx = pd.bdate_range("2020-01-01", periods=40)
    r = pd.Series(np.arange(40, dtype=float), index=idx)  # 用序号便于人工验算
    h, lag = 5, 2
    fwd = bt.forward_relative_return(r, horizon=h, exec_lag=lag)
    t = 10
    expected = r.iloc[t + lag: t + lag + h].sum()
    assert abs(fwd.iloc[t] - expected) < 1e-9, \
        f"前瞻区间错位: got {fwd.iloc[t]}, expect {expected}"

    # 策略端一致性：pos.shift(lag) 在 t+lag 处等于 pos 在 t 处
    pos = pd.Series(np.arange(40, dtype=float), index=idx)
    assert pos.shift(lag).iloc[t + lag] == pos.iloc[t]
    print("✓ 前瞻收益与执行滞后口径一致")


# ==========================================================================
def test_shuffled_signal_has_no_edge():
    """
    把信号随机打乱，IC 应当落在 0 附近。
    如果打乱后仍有显著 IC，说明代码里存在结构性泄漏。
    """
    panel = sig.build_signal_panel(PRICES, MACRO)
    scores = scoring.run_scoring(panel)
    legs = bt.leg_returns(PRICES)
    rel = bt.relative_return(legs)

    rng = np.random.default_rng(3)
    ics = []
    for _ in range(30):
        shuffled = pd.Series(
            rng.permutation(scores["spread"].dropna().to_numpy()),
            index=scores["spread"].dropna().index,
        )
        fwd = bt.forward_relative_return(rel, 20)
        ics.append(bt.spearman(shuffled, fwd))
    mean_ic = float(np.nanmean(ics))
    assert abs(mean_ic) < 0.05, f"打乱后仍有 IC={mean_ic:.4f}，存在结构性泄漏"
    print(f"✓ 随机打乱后 IC≈{mean_ic:+.4f}（无结构性泄漏）")


# ==========================================================================
def test_perfect_foresight_sanity():
    """
    反向对照：用**真实的未来收益**当信号，策略必须赚大钱。
    这条不是检查正确性，而是检查回测引擎本身没写反——
    如果连完美预知都不赚钱，说明 shift 方向搞错了。
    """
    legs = bt.leg_returns(PRICES)
    rel = bt.relative_return(legs)
    lag = C.BACKTEST.execution_lag
    cheat = np.sign(rel.shift(-lag)).fillna(0.0)   # 故意的前视，仅用于自检
    res = bt.run_strategy(cheat, legs, cost_bps=0.0)
    sharpe = res["stats"]["long_short_gross"]["sharpe"]
    assert sharpe > 3, f"完美预知信号夏普仅 {sharpe:.2f}，回测引擎时序可能写反"
    print(f"✓ 完美预知对照组夏普 {sharpe:.1f}（回测引擎时序正确）")


# ==========================================================================
def test_weight_schemes_truncation_invariance():
    """
    【IC加权最危险的地方】权重本身是用"信号 vs 未来收益"估出来的，
    稍有不慎就会拿还没实现的收益去定今天的权重。
    这里对三种方案都做截断不变性：截断日之前的权重必须逐点相同。
    """
    cut = pd.Timestamp("2021-06-30")
    panel_full = sig.build_signal_panel(PRICES, MACRO)
    dims_full = scoring.dimension_scores(panel_full)
    rel_full = bt.relative_return(bt.leg_returns(PRICES))

    p_cut = PRICES[PRICES["date"] <= cut]
    m_cut = MACRO[MACRO.index <= cut]
    dims_cut = scoring.dimension_scores(sig.build_signal_panel(p_cut, m_cut))
    rel_cut = bt.relative_return(bt.leg_returns(p_cut))

    for scheme in ("equal", "inverse_vol", "ic_weighted"):
        a = scoring.resolve_weights(dims_full, rel_ret=rel_full, scheme=scheme).loc[:cut]
        b = scoring.resolve_weights(dims_cut, rel_ret=rel_cut, scheme=scheme).loc[:cut]
        common = a.index.intersection(b.index)
        for col in a.columns:
            x, y = a.loc[common, col], b.loc[common, col]
            both = x.notna() & y.notna()
            assert both.sum() > 100, f"{scheme}/{col} 有效样本太少，测试无意义"
            assert np.allclose(x[both], y[both], atol=1e-10), \
                f"权重方案 {scheme} 的 {col} 截断前后不一致 → 权重估计偷看了未来"
    print("✓ 三种权重方案均通过截断不变性（IC加权无未来数据泄漏）")


def test_ic_weight_realization_lag():
    """
    直接验证 IC 加权的滞后量：t 日权重所依赖的最新一对样本，
    其前瞻收益必须在 t 日之前就已完全实现。
    做法：只篡改 cut 之后的收益，cut 之前的权重必须纹丝不动。
    """
    cut = pd.Timestamp("2021-06-30")
    panel = sig.build_signal_panel(PRICES, MACRO)
    dims = scoring.dimension_scores(panel)
    rel = bt.relative_return(bt.leg_returns(PRICES))

    tampered = rel.copy()
    tampered[tampered.index > cut] += 0.05      # 未来收益整体加5%/天

    base = scoring.ic_weights(scoring.dimension_spread(dims), rel)
    after = scoring.ic_weights(scoring.dimension_spread(dims), tampered)

    lag = C.SCORING.ic_weight_horizon + C.BACKTEST.execution_lag - 1
    safe = base.index[base.index <= cut]
    x, y = base.loc[safe], after.loc[safe]
    both = x.notna() & y.notna()
    assert np.allclose(x[both], y[both], atol=1e-10), \
        f"篡改 {cut.date()} 之后的收益改变了之前的权重 → IC加权存在前视（应滞后{lag}日）"
    print(f"✓ IC加权滞后 {lag} 个交易日，只用已实现收益")


def test_inverse_vol_equalizes_risk():
    """逆波动加权的目的是让各维度风险贡献接近，这里验证它确实压缩了风险离散度。"""
    panel = sig.build_signal_panel(PRICES, MACRO)
    dims = scoring.dimension_scores(panel)
    spread = scoring.dimension_spread(dims).dropna()
    w = scoring.inverse_vol_weights(spread).reindex(spread.index).ffill()

    eq = spread.mul(1.0 / spread.shape[1])
    iv = spread.mul(w)
    # 各维度贡献的标准差离散程度（越小越接近风险平价）
    disp_eq = eq.std().std()
    disp_iv = iv.std().std()
    assert disp_iv < disp_eq, f"逆波动未降低风险离散度: {disp_iv:.4f} !< {disp_eq:.4f}"
    print(f"✓ 逆波动加权把维度风险贡献离散度从 {disp_eq:.4f} 压到 {disp_iv:.4f}")


# ==========================================================================
def test_hysteresis_reduces_turnover():
    panel = sig.build_signal_panel(PRICES, MACRO)
    scores = scoring.run_scoring(panel)
    legs = bt.leg_returns(PRICES)
    t0 = bt.run_strategy(scoring.build_positions(scores["spread"], band=0.0), legs)
    t1 = bt.run_strategy(scoring.build_positions(scores["spread"], band=0.3), legs)
    a = t0["stats"]["long_short_net"]["ann_turnover"]
    b = t1["stats"]["long_short_net"]["ann_turnover"]
    assert b < a, f"迟滞带未降低换手: {b:.2f} !< {a:.2f}"
    print(f"✓ 迟滞带把年换手从 {a:.1f} 降到 {b:.1f}")


# ==========================================================================
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"运行 {len(tests)} 项防前视与一致性检查 ...\n")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"✗ {t.__name__}: {e}")
    print(f"\n{'全部通过' if failed == 0 else f'{failed} 项失败'}")
    sys.exit(1 if failed else 0)
