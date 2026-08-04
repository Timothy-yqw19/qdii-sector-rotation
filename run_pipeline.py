"""
端到端跑一遍：数据 → 信号 → 打分 → 回测 → 落盘 + 终端报告。

用法：
    python run_pipeline.py              # 用 data/ 下缓存
    python run_pipeline.py --fetch      # 先联网下载再跑
    python run_pipeline.py --offline    # 无网络时用合成数据验证逻辑
    python run_pipeline.py --no-sens    # 跳过权重敏感性（快）
"""

from __future__ import annotations

import os
import sys

import warnings

import pandas as pd

# 全 NaN 切片做 reduce 时 numpy 会抛 RuntimeWarning，这在信号预热期（前252天
# z-score 还没出值）是预期行为，不是错误。只屏蔽这一条，不做全局静音。
warnings.filterwarnings("ignore", message="invalid value encountered in reduce",
                        category=RuntimeWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config as C  # noqa: E402
from src import backtest as bt  # noqa: E402
from src import data_fetcher as dfetch  # noqa: E402
from src import scoring  # noqa: E402
from src import signals as sig  # noqa: E402


def load_data(argv: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if "--offline" in argv:
        print("[数据] 离线模式：合成数据（仅验证逻辑，不可用于投资结论）")
        prices, macro = dfetch.make_synthetic_data()
        meta = {"price_source": "synthetic", "dividend_adjusted": True}
        dfetch.save_cache(prices, macro, root=ROOT, meta=meta)
        return prices, macro, meta
    if "--fetch" in argv:
        pause = 3.0 if "--slow" in argv else 0.5
        print("[数据] 联网下载中（多源回退：Tiingo → Stooq → yfinance）...")
        # 先抓价格：失败会直接抛错，此时缓存尚未被触碰，旧数据安全
        prices, meta = dfetch.fetch_prices(pause=pause)
        # 抓到就立刻落盘。宏观那步再挂也不会白抓一遍价格。
        dfetch.save_prices(prices, root=ROOT, meta=meta)

        try:
            macro = dfetch.fetch_macro()
            dfetch.save_macro(macro, root=ROOT)
        except RuntimeError as e:
            cached = dfetch.load_macro_cache(root=ROOT)
            if cached is None:
                raise
            last = cached.index.max()
            stale = (pd.Timestamp.today().normalize() - last).days
            print(f"\n  [降级] FRED 本次不可用，改用已有宏观缓存"
                  f"（最新 {last.date()}，距今 {stale} 天）。")
            print("  宏观数据本身变化很慢，沿用几天旧值对结论影响有限；"
                  "但若超过两周，实际利率/信用利差维度会失真，务必重抓。")
            print(f"  原始错误：{str(e).splitlines()[0]}")
            macro = cached
        return prices, macro, meta
    print("[数据] 读取本地缓存 ...")
    prices, macro = dfetch.load_cache(root=ROOT)
    dfetch.assert_cache_usable(prices)
    return prices, macro, dfetch.load_meta(root=ROOT)


def fmt_pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x:+.2%}"


def main() -> None:
    argv = sys.argv[1:]
    try:
        prices, macro, meta = load_data(argv)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\n[中止] {e}")
        sys.exit(1)
    pe = dfetch.load_optional_pe(root=ROOT)

    def _arg(name, default=None):
        return next((a.split("=", 1)[1] for a in argv if a.startswith(f"--{name}=")),
                    default)

    scheme = _arg("scheme")
    rebalance = _arg("rebalance")
    dims_arg = _arg("dims")
    active_dims = [d.strip() for d in dims_arg.split(",")] if dims_arg else None

    print("\n[信号] 构建信号面板（滚动z-score + 发布滞后对齐）...")
    panel = sig.build_signal_panel(prices, macro, pe)

    print(f"[打分] 维度分 → 合成分 → 轮动分差（权重方案: "
          f"{scheme or C.SCORING.weight_scheme}）...")
    legs = bt.leg_returns(prices)
    scores = scoring.run_scoring(panel, scheme=scheme,
                                 rel_ret=bt.relative_return(legs),
                                 active_dims=active_dims, rebalance=rebalance)

    print("[回测] IC / 分层 / 净值 / 权重方案对比 / 权重敏感性 ...")
    results = bt.run_full_backtest(
        scores, prices,
        run_sensitivity="--no-sens" not in argv,
        panel=panel,
        compare_schemes_flag="--no-compare" not in argv,
    )

    # ---------------- 终端报告 ----------------
    sep = "=" * 72
    print(f"\n{sep}\n全球宏观-板块轮动打分系统 · 回测报告\n{sep}")
    print(f"样本区间   : {scores['spread'].dropna().index.min().date()} "
          f"→ {scores['spread'].dropna().index.max().date()}")
    print(f"轮动方向   : 多 {C.SECTORS[C.LONG_LEG]['label']}"
          f"({C.SECTORS[C.LONG_LEG]['primary']}) / "
          f"空 {C.SECTORS[C.SHORT_LEG]['label']}({C.SECTORS[C.SHORT_LEG]['primary']})")
    print(f"执行假设   : 信号T日收盘生成 → T+1收盘执行 → T+2起计收益 "
          f"(execution_lag={C.BACKTEST.execution_lag})")
    print(f"交易成本   : {C.BACKTEST.cost_bps:.0f} bps/腿, 迟滞带 "
          f"{C.BACKTEST.hysteresis_band}")
    print(f"价格数据源 : {meta.get('price_source', '未知')}"
          f"（{'已分红复权' if meta.get('dividend_adjusted') else '未做分红复权'}）")
    print(f"权重方案   : {scores['scheme']}｜启用维度: "
          f"{'/'.join(C.DIMENSION_LABELS.get(d, d) for d in scores['active_dims'])}"
          f"｜调仓频率: {scores['rebalance']}")
    note = dfetch.dividend_bias_note(meta)
    if note:
        print(f"\n[!] {note}")

    print(f"\n--- 0. 宏观数据覆盖体检 ---")
    cov = dfetch.report_macro_coverage(macro)
    if not cov.empty:
        print(cov.to_string(index=False))
        if (cov["flag"] != "").any():
            print("  [!] 带标记的序列疑似被数据源静默截断，"
                  "会让该信号在早期子样本全是NaN。重抓或换取数路径后再看诊断表。")

    print(f"\n--- 1. 信息系数 (分差 → 未来相对收益) ---")
    print(results["ic"].round(4).to_string())

    nw = results.get("nw_test", {})
    if nw:
        print("\n  显著性修正（Newey-West HAC，t 随滞后阶数的衰减）：")
        for h, d in nw.items():
            if not d:
                continue
            chain = "  →  ".join(f"{k} {v:.2f}" for k, v in d["t_by_lag"].items())
            print(f"    {h}日前瞻: t_naive={d['t_naive']:.2f}  →  {chain}")
        d0 = next(v for v in nw.values() if v)
        print(f"\n    信号持续性: ρ1={d0['rho1_signal']:.3f}, "
              f"自相关半衰期≈{d0['half_life_days']:.0f} 交易日"
              f"（约 {d0['half_life_days'] / 21:.1f} 个月）")
        print(f"    样本: {d0['years']:.1f} 年 / {d0['n_obs']} 个交易日 / "
              f"信号方向变号 {d0['n_sign_flips']} 次 "
              f"(≈ 每年 {d0['n_sign_flips'] / d0['years']:.1f} 次独立押注)")
        print(f"    AR(1)有效样本数 N_eff≈{d0['n_effective_ar1']:.0f} "
              f"——ρ→1 时该公式会机械崩塌，只能当数量级下界，别当真实自由度")
        print("  解读：该报的是 t 随 L 增大后的稳定值，不是 t_naive。"
              "若 L 拉到一年 t 仍 >2，说明结果扛得住自相关修正；"
              "若掉到 1 附近，就老实说『统计上不显著，只是方向性证据』。")

    print(f"\n--- 1.5 逐维度 IC 拆解（合成分为什么是这个结果）---")
    print(results["dimension_ic"].round(4).to_string())
    print("\n各维度单独作为信号的策略表现：")
    print(results["dimension_strategies"].round(4).to_string())
    print("解读：若某维度 IC 显著为正而合成分接近0，说明它被噪音维度稀释了，"
          "该做的是砍维度而不是放弃框架。")

    if "signal_ic" in results:
        print(f"\n--- 1.55 单信号 IC（维度是集体有效，还是一个变量在扛）---")
        print(results["signal_ic"].round(4).to_string())
        print("P1/P2/P3 为三个等分子样本的 IC(20d)。"
              "若某维度只有一个信号 IC 显著、其余接近0，"
              "那这个模型的真实身份是『该单一变量择时』，讲的时候必须改名。")

    print(f"\n--- 1.6 子样本稳健性【单一regime照妖镜】---")
    sub = results.get("subperiods")
    if sub is not None and not sub.empty:
        print(sub.round(4).to_string())
    dsub = results.get("dimension_subperiod_ic")
    if dsub is not None and not dsub.empty:
        print("\n逐维度 × 逐子样本 IC(20d):")
        print(dsub.round(4).to_string())
    print("解读：全样本IC为正但只有某一段极强 → 那是单一regime的单边押注，不是择时能力。"
          "各段符号一致才算稳健。rel_drift_ann 是该段两板块的真实相对漂移，"
          "用来判断信号是不是只在顺风段有效。")

    print(f"\n--- 2. 分层测试 (horizon=20d) ---")
    print(results["buckets"][20].round(4).to_string())

    print(f"\n--- 3. 策略表现 ---")
    st = pd.DataFrame(results["strategy"]["stats"]).T
    cols = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "ann_turnover"]
    print(st[[c for c in cols if c in st.columns]].round(3).to_string())
    print("注意 excess_vs_static_long：这段样本里科技相对消费本身有很强的结构性漂移，"
          "只跟50/50比会让beta冒充alpha。跑不赢 static_long_leg 就等于没有择时价值。")

    if "scheme_comparison" in results:
        print(f"\n--- 3.5 权重方案对比 ---")
        print(results["scheme_comparison"].round(4).to_string())
        print("解读：若 ic_weighted 相对 equal 提升有限，就老实用等权。"
              "学术与业界的共识是等权极难被稳定打败。")

    if "sensitivity_summary" in results:
        s = results["sensitivity_summary"]
        print(f"\n--- 4. 权重敏感性 ({len(results['sensitivity'])} 组随机权重) ---")
        print(f"IC>0 的权重占比        : {s['IC_positive_share']:.1%}")
        print(f"IC 中位数 [5%, 95%]    : {s['IC_median']:.4f} "
              f"[{s['IC_p05']:.4f}, {s['IC_p95']:.4f}]")
        print(f"净夏普>0 的权重占比    : {s['sharpe_net_positive_share']:.1%}")
        print(f"净夏普中位数           : {s['sharpe_net_median']:.3f}")

    print(f"\n--- 5. 当前观点 ---")
    disp = scores["display"].dropna()
    if not disp.empty:
        last = disp.iloc[-1]
        d = scores["spread"].dropna().iloc[-1]
        for sector, meta in C.SECTORS.items():
            print(f"  {meta['label']:<12}({meta['primary']}) : {last[sector]:.1f} / 100")
        print(f"  分差(z)      : {d:+.2f}   建议仓位: "
              f"{scores['positions'].dropna().iloc[-1]:+.2f}")
        print(f"  日期         : {disp.index[-1].date()}")

    # ---------------- 落盘 ----------------
    outdir = os.path.join(ROOT, C.OUTPUT_DIR)
    os.makedirs(outdir, exist_ok=True)
    scores["display"].to_csv(f"{outdir}/scores_display.csv")
    scores["spread"].rename("spread").to_csv(f"{outdir}/spread.csv")
    for s, dfm in scores["dimension_scores"].items():
        dfm.to_csv(f"{outdir}/dimension_scores_{s}.csv")
    results["ic"].to_csv(f"{outdir}/ic.csv")
    results["strategy"]["returns"].to_csv(f"{outdir}/strategy_returns.csv")
    pd.DataFrame(results["strategy"]["stats"]).T.to_csv(f"{outdir}/strategy_stats.csv")
    if "sensitivity" in results:
        results["sensitivity"].to_csv(f"{outdir}/weight_sensitivity.csv", index=False)
    if "scheme_comparison" in results:
        results["scheme_comparison"].to_csv(f"{outdir}/scheme_comparison.csv")
    if "signal_ic" in results:
        results["signal_ic"].to_csv(f"{outdir}/signal_ic.csv")
    results["dimension_ic"].to_csv(f"{outdir}/dimension_ic.csv")
    if results.get("subperiods") is not None:
        results["subperiods"].to_csv(f"{outdir}/subperiods.csv")
    if isinstance(scores["weights"], pd.DataFrame):
        scores["weights"].to_csv(f"{outdir}/dimension_weights.csv")
    print(f"\n结果已落盘 → {C.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
