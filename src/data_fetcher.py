"""
数据层：ETF 量价 (yfinance) + 宏观 (FRED CSV 直连，无需 API key)。

关键点：
- 所有下载结果落地到 data/*.csv，回测可复现；离线也能跑。
- 宏观数据在这里**只做原始存储**，发布滞后统一在 signals.py 里施加，
  避免“滞后逻辑散落在数据层”导致的前视偏差隐患。
- 无网络环境下可用 make_synthetic_data() 生成结构相似的模拟数据，
  用于验证 pipeline 逻辑（不能用于得出投资结论）。
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


# ==========================================================================
# 量价数据
# ==========================================================================

# --------------------------------------------------------------------------
# 三个数据源的适配器。统一返回 index=日期、列=[close, volume] 的 DataFrame。
# --------------------------------------------------------------------------

def _fetch_stooq(ticker: str, start: str) -> pd.DataFrame:
    """
    Stooq：无需 key、不限流、历史很长。
    代价是**不做分红复权**（拆股是复权的），相对收益会有股息偏差，
    这一点会在 source_meta 里如实记录并在报告/仪表盘上提示。
    """
    import requests

    url = C.STOOQ_CSV.format(symbol=f"{ticker.lower()}.us")
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    if "Date" not in r.text[:200]:
        raise RuntimeError(f"Stooq 返回异常内容: {r.text[:80]!r}")
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
    df = df.rename(columns={"Close": "close", "Volume": "volume"})
    return df.loc[df.index >= pd.Timestamp(start), ["close", "volume"]]


def _fetch_tiingo(ticker: str, start: str) -> pd.DataFrame:
    """
    Tiingo：免费 key，500 req/小时，**已做拆股与分红复权**，是三者里质量最好的。
    申请：https://www.tiingo.com/ → 设置环境变量 TIINGO_API_KEY
    """
    import requests

    key = os.getenv("TIINGO_API_KEY")
    if not key:
        raise RuntimeError("未设置 TIINGO_API_KEY")
    r = requests.get(
        C.TIINGO_URL.format(ticker=ticker),
        params={"startDate": start, "format": "json", "resampleFreq": "daily"},
        headers={"Content-Type": "application/json", "Authorization": f"Token {key}"},
        timeout=30,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        raise RuntimeError("Tiingo 返回空")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.set_index("date")
    # 注意：Tiingo 同时返回 close 与 adjClose。必须**先选后改名**，
    # 否则会出现两列同名 close，后续 df["close"] 拿到的是 DataFrame 而非 Series。
    missing = {"adjClose", "adjVolume"} - set(df.columns)
    if missing:
        raise RuntimeError(f"Tiingo 返回缺少复权字段 {missing}")
    out = df[["adjClose", "adjVolume"]].copy()
    out.columns = ["close", "volume"]
    return out


def _fetch_yfinance(ticker: str, start: str) -> pd.DataFrame:
    """yfinance：复权正确，但 2026 年限流极凶且接口常变，只作最后手段。"""
    import yfinance as yf

    df = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError("yfinance 返回空")
    df = df[["Close", "Volume"]]
    df.columns = ["close", "volume"]
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df


_ADAPTERS = {"tiingo": _fetch_tiingo, "stooq": _fetch_stooq, "yfinance": _fetch_yfinance}


def _try_source(source: str, tickers: list[str], start: str,
                pause: float, tries: int) -> tuple[list[pd.DataFrame], list[str]]:
    """在**单一数据源**内抓全部标的。不跨源混用——否则两条腿的复权口径会不一致。"""
    import time

    fetch = _ADAPTERS[source]
    frames, got = [], []
    for n, t in enumerate(tickers):
        for i in range(tries):
            try:
                raw = fetch(t, start)
                # 各源的原始返回字段命名不一，统一在这里做一次防御：
                # 列名重复会让下游 df["close"] 拿到 DataFrame 而不是 Series，
                # 报错点离病因很远，不如在入口就掐掉。
                if list(raw.columns) != ["close", "volume"]:
                    raise RuntimeError(f"适配器返回列异常: {list(raw.columns)}")
                sub = raw.reset_index()
                sub = sub.rename(columns={sub.columns[0]: "date"})
                sub["ticker"] = t
                sub = sub[["date", "ticker", "close", "volume"]].dropna(subset=["close"])
                if sub.empty:
                    raise RuntimeError("清洗后无有效行")
                frames.append(sub)
                got.append(t)
                print(f"  [ok] {source}/{t}: {len(sub)} 行 "
                      f"({sub['date'].min().date()} → {sub['date'].max().date()})")
                break
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {str(e)[:70]}"
                if i < tries - 1:
                    wait = 3 * (2 ** i)
                    print(f"  [retry] {source}/{t} {msg} → {wait}s")
                    time.sleep(wait)
                else:
                    print(f"  [fail] {source}/{t} {msg}")
        if n < len(tickers) - 1:
            time.sleep(pause)
    return frames, got


def fetch_prices(start: str = "2007-01-01", tickers: list[str] | None = None,
                 pause: float = 0.5, tries: int = 3,
                 sources: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """
    按优先级依次尝试数据源，第一个能拿全**必需标的**的源胜出。

    必需标的 = 基准 + 各板块 primary。全都拿不到时**直接抛错**，绝不返回半成品——
    否则空数据会被写进缓存覆盖好数据，还要到信号层才报一个莫名其妙的 KeyError。

    返回 (long格式价格, 元数据dict)
    """
    tickers = tickers or C.ALL_TICKERS
    required = {C.BENCHMARK} | {m["primary"] for m in C.SECTORS.values()}
    attempts = []

    for source in (sources or C.PRICE_SOURCE_PRIORITY):
        if source == "tiingo" and not os.getenv("TIINGO_API_KEY"):
            print("  [skip] tiingo：未设置 TIINGO_API_KEY（强烈建议申请，免费且已复权）")
            attempts.append((source, "无key"))
            continue

        print(f"\n[尝试数据源] {source}")
        frames, got = _try_source(source, tickers, start, pause, tries)
        missing = sorted(required - set(got))
        if missing:
            print(f"  [x] {source} 缺必需标的 {missing}，换下一个源")
            attempts.append((source, f"缺 {missing}"))
            continue

        optional_missing = sorted(set(tickers) - set(got))
        if optional_missing:
            print(f"  [warn] 辅助标的缺失（不影响主流程）: {optional_missing}")

        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        meta = {
            "price_source": source,
            "dividend_adjusted": C.SOURCE_IS_ADJUSTED.get(source, False),
            "tickers": sorted(got),
            "fetched_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
        print(f"  [√] 采用 {source}"
              f"（{'已复权' if meta['dividend_adjusted'] else '未做分红复权'}）")
        report_quality(df)
        return df, meta

    raise RuntimeError(
        "所有数据源都拿不到必需标的。尝试记录: "
        + "; ".join(f"{s}({why})" for s, why in attempts) + "\n"
        "建议按顺序排查：\n"
        "  1) 申请免费 Tiingo key（最省事，且数据已复权）：\n"
        "     https://www.tiingo.com/  然后 export TIINGO_API_KEY=你的key\n"
        "  2) 检查网络/代理是否能访问 stooq.com\n"
        "  3) Yahoo 限流通常 15-60 分钟自动解除，可稍后重试\n"
        "已有缓存未被覆盖。"
    )


def report_macro_coverage(macro: pd.DataFrame) -> pd.DataFrame:
    """
    宏观序列覆盖体检。

    【这个检查救过一次命，也纠正过一次误判，两个教训都值得记下来】

    起因：BAMLH0A0HYM2（ICE BofA 高收益 OAS）只抓到 787 行、2023-08 起。
    第一反应是"端点静默截断"，于是加了多路径重试 + 取最长的逻辑。
    但去查 FRED 页面才发现真正原因是**授权限制**：
        "Starting in April 2026, this series will only include 3 years of observations."
    ICE Data 的授权变更让 FRED 只保留滚动 3 年窗口，换端点、重抓统统没用。

    两个教训：
      1. 覆盖率检查必须做——否则会把"数据没有"误读成"信号无效"；
      2. 报警之后必须去查数据源说明——否则会把"授权限制"误诊成"技术故障"，
         然后在错误的方向上写一堆重试代码。
    现在对已知受限的序列在 config 里标 expected_min_years=0，不再误报。
    """
    known_limits = {s.fred_id: s.expected_min_years for s in C.MACRO_SERIES}
    rows = []
    for col in macro.columns:
        s = macro[col].dropna()
        if s.empty:
            rows.append({"series": col, "n": 0, "start": None, "end": None,
                         "years": 0.0, "flag": "全空"})
            continue
        years = (s.index.max() - s.index.min()).days / 365.25
        expected = known_limits.get(col, 12.0)
        if expected <= 0:
            flag = "已知受限(授权)，非故障"
        elif years < expected:
            flag = f"覆盖仅{years:.1f}年 < 预期{expected:.0f}年，先查数据源说明再改代码"
        else:
            flag = ""
        rows.append({
            "series": col, "n": len(s),
            "start": s.index.min().date(), "end": s.index.max().date(),
            "years": round(years, 1), "flag": flag,
        })
    return pd.DataFrame(rows)


def report_quality(prices: pd.DataFrame) -> None:
    """
    数据质量体检。免费数据源最常见的坑就是静默的坏值，
    与其在回测里表现为一个诡异的夏普，不如在这里先抓出来。
    """
    print("  [体检]")
    for t, g in prices.groupby("ticker"):
        r = g.set_index("date")["close"].pct_change()
        extreme = int((r.abs() > 0.20).sum())
        zero_vol = int((g["volume"] <= 0).sum())
        gaps = int((g["date"].diff().dt.days > 5).sum())
        flags = []
        if extreme:
            flags.append(f"单日>20%波动 {extreme} 次")
        if zero_vol:
            flags.append(f"零成交量 {zero_vol} 天")
        if gaps:
            flags.append(f">5日缺口 {gaps} 处")
        print(f"    {t:<5} {len(g):>5} 行  " + ("; ".join(flags) if flags else "无异常"))


# ==========================================================================
# 宏观数据（FRED 公开 CSV，无需 key）
# ==========================================================================

# 重试预算：这些数字是有意压小的。
# 教训：urllib3 的 Retry 适配器 + 自己写的重试循环会**相乘**，
# 再配上 45 秒超时，最坏情况能磨一个多小时且全程无输出，看起来就是死机。
# 现在只保留一层重试，超时压到 12 秒，且每次尝试都打印进度。
FRED_TIMEOUT = 12
FRED_ATTEMPTS = 2
# 覆盖跨度低于这个年数就认为该路径可能被截断，去试别的路径
MIN_HISTORY_YEARS = 12.0


def _fred_session():
    """朴素 session，**不挂** urllib3 Retry 适配器——重试统一由上层显式控制。"""
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def _parse_fred_csv(text: str, sid: str) -> pd.Series:
    raw = pd.read_csv(io.StringIO(text))
    date_col = raw.columns[0]
    val_col = [c for c in raw.columns if c != date_col][0]
    raw[date_col] = pd.to_datetime(raw[date_col])
    raw[val_col] = pd.to_numeric(raw[val_col], errors="coerce")  # FRED 用 '.' 表示缺失
    return raw.set_index(date_col)[val_col].rename(sid)


FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"


def _fetch_fred_api(sid: str, key: str, session) -> pd.Series:
    """
    FRED 官方 API（api.stlouisfed.org）。返回完整历史，无网页端点的授权/长度限制。
    只用 requests，不需要 fredapi 包。
    """
    r = session.get(
        FRED_API_URL,
        params={"series_id": sid, "api_key": key, "file_type": "json"},
        timeout=30,
    )
    if r.status_code == 400:
        raise RuntimeError("API 拒绝请求，多半是 FRED_API_KEY 无效")
    r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs:
        raise RuntimeError("API 返回空 observations")
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # '.' → NaN
    return df.set_index("date")["value"].rename(sid).sort_index()


def _span_years(ser: pd.Series) -> float:
    s = ser.dropna()
    return 0.0 if s.empty else (s.index.max() - s.index.min()).days / 365.25


def _fetch_fred_one(sid: str, session, min_years: float = MIN_HISTORY_YEARS) -> pd.Series:
    """
    多条取数路径依次尝试，**每一步都打印**——
    长时间静默是最糟的体验，用户无法区分"在重试"和"死机"。

    【关键：不是"拿到数据就收工"，而是要校验覆盖跨度】
    实测 BAMLH0A0HYM2（高收益利差）从 fredgraph.csv 端点只回了 2023 年至今 787 行，
    而该序列实际有 1996 年至今的日频数据。这种**静默截断**不报错，
    却会让信号在早期子样本全是 NaN，很容易被误读成"该信号早期无效"。
    所以：跨度不足 min_years 时不接受，继续试其他路径，最后保留最长的那份。
    """
    import time

    errors, candidates = [], []

    def _consider(ser: pd.Series, label: str) -> pd.Series | None:
        yrs = _span_years(ser)
        candidates.append((yrs, label, ser))
        if yrs >= min_years:
            print(f" ok ({yrs:.0f}年)")
            return ser
        print(f" 仅{yrs:.1f}年，疑似截断，继续找更长的")
        return None

    key = os.getenv("FRED_API_KEY")
    if key:
        # 官方 API 优先。注意它在 **api.stlouisfed.org**，
        # 与网页端点 fred.stlouisfed.org 是不同主机——
        # 实测有的网络只挡后者，此时配个 key 就全通了（./run.sh netcheck 可验证）。
        # 直接用 requests 调 JSON 接口，不依赖 fredapi 包，少一个安装步骤。
        try:
            print(f"    {sid}: 官方API ...", end="", flush=True)
            ser = _fetch_fred_api(sid, key, session)
            if len(ser.dropna()):
                got = _consider(ser, "api")
                if got is not None:
                    return got
            else:
                print(" 空")
        except Exception as e:  # noqa: BLE001
            print(f" {type(e).__name__}: {str(e)[:60]}")
            errors.append(f"api:{type(e).__name__}")

    paths = [
        ("csv", FRED_CSV.format(sid=sid), lambda t: _parse_fred_csv(t, sid)),
        ("txt", f"https://fred.stlouisfed.org/data/{sid}.txt",
         lambda t: _parse_fred_txt(t, sid)),
    ]
    for name, url, parser in paths:
        for attempt in range(1, FRED_ATTEMPTS + 1):
            print(f"    {sid}: {name} 第{attempt}次 ...", end="", flush=True)
            try:
                r = session.get(url, timeout=FRED_TIMEOUT)
                r.raise_for_status()
                ser = parser(r.text)
                if not len(ser.dropna()):
                    raise RuntimeError("解析后为空")
                got = _consider(ser, name)
                if got is not None:
                    return got
                break          # 这条路径通了但偏短，换下一条，不必再重试本条
            except Exception as e:  # noqa: BLE001
                print(f" {type(e).__name__}")
                errors.append(type(e).__name__)
                if attempt < FRED_ATTEMPTS:
                    time.sleep(2)

    if candidates:
        yrs, label, ser = max(candidates, key=lambda x: x[0])
        print(f"    {sid}: 所有路径都偏短，采用最长的一份（{label}, {yrs:.1f}年）")
        return ser
    raise RuntimeError(f"全部路径失败 ({', '.join(dict.fromkeys(errors))})")


def _parse_fred_txt(text: str, sid: str) -> pd.Series:
    """data/<ID>.txt 是空格分隔、带一段说明头，第一行以 DATE 开头才是表头。"""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().upper().startswith("DATE"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), sep=r"\s+")
    df.columns = ["date", "value"][: len(df.columns)]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].rename(sid)


def fetch_macro(series: list[C.MacroSeries] | None = None) -> pd.DataFrame:
    """
    返回 wide 格式: index=date, columns=fred_id。存的是**原始未滞后**的值。
    """
    series = series or C.MACRO_SERIES
    session = _fred_session()
    out, failed, consecutive = {}, [], 0
    for s in series:
        # 早停：连续两个序列全路径失败，说明是整体不通（网络/TLS/被墙），
        # 没必要把剩下的也各磨一分钟。快点失败、快点回退缓存。
        if consecutive >= 2 and not out:
            print(f"  [早停] 连续 {consecutive} 个序列全败，判定 FRED 整体不可达，"
                  f"跳过剩余 {len(series) - len(failed)} 个")
            failed.extend(x.fred_id for x in series if x.fred_id not in failed
                          and x.fred_id not in out)
            break
        try:
            ser = _fetch_fred_one(s.fred_id, session,
                                  min_years=s.expected_min_years)
            out[s.fred_id] = ser
            consecutive = 0
            print(f"  [ok] {s.fred_id} ({s.name}) {len(ser)} 行")
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] FRED {s.fred_id} 下载失败: {e}")
            failed.append(s.fred_id)
            consecutive += 1

    if not out:
        has_key = bool(os.getenv("FRED_API_KEY"))
        raise RuntimeError(
            "所有 FRED 序列都下载失败。\n"
            + ("  已设 FRED_API_KEY 仍失败 → key 可能无效，或 api.stlouisfed.org 也不通。\n"
               "  跑 ./run.sh netcheck 看官方API那一行。\n"
               if has_key else
               "  **未设 FRED_API_KEY**。先跑 ./run.sh netcheck：\n"
               "  若『FRED官方API』可达而『FRED网页端点』不可达（这是常见情况，\n"
               "  两者是不同主机名），申请一个免费 key 即可解决：\n"
               "      https://fred.stlouisfed.org/docs/api/api_key.html\n"
               "      echo 'export FRED_API_KEY=你的key' >> ~/.zshrc && source ~/.zshrc\n"
               "  不需要装任何额外的包。\n")
            + "  已有 data/macro.csv 缓存时程序会自动回退使用（宏观变化慢，"
              "沿用几天影响有限，超过两周务必重抓）。"
        )
    if failed:
        print(f"  [warn] 以下序列缺失，对应信号将为空: {failed}")
    return pd.DataFrame(out).sort_index()


def fetch_macro_vintage(series: list[C.MacroSeries] | None = None) -> pd.DataFrame | None:
    """
    ALFRED 真实数据窗（point-in-time）：取每个观测值**当时首次发布**的版本。

    与固定发布滞后近似的区别：
      固定滞后只处理"什么时候能看到"，不处理"当时看到的数值是多少"。
      被修订的序列（GDP、PMI、就业数据）后来的修订值会污染历史决策。
      ALFRED 给出每次发布的完整历史，能彻底消除修订偏差。

    对本项目的诚实说明：当前 4 个宏观序列里，DFII10 / DTWEXBGS / BAMLH0A0HYM2
    都是市场观测值，**几乎不修订**，所以固定滞后 ≈ 真实数据窗；只有 WALCL 偶有修订。
    换句话说这一层现在收益不大——但只要后面加入 PMI / CPI / 就业这类会被大幅修订的
    序列，它就是必需的。先把接口留好。

    需要免费 FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
    设置 FRED_API_KEY 环境变量后自动启用；没有就返回 None，主流程回退到固定滞后。
    """
    key = os.getenv("FRED_API_KEY")
    if not key:
        return None
    try:
        from fredapi import Fred
    except ImportError:
        print("  [info] 未安装 fredapi，跳过 ALFRED 数据窗（pip install fredapi）")
        return None

    fred = Fred(api_key=key)
    out = {}
    for s in (series or C.MACRO_SERIES):
        try:
            rel = fred.get_series_all_releases(s.fred_id)
            # 每个观测日只保留**最早**的那次发布，并以发布日为索引
            first = (rel.sort_values("realtime_start")
                        .groupby("date", as_index=False).first())
            ser = pd.Series(
                first["value"].astype(float).values,
                index=pd.to_datetime(first["realtime_start"]),
            )
            out[s.fred_id] = ser[~ser.index.duplicated(keep="last")].sort_index()
            print(f"  [ok] ALFRED {s.fred_id}: {len(ser)} 个首发观测")
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] ALFRED {s.fred_id} 失败({type(e).__name__})，该序列回退固定滞后")

    return pd.DataFrame(out).sort_index() if out else None


# ==========================================================================
# 缓存读写
# ==========================================================================

def save_prices(prices: pd.DataFrame, root: str = ".", meta: dict | None = None) -> None:
    """
    价格单独落盘。抓价格是最贵的一步（几千行×6标的），
    绝不能因为后面的宏观下载挂了就整批丢弃、下次从头再来。
    """
    import json

    os.makedirs(os.path.join(root, C.DATA_DIR), exist_ok=True)
    prices.to_csv(os.path.join(root, C.CACHE_PRICES), index=False)
    if meta:
        with open(os.path.join(root, C.CACHE_META), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  已缓存价格 → {C.CACHE_PRICES}")


def save_macro(macro: pd.DataFrame, root: str = ".") -> None:
    os.makedirs(os.path.join(root, C.DATA_DIR), exist_ok=True)
    macro.to_csv(os.path.join(root, C.CACHE_MACRO))
    print(f"  已缓存宏观 → {C.CACHE_MACRO}")


def save_cache(prices: pd.DataFrame, macro: pd.DataFrame, root: str = ".",
               meta: dict | None = None) -> None:
    save_prices(prices, root=root, meta=meta)
    save_macro(macro, root=root)


def load_macro_cache(root: str = ".") -> pd.DataFrame | None:
    path = os.path.join(root, C.CACHE_MACRO)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_meta(root: str = ".") -> dict:
    import json

    path = os.path.join(root, C.CACHE_META)
    if not os.path.exists(path):
        return {"price_source": "unknown", "dividend_adjusted": None}
    with open(path) as f:
        return json.load(f)


def dividend_bias_note(meta: dict) -> str | None:
    """未复权数据源必须在每个出口（终端报告、仪表盘）反复提示，否则很容易忘。"""
    if meta.get("dividend_adjusted") is False:
        return (
            f"价格源 {meta.get('price_source')} **未做分红复权**。"
            "QQQ 股息率约0.5%、XLY约1.2%，做相对收益时科技腿每年被系统性高估约0.7pct，"
            "对年化波动仅约6%的相对策略不可忽略。"
            "解决：申请免费 Tiingo key 后 export TIINGO_API_KEY=... 重抓。"
        )
    return None


def load_cache(root: str = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_csv(os.path.join(root, C.CACHE_PRICES), parse_dates=["date"])
    m = pd.read_csv(os.path.join(root, C.CACHE_MACRO), index_col=0, parse_dates=True)
    return p, m


def assert_cache_usable(prices: pd.DataFrame) -> None:
    """
    读缓存后立刻体检。宁可在这里报一句人话，
    也不要让空缓存一路飘到信号层，最后抛一个看不懂的 KeyError: 'SPY'。
    """
    required = {C.BENCHMARK} | {m["primary"] for m in C.SECTORS.values()}
    have = set(prices["ticker"].unique()) if len(prices) else set()
    missing = sorted(required - have)
    if missing:
        raise RuntimeError(
            f"缓存里缺少必需标的 {missing}（当前只有 {sorted(have) or '空'}）。\n"
            "多半是上一次下载被 Yahoo 限流写进了空数据。重新抓一次：\n"
            "    python run_pipeline.py --fetch --slow"
        )


def load_optional_pe(root: str = ".") -> pd.DataFrame | None:
    """
    可选：用户自备的真实 forward PE 历史。
    格式: date, ticker, pe  → 若存在，估值维度会自动改用真实PE分位数。
    """
    path = os.path.join(root, C.OPTIONAL_PE_CSV)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"  [ok] 检测到真实PE数据 {path}，估值维度将使用真实PE分位数")
    return df


# ==========================================================================
# 离线合成数据（仅用于验证 pipeline，不可用于投资结论）
# ==========================================================================

def make_synthetic_data(
    start: str = "2010-01-01", end: str = "2025-12-31", seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成结构相似的模拟数据：
    - 共同市场因子 + 板块特异因子，科技对实际利率有真实负载荷（便于验证符号逻辑）
    - 宏观序列带各自的频率（WALCL 周频，其余日频）
    """
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range(start, end)
    n = len(bdays)

    # --- 宏观潜在过程 ---
    # 序列名必须与 config.MACRO_SERIES 保持一致，否则测试会以 KeyError 形式炸掉。
    real_rate = np.cumsum(rng.normal(0, 0.03, n)) * 0.5 + 1.0     # DFII10, %
    dxy = 100 * np.exp(np.cumsum(rng.normal(0, 0.0035, n)))       # 美元指数
    credit = np.abs(2 + np.cumsum(rng.normal(0, 0.03, n)))        # BAA10Y 信用利差, %
    walcl = 4e6 * np.exp(np.cumsum(rng.normal(0.0004, 0.002, n)))  # Fed 总资产
    vix = np.abs(18 + np.cumsum(rng.normal(0, 0.25, n)))          # VIX

    # 让实际利率变化对科技有真实的负向影响（构造可检验的信号）
    d_real_60 = pd.Series(real_rate).diff(60).fillna(0).values

    mkt = rng.normal(0.0003, 0.010, n)
    specs = {
        "TECH": rng.normal(0.0002, 0.009, n) - 0.010 * d_real_60,
        "CONSUMER": rng.normal(0.0001, 0.008, n) - 0.004 * d_real_60,
    }

    beta = {"SPY": 1.0, "QQQ": 1.15, "SMH": 1.45, "XLK": 1.10, "XLY": 1.05, "XRT": 1.00}
    sector_of = {"QQQ": "TECH", "SMH": "TECH", "XLK": "TECH", "XLY": "CONSUMER", "XRT": "CONSUMER"}

    frames = []
    for t, b in beta.items():
        r = b * mkt + (specs[sector_of[t]] if t in sector_of else 0.0)
        r = r + rng.normal(0, 0.003, n)
        close = 100 * np.exp(np.cumsum(r))
        vol = np.exp(rng.normal(16, 0.35, n)) * (1 + 2 * np.abs(r))
        frames.append(pd.DataFrame(
            {"date": bdays, "ticker": t, "close": close, "volume": vol}
        ))
    prices = pd.concat(frames, ignore_index=True)

    macro = pd.DataFrame(
        {"DFII10": real_rate, "DTWEXBGS": dxy, "BAA10Y": credit,
         "WALCL": walcl, "VIXCLS": vix},
        index=bdays,
    )
    # WALCL 是周频序列：只在周三有值
    macro.loc[macro.index.dayofweek != 2, "WALCL"] = np.nan
    return prices, macro


# ==========================================================================
# CLI
# ==========================================================================

def netcheck() -> None:
    """
    逐主机连通性诊断。

    动机：FRED 反复 ReadTimeout 时，第一反应是"FRED 挂了"，
    但 FRED 的网页端点(fred.stlouisfed.org)和官方API(api.stlouisfed.org)
    是**两个不同主机**，很可能只有前者被挡。
    不做逐主机测试就会在错误的假设上浪费很多时间。
    """
    import time

    import requests

    targets = [
        ("Tiingo API", "https://api.tiingo.com/api/test", "价格数据（当前主力）"),
        ("Stooq", "https://stooq.com/q/d/l/?s=spy.us&i=d", "价格数据（备用）"),
        ("FRED 网页端点", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
         "宏观数据（默认路径）"),
        ("FRED 官方API", "https://api.stlouisfed.org/fred/series?series_id=DGS10"
                        "&api_key=abcdefghijklmnopqrstuvwxyz123456&file_type=json",
         "宏观数据（需FRED_API_KEY，注意是不同主机名）"),
        ("Yahoo Finance", "https://query1.finance.yahoo.com/v8/finance/chart/SPY",
         "价格数据（最后手段）"),
    ]
    print("逐主机连通性诊断（超时 10 秒）\n" + "=" * 66)
    for name, url, purpose in targets:
        t0 = time.time()
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            dt = time.time() - t0
            # 400/403 说明网络通、只是缺凭证，对连通性诊断而言算成功
            verdict = "可达" if r.status_code < 500 else f"服务端错误 {r.status_code}"
            print(f"  {name:<16} {verdict:<12} HTTP {r.status_code}  {dt:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<16} {'不可达':<12} {type(e).__name__}  "
                  f"{time.time() - t0:.1f}s")
        print(f"    └ 用途: {purpose}")

    print("=" * 66)
    print("怎么读这张表：")
    print("  · FRED网页端点不可达、但官方API可达 → 申请免费 FRED_API_KEY 即可解决")
    print("    https://fred.stlouisfed.org/docs/api/api_key.html")
    print("    然后 export FRED_API_KEY=... 并 pip install fredapi")
    print("  · 两个都不可达 → 是到 stlouisfed.org 整体的网络问题（代理/DNS/防火墙）")
    print("  · Tiingo 可达即可保证价格数据正常，宏观可暂时沿用缓存")


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if "--netcheck" in sys.argv:
        netcheck()
        return

    offline = "--offline" in sys.argv

    if offline:
        print("[离线模式] 生成合成数据（仅用于验证逻辑）")
        prices, macro = make_synthetic_data()
        meta = {"price_source": "synthetic", "dividend_adjusted": True}
    else:
        pause = 3.0 if "--slow" in sys.argv else 0.5
        print(f"[1/2] 下载 ETF 量价（标的间隔 {pause}s）...")
        try:
            prices, meta = fetch_prices(pause=pause)
        except RuntimeError as e:
            print(f"\n[中止] {e}")
            sys.exit(1)
        print(f"\n[2/2] 下载 FRED 宏观 ...")
        macro = fetch_macro()

    save_cache(prices, macro, root=root, meta=meta)
    note = dividend_bias_note(meta)
    if note:
        print(f"\n  [注意] {note}")
    print(f"日期范围: {prices['date'].min().date()} → {prices['date'].max().date()}")


if __name__ == "__main__":
    main()
