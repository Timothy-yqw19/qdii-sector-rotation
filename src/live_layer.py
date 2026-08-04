"""
实时增强层：新闻/社媒情绪 + 卖方盈利修正。

【为什么它不进回测 —— 这是必须主动讲清楚的方法论选择】
情绪与盈利修正在免费数据源上**拿不到足够长的历史**：
  - Reddit / 新闻 API 只能回溯到近期，且历史抓取受限流约束；
  - 免费档盈利预期数据点稀疏、且存在幸存者偏差与回填(backfill)问题。
硬把这两类数据塞进十几年的回测，只会得到一个用短样本外推、
且大概率带回填偏差的假 IC。所以本项目的处理是：
  - **回测层**只用可长期、可复现获取的量价 + 宏观信号；
  - **实时增强层**在仪表盘上单独展示，作为当下的“确认/证伪”参考，
    明确标注为未回测信号，不参与仓位计算。
这比假装全都回测过更诚实，也是买方研究里常见的“核心+卫星”信号处理方式。

依赖全部是可选的：装了就用，没装就降级，不会让主流程挂掉。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

# --------------------------------------------------------------------------
# 板块关键词与代表性成分股（用于新闻过滤与盈利修正取样）
# --------------------------------------------------------------------------

SECTOR_KEYWORDS = {
    "TECH": ["ai", "chip", "semiconductor", "gpu", "nvidia", "data center",
             "cloud", "software", "hardware"],
    "CONSUMER": ["consumer", "retail", "spending", "discretionary", "apparel",
                 "restaurant", "e-commerce", "housing"],
}

SECTOR_BASKET = {
    "TECH": ["NVDA", "MSFT", "AVGO", "AMD", "TSM", "MU", "AMAT", "ORCL"],
    "CONSUMER": ["AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG"],
}

REDDIT_SUBS = ["investing", "stocks", "wallstreetbets"]


@dataclass
class LiveSignal:
    sector: str
    name: str
    value: float | None
    detail: str
    available: bool


# ==========================================================================
# 情绪模型：FinBERT 优先，词典法降级
# ==========================================================================

_FINBERT = None


def _load_finbert():
    """
    FinBERT (ProsusAI/finbert) 在金融文本上显著优于 VADER：
    VADER 会把 'beat'、'short'、'bear' 这类词按通用语义判错，
    而这些恰恰是财经文本里最高频的方向性词汇。
    """
    global _FINBERT
    if _FINBERT is not None:
        return _FINBERT
    try:
        from transformers import pipeline
        _FINBERT = pipeline(
            "sentiment-analysis", model="ProsusAI/finbert", truncation=True, max_length=512
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [info] FinBERT 不可用({type(e).__name__})，降级到词典法")
        _FINBERT = False
    return _FINBERT


_POS = {"beat", "beats", "surge", "surges", "rally", "upgrade", "record", "strong",
        "growth", "outperform", "raise", "raised", "bullish", "jump", "soar", "boost"}
_NEG = {"miss", "misses", "plunge", "slump", "downgrade", "weak", "cut", "cuts",
        "warn", "warns", "layoff", "bearish", "fall", "drop", "slide", "recall"}


def _lexicon_score(text: str) -> float:
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    p, n = len(words & _POS), len(words & _NEG)
    return 0.0 if p + n == 0 else (p - n) / (p + n)


def score_texts(texts: list[str]) -> tuple[float, str]:
    """返回 (净情绪 ∈ [-1,1], 使用的方法)。"""
    texts = [t for t in texts if t and len(t) > 10]
    if not texts:
        return np.nan, "no_text"

    model = _load_finbert()
    if model:
        try:
            res = model(texts)
            mapping = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            vals = [mapping.get(r["label"].lower(), 0.0) * r["score"] for r in res]
            return float(np.mean(vals)), "finbert"
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] FinBERT 推理失败: {e}，降级到词典法")

    return float(np.mean([_lexicon_score(t) for t in texts])), "lexicon"


# ==========================================================================
# 数据源 1：Yahoo Finance 新闻（免费、无需 key）
# ==========================================================================

def fetch_news_headlines(tickers: list[str], per_ticker: int = 12) -> list[str]:
    try:
        import yfinance as yf
    except ImportError:
        return []
    heads = []
    for t in tickers:
        try:
            for item in (yf.Ticker(t).news or [])[:per_ticker]:
                title = item.get("title") or item.get("content", {}).get("title")
                if title:
                    heads.append(title)
        except Exception:  # noqa: BLE001, S112
            continue
    return heads


# ==========================================================================
# 数据源 2：Reddit（可选，需 praw 凭证）
# ==========================================================================

def fetch_reddit_titles(keywords: list[str], limit: int = 120) -> list[str]:
    """
    需要环境变量 REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT。
    没配就静默返回空，仪表盘会显示“未接入”。
    """
    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    if not (cid and csec):
        return []
    try:
        import praw
        reddit = praw.Reddit(
            client_id=cid, client_secret=csec,
            user_agent=os.getenv("REDDIT_USER_AGENT", "qdii-rotation/0.1"),
        )
        titles = []
        for sub in REDDIT_SUBS:
            for post in reddit.subreddit(sub).hot(limit=limit // len(REDDIT_SUBS)):
                text = f"{post.title} {getattr(post, 'selftext', '')[:300]}"
                if any(k in text.lower() for k in keywords):
                    titles.append(text)
        return titles
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] Reddit 抓取失败: {e}")
        return []


# ==========================================================================
# 情绪信号
# ==========================================================================

def sector_sentiment(sector: str) -> LiveSignal:
    kws = SECTOR_KEYWORDS[sector]
    texts = fetch_news_headlines([C.SECTORS[sector]["primary"]] + SECTOR_BASKET[sector][:5])
    texts = [t for t in texts if any(k in t.lower() for k in kws)] or texts
    texts += fetch_reddit_titles(kws)

    if not texts:
        return LiveSignal(sector, "新闻/社媒情绪", None, "无可用文本（检查网络或依赖）", False)

    val, method = score_texts(texts)
    return LiveSignal(
        sector, "新闻/社媒情绪", val,
        f"{len(texts)} 条文本, 方法={method}", not pd.isna(val),
    )


# ==========================================================================
# 盈利修正信号
# ==========================================================================

def sector_earnings_revision(sector: str) -> LiveSignal:
    """
    优先用 yfinance 的 EPS 修正表（上调家数 vs 下调家数）；
    拿不到则降级为分析师目标价隐含上行空间的截面均值。
    两者都只反映**当下**状态，不构成时间序列，故不进回测。
    """
    try:
        import yfinance as yf
    except ImportError:
        return LiveSignal(sector, "盈利修正", None, "yfinance 未安装", False)

    ups, downs, upsides = 0, 0, []
    for t in SECTOR_BASKET[sector]:
        tk = yf.Ticker(t)
        try:  # 路径一：EPS 修正家数
            rev = tk.eps_revisions
            if rev is not None and not rev.empty:
                row = rev.loc["0q"] if "0q" in rev.index else rev.iloc[0]
                ups += float(row.get("upLast30days", 0) or 0)
                downs += float(row.get("downLast30days", 0) or 0)
                continue
        except Exception:  # noqa: BLE001
            pass
        try:  # 路径二：目标价隐含上行
            info = tk.info
            tgt, cur = info.get("targetMeanPrice"), info.get("currentPrice")
            if tgt and cur:
                upsides.append(tgt / cur - 1)
        except Exception:  # noqa: BLE001
            continue

    if ups + downs > 0:
        breadth = (ups - downs) / (ups + downs)
        return LiveSignal(sector, "盈利修正广度", breadth,
                          f"近30日上调{int(ups)}家 / 下调{int(downs)}家", True)
    if upsides:
        return LiveSignal(sector, "目标价隐含上行(代理)", float(np.mean(upsides)),
                          f"{len(upsides)} 只成分股均值", True)
    return LiveSignal(sector, "盈利修正", None, "数据源无返回", False)


# ==========================================================================
# 汇总
# ==========================================================================

def run_live_layer() -> pd.DataFrame:
    rows = []
    for sector in C.SECTORS:
        for s in (sector_sentiment(sector), sector_earnings_revision(sector)):
            rows.append({
                "板块": C.SECTORS[sector]["label"],
                "sector": sector,
                "信号": s.name,
                "数值": s.value,
                "说明": s.detail,
                "可用": s.available,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run_live_layer().to_string(index=False))
