# data/

本目录存放抓取到的原始数据缓存，**内容不进版本库**（见 `.gitignore`）。

跑 `./run.sh fetch` 后会生成：

| 文件 | 内容 |
|---|---|
| `prices.csv` | 6 个 ETF 的日频收盘价与成交量，2007 至今，已做拆股与分红复权 |
| `macro.csv` | FRED 宏观序列：DFII10 / DTWEXBGS / BAA10Y / WALCL / VIXCLS |
| `source_meta.json` | 本次采用的价格数据源、是否复权、抓取时间 |

## 可选：真实 PE 历史

若提供 `pe_history.csv`（列：`date, ticker, pe`），估值维度会自动从
"相对价格拉伸度代理"切换为真实 PE 分位数。免费源拿不到长历史 forward PE，
所以默认用代理。

## 数据源说明

价格按 Tiingo → Stooq → yfinance 优先级自动回退，同一次运行不跨源混用。
宏观走 FRED；若 `fred.stlouisfed.org` 不通，配 `FRED_API_KEY` 走官方 API
（`api.stlouisfed.org`，不同主机名）。详见根目录 README 第六节。

**注意**：`BAMLH0A0HYM2`（ICE BofA 高收益 OAS）自 2026 年 4 月起被 FRED
限制为滚动 3 年窗口（授权原因），因此本项目改用 `BAA10Y` 度量信用利差。
