"""
全局配置：板块池、信号定义、方向先验、数据滞后、回测参数。

设计原则（面试可讲）：
1. 每个信号的方向(direction)不是拍脑袋，而是写明经济学理由(rationale)，
   并在回测中做符号稳健性检验。
2. 慢频宏观数据必须按“真实可获得日”滞后，不能用发布日当天的值。
3. 不预设 20%/15%/10% 这种假精度权重，维度内均值、维度间等权，
   再用权重敏感性分析证明结论不依赖某一组权重。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# 一、板块池（纯美股，做相对强弱而非绝对择时）
# --------------------------------------------------------------------------

BENCHMARK = "SPY"

SECTORS: dict[str, dict] = {
    "TECH": {
        "label": "硬科技 / AI",
        "primary": "QQQ",           # 打分与交易标的
        "proxies": ["SMH", "XLK"],  # 辅助信号（半导体贝塔更纯）
    },
    "CONSUMER": {
        "label": "可选消费",
        "primary": "XLY",
        "proxies": ["XRT"],
    },
}

# 轮动信号 = SCORE[LONG_LEG] - SCORE[SHORT_LEG]
LONG_LEG, SHORT_LEG = "TECH", "CONSUMER"

ALL_TICKERS = sorted(
    {BENCHMARK}
    | {s["primary"] for s in SECTORS.values()}
    | {p for s in SECTORS.values() for p in s["proxies"]}
)

# --------------------------------------------------------------------------
# 二、FRED 宏观序列（含真实发布滞后，单位=日历日）
# --------------------------------------------------------------------------
# 滞后来源：FRED 各序列的实际发布节奏。DFII10/BAMLH0A0HYM2 为 T+1 发布；
# WALCL 为周四发布上周三数据，最坏情况约 8 天可见 → 保守取 9 天。


@dataclass(frozen=True)
class MacroSeries:
    fred_id: str
    name: str
    publication_lag_days: int
    transform: str  # 'diff_60d' | 'diff_20d' | 'pct_chg_60d' | 'level'
    # 该序列合理的最短历史（年）。低于此值就怀疑取数路径有问题、去试别的路径。
    # 对于本身就受限的序列（见下方 BAMLH0A0HYM2），设成 0 表示"短是正常的"。
    expected_min_years: float = 12.0


MACRO_SERIES: list[MacroSeries] = [
    MacroSeries("DFII10", "10年期TIPS实际利率", 1, "diff_60d"),
    MacroSeries("DTWEXBGS", "美元指数(广义)", 4, "pct_chg_60d"),
    # 信用利差：用穆迪 Baa - 10年美债，而不是 ICE BofA 高收益 OAS。
    # 【踩过的坑，值得写下来】原本用 BAMLH0A0HYM2（高收益OAS），
    # 抓下来只有 787 行、2023-08 起。一开始以为是端点静默截断，
    # 查 FRED 页面才发现是**授权限制**：
    #   "Starting in April 2026, this series will only include 3 years of observations."
    # ICE Data 的授权变更让 FRED 只保留滚动3年窗口，换端点、重抓都没用。
    # BAA10Y 测的是同一件事（信用风险溢价 = 风险偏好），
    # 由穆迪编制、无授权限制，FRED 上有 1986 年至今的日频数据。
    MacroSeries("BAA10Y", "穆迪Baa-10年美债信用利差", 1, "diff_20d"),
    MacroSeries("WALCL", "美联储总资产", 9, "pct_chg_60d"),
    # VIX：另一个独立的风险偏好度量，1990年至今。
    # 与信用利差相关但不重合——前者是期权市场的隐含预期，后者是信用市场的实际定价。
    MacroSeries("VIXCLS", "VIX波动率指数", 1, "diff_20d"),
]

# --------------------------------------------------------------------------
# 三、信号定义：维度 → 信号 → 各板块方向先验
# --------------------------------------------------------------------------
# direction 语义：+1 表示“该信号的 z-score 越高，越利好这个板块”。
# 幅度差异（如 -1.0 vs -0.5）体现的是敏感度差异，不是权重——
# 它决定了同一个宏观冲击在两个板块间的“相对”影响。


@dataclass(frozen=True)
class Signal:
    key: str
    name: str
    dimension: str
    directions: dict[str, float]
    rationale: str


DIMENSIONS = [
    "momentum_positioning",   # 快：日频，价格与持仓
    "macro_liquidity",        # 混频：宏观流动性与风险偏好
    "valuation",              # 慢：相对估值 / 均值回归
]

# 【最终模型只启用宏观维度 —— 这是被数据筛出来的结果，不是一开始的设计】
# 三个维度都做了完整诊断（./run.sh report 可复现），结论：
#   · momentum_positioning : IC_20d = -0.049，单独夏普 -0.45，年换手 25.0 → 否决
#     配对相对动量与跨行业横截面动量是两回事，前者在本样本上呈反转。
#   · valuation            : IC_20d = -0.028，单独夏普 -0.19，年换手 11.8 → 否决
#   · macro_liquidity      : IC_20d = +0.187，三个子样本 IC 全为正 → 保留
# 保留另外两个维度的代码，是为了让否决过程可复现：
#   ./run.sh report                          三维全开，看它们如何互相稀释
#   ./run.sh macro                           最终模型
#   --dims=momentum_positioning              单独复现被否决的维度
ACTIVE_DIMENSIONS = ["macro_liquidity"]

DIMENSION_LABELS = {
    "momentum_positioning": "动量与持仓",
    "macro_liquidity": "宏观流动性与风险偏好",
    "valuation": "相对估值(均值回归)",
}

SIGNALS: list[Signal] = [
    # ---- 维度1：动量与持仓（快信号，日频真实更新） ----
    Signal(
        key="rel_mom_20",
        name="相对SPY 20日动量",
        dimension="momentum_positioning",
        directions={"TECH": 1.0, "CONSUMER": 1.0},
        rationale="板块相对基准的短期动量，横截面动量在1-3个月尺度上正相关于未来收益。",
    ),
    Signal(
        key="rel_mom_60",
        name="相对SPY 60日动量",
        dimension="momentum_positioning",
        directions={"TECH": 1.0, "CONSUMER": 1.0},
        rationale="中期趋势，过滤20日动量的噪音。",
    ),
    Signal(
        key="risk_adj_mom_60",
        name="波动调整后60日动量",
        dimension="momentum_positioning",
        directions={"TECH": 1.0, "CONSUMER": 1.0},
        rationale="用已实现波动率归一，避免高波动板块单纯因为波动大而得高分。",
    ),
    Signal(
        key="money_flow_20",
        name="20日资金流向代理",
        dimension="momentum_positioning",
        directions={"TECH": 1.0, "CONSUMER": 1.0},
        rationale=(
            "sign(日收益)×成交额 的20日净额 / 20日总成交额。ETF真实申赎份额历史"
            "在免费源上不可得，用量价合成的资金流代理，方向与净流入一致。"
        ),
    ),
    # ---- 维度2：宏观流动性与风险偏好（混频，按发布滞后对齐） ----
    Signal(
        key="DFII10_diff_60d",
        name="10年实际利率60日变化",
        dimension="macro_liquidity",
        directions={"TECH": -1.0, "CONSUMER": -0.5},
        rationale=(
            "实际利率上行压制长久期资产估值。科技板块现金流久期显著长于可选消费，"
            "故敏感度更高；消费同时受信贷成本影响，方向同为负但幅度较小。"
        ),
    ),
    Signal(
        key="DTWEXBGS_pct_chg_60d",
        name="美元指数60日涨幅",
        dimension="macro_liquidity",
        directions={"TECH": -1.0, "CONSUMER": -0.3},
        rationale=(
            "强美元压制海外收入折算。纳指/半导体海外收入占比约50-60%，"
            "XLY成分以内需为主(海外占比约15%)，故科技受损更大。"
        ),
    ),
    Signal(
        key="BAA10Y_diff_20d",
        name="信用利差(Baa-10Y)20日变化",
        dimension="macro_liquidity",
        directions={"TECH": -1.0, "CONSUMER": -0.7},
        rationale=(
            "信用利差走阔=risk-off，高贝塔板块受创更深。科技(尤其半导体)贝塔"
            "历史上高于可选消费，故设更高敏感度。"
            "用穆迪Baa-10Y而非ICE高收益OAS，因为后者自2026年4月起被FRED"
            "限制为滚动3年窗口（授权原因），无法支撑长样本回测。"
        ),
    ),
    Signal(
        key="VIXCLS_diff_20d",
        name="VIX 20日变化",
        dimension="macro_liquidity",
        directions={"TECH": -1.0, "CONSUMER": -0.7},
        rationale=(
            "隐含波动率上行=风险偏好收缩，高贝塔成长股回撤更大。"
            "与信用利差同向但信息源不同：VIX是期权市场的隐含预期，"
            "信用利差是信用市场的实际定价，两者背离时往往是转折点。"
        ),
    ),
    Signal(
        key="WALCL_pct_chg_60d",
        name="美联储总资产60日变化",
        dimension="macro_liquidity",
        directions={"TECH": 1.0, "CONSUMER": 0.4},
        rationale=(
            "央行扩表=流动性宽松，利好久期长、对流动性最敏感的成长资产。"
        ),
    ),
    # ---- 维度3：相对估值 / 均值回归（慢信号） ----
    Signal(
        key="rel_price_pctile_756",
        name="相对价格比3年分位数",
        dimension="valuation",
        directions={"TECH": -1.0, "CONSUMER": -1.0},
        rationale=(
            "板块/SPY 价格比的滚动3年分位数。分位数高=相对基准已大幅拉伸，"
            "构成均值回归逆风。注意：这是'相对价格拉伸度'而非真实PE——"
            "免费源拿不到长历史forward PE，若提供 data/pe_history.csv 会自动改用真实PE分位数。"
        ),
    ),
    Signal(
        key="dist_ma200_pctile",
        name="偏离200日均线分位数",
        dimension="valuation",
        directions={"TECH": -1.0, "CONSUMER": -1.0},
        rationale="价格相对200日均线的偏离度分位数，捕捉短期超买导致的回归压力。",
    ),
]

SIGNALS_BY_KEY = {s.key: s for s in SIGNALS}


def signals_in(dimension: str) -> list[Signal]:
    return [s for s in SIGNALS if s.dimension == dimension]


# --------------------------------------------------------------------------
# 四、标准化与打分参数
# --------------------------------------------------------------------------


@dataclass
class ScoringConfig:
    # 滚动z-score窗口（交易日）。只用截止当日的信息，防前视。
    zscore_window: int = 504          # ~2年
    zscore_min_periods: int = 252     # 至少1年样本才出分
    zscore_clip: float = 3.0          # 截断极端值，防单点主导
    # 分位数类信号的滚动窗口
    pctile_window: int = 756          # ~3年
    # 合成分平滑（降低日频换手噪音）
    score_smooth_days: int = 5

    # ---- 维度权重方案 ----
    # 'equal'       等权。最稳健的基线，也是学术上最难被打败的基准。
    # 'inverse_vol' 逆波动加权（风险平价思路）：让每个维度对最终分差的**风险贡献**
    #               大致相等，而不是让名义权重相等。解决"某个维度天生波动大、
    #               实际上主导了整个信号"的问题。
    # 'ic_weighted' 滚动IC加权：按各维度近期预测力分配权重。
    #               业界做法（见 FactSet: A Practical Approach to Weighting Signals），
    #               但极易过拟合，因此本实现强制两道保险：
    #                 (a) IC 只用**已完全实现**的前瞻收益估计，严格样本外；
    #                 (b) 向等权收缩 shrinkage，默认只走一半。
    weight_scheme: str = "equal"
    dimension_weights: dict[str, float] = field(
        default_factory=lambda: {d: 1.0 / len(DIMENSIONS) for d in DIMENSIONS}
    )
    # 逆波动加权的滚动窗口
    vol_weight_window: int = 252
    # IC 加权参数
    ic_weight_window: int = 756       # 估计IC的滚动窗口（~3年）
    ic_weight_horizon: int = 20       # 用20日前瞻收益算IC
    ic_weight_shrinkage: float = 0.5  # 0=纯等权, 1=纯IC加权。默认各一半。
    ic_weight_floor: float = 0.05     # 每个维度的权重下限，防止某维度被完全清零


@dataclass
class BacktestConfig:
    # 执行滞后（交易日）：
    #   信号用 t 日收盘价算出 → t+1 收盘执行 → 从 t+2 开始赚取收益。
    #   对应 position.shift(2)。设为1即为“收盘价当天成交”的乐观假设。
    execution_lag: int = 2
    # IC 的前瞻窗口
    ic_horizons: tuple[int, ...] = (5, 20)
    # 仓位映射：分差经 tanh 映射到 [-1, 1] 的多空仓位
    position_scale: float = 1.0
    # 迟滞带(hysteresis)：分差绝对值低于该阈值时维持原仓位，抑制换手
    hysteresis_band: float = 0.15
    # 调仓频率: None/'D' 日频, 'W' 周频, 'M' 月频。
    # 定为月频的依据：宏观维度 IC 随期限单调上升（5日 0.085 → 20日 0.187 → 60日 0.209），
    # 且信号自相关半衰期约 10 个月。这是慢信号，日频调仓只会把价值烧在换手上。
    rebalance_freq: str | None = "M"
    # 单边交易成本（bps of turnover）
    cost_bps: float = 5.0
    # 权重敏感性分析抽样次数
    n_weight_draws: int = 300
    random_seed: int = 42


# --------------------------------------------------------------------------
# 四点五、Regime（风险状态）切换
# --------------------------------------------------------------------------
# 【要解决的方法论缺口】
# 基线模型的方向先验是固定的，隐含假设"宏观变量→板块"的传导在任何环境下都一样。
# 这个假设明显可疑：正常时期降息利好长久期成长股，但衰退期的降息是**对危机的反应**，
# 此时"实际利率下行"往往伴随风险资产下跌，传导方向被打断甚至反转。
#
# 【为什么不用数据去拟合 regime 参数】
# 全样本只有约 86 次信号方向切换（每年 4.7 次）。
# 若再按 regime 二分、每个信号各估一套系数，等于用几十个观测去估十几个参数，
# 几乎必然过拟合。所以本项目的做法是：
#   1. 用经济学理由**事先声明**一套乘数（下方 REGIME_MULTIPLIERS）；
#   2. 然后严格检验它相对固定先验是否真的改善；
#   3. 若改善不明显，就如实报告并**保持基线模型不变**。
# 这与前面否决动量/估值两个维度是同一套纪律。


@dataclass
class RegimeConfig:
    # 用哪些序列度量市场压力（取水平值的滚动分位数后平均）
    stress_series: tuple[str, ...] = ("BAA10Y", "VIXCLS")
    pctile_window: int = 756          # ~3年滚动分位
    # 迟滞阈值：压力分位数上穿 enter 进入 risk-off，下穿 exit 才回到 risk-on。
    # 两个阈值不同是为了避免在边界反复横跳（与仓位迟滞带同样的道理）。
    enter_risk_off: float = 0.70
    exit_risk_off: float = 0.50
    # 是否启用。默认 False —— 先当作待检验的假设，而不是既定设计。
    enabled: bool = False
    # risk-off 时整体降低敞口的比例（1.0 = 不降）。作为独立变体单独检验。
    risk_off_gross_scale: float = 1.0


REGIME = RegimeConfig()

REGIME_LABELS = {"risk_on": "风险偏好(risk-on)", "risk_off": "风险规避(risk-off)"}

# 各信号在 risk-off 状态下的方向乘数（risk-on 恒为 1.0 作为基准）。
# 全部来自经济学推理，不是拟合出来的：
REGIME_MULTIPLIERS: dict[str, float] = {
    # 利率类：压力期传导被打断。降息是对危机的反应而非宽松红利，
    # 此时"实际利率下行→利好成长"的逻辑失效，故大幅衰减而非简单沿用。
    "DFII10_diff_60d": 0.3,
    # 央行扩表：压力期的扩表同样是被动救火，信号含义被污染，衰减一半。
    "WALCL_pct_chg_60d": 0.5,
    # 风险类：压力期反而更灵敏——利差走阔、VIX跳升对高贝塔的杀伤被放大。
    "BAA10Y_diff_20d": 1.5,
    "VIXCLS_diff_20d": 1.5,
    # 美元：压力期的美元走强是避险资金回流的症状，对海外收入占比高的科技冲击更大。
    "DTWEXBGS_pct_chg_60d": 1.3,
}


def regime_multiplier(signal_key: str, regime: str) -> float:
    """risk-on 恒为 1.0；risk-off 查表，未列出的信号默认不变。"""
    if regime != "risk_off":
        return 1.0
    return REGIME_MULTIPLIERS.get(signal_key, 1.0)


# --------------------------------------------------------------------------
# 四点六、多板块对扩展：把方向先验从"手写"改成"由板块属性推导"
# --------------------------------------------------------------------------
# 【为什么需要】
# 单对(科技vs消费)只有 23 个第5档独立事件，样本量撑不起任何二次切分
# （见 find_patterns.py：六条假设全部因样本不足或不显著而无法采纳）。
# 扩到多对可以把事件数提升一个数量级，并且顺带回答一个更硬的问题：
# 这个宏观信号是**只在科技vs消费上成立**，还是有跨板块的普遍性？
#
# 【为什么不能手写方向先验】
# "科技久期长、海外收入占比高"这套推理没法直接套到金融vs公用事业上。
# 手写 N 对就要写 N 套先验，既不可审也容易前后矛盾。
# 改成：每个板块打三个属性分，信号方向由属性自动推导。
# 加一个新板块只需填三个数，逻辑保持一致。

# 属性取值参考现实特征，量纲统一到 0~1.3：
#   duration : 现金流久期。成长股高、公用事业因高分红久期看似低但对利率极敏感
#              （债券替代属性），故给中高值；必需消费与金融偏低。
#   foreign  : 海外收入占比。半导体/科技最高，公用事业几乎为0。
#   beta     : 相对大盘的波动/周期敏感度。
SECTOR_ATTRS: dict[str, dict[str, float]] = {
    "TECH":        {"duration": 1.00, "foreign": 1.00, "beta": 1.00},
    "SEMIS":       {"duration": 1.00, "foreign": 1.30, "beta": 1.30},
    "CONS_DISC":   {"duration": 0.50, "foreign": 0.30, "beta": 0.90},
    "STAPLES":     {"duration": 0.25, "foreign": 0.40, "beta": 0.40},
    "UTILITIES":   {"duration": 0.80, "foreign": 0.05, "beta": 0.45},
    "FINANCIALS":  {"duration": 0.25, "foreign": 0.25, "beta": 1.05},
    "HEALTHCARE":  {"duration": 0.60, "foreign": 0.45, "beta": 0.65},
    "INDUSTRIALS": {"duration": 0.40, "foreign": 0.50, "beta": 1.00},
}

SECTOR_TICKERS: dict[str, dict] = {
    "TECH":        {"primary": "XLK", "proxies": ["QQQ"], "label": "科技"},
    "SEMIS":       {"primary": "SMH", "proxies": [],      "label": "半导体"},
    "CONS_DISC":   {"primary": "XLY", "proxies": ["XRT"], "label": "可选消费"},
    "STAPLES":     {"primary": "XLP", "proxies": [],      "label": "必需消费"},
    "UTILITIES":   {"primary": "XLU", "proxies": [],      "label": "公用事业"},
    "FINANCIALS":  {"primary": "XLF", "proxies": [],      "label": "金融"},
    "HEALTHCARE":  {"primary": "XLV", "proxies": [],      "label": "医疗"},
    "INDUSTRIALS": {"primary": "XLI", "proxies": [],      "label": "工业"},
}

# 每个宏观信号由哪个属性驱动、符号如何。
# 这是全部的方向逻辑，加板块不需要再碰这里。
SIGNAL_ATTR_MAP: dict[str, tuple[str, float]] = {
    "DFII10_diff_60d":      ("duration", -1.0),  # 实际利率上行压制长久期
    "DTWEXBGS_pct_chg_60d": ("foreign",  -1.0),  # 强美元压制海外收入
    "BAA10Y_diff_20d":      ("beta",     -1.0),  # 利差走阔打击高贝塔
    "VIXCLS_diff_20d":      ("beta",     -1.0),  # 波动上行同理
    "WALCL_pct_chg_60d":    ("duration", +1.0),  # 扩表利好长久期
}

# 待检验的板块对。选择标准：两侧在某个属性上有明显差异，
# 否则该信号在分差里会被抵消掉（见 README 第一节关于"相对"的说明）。
SECTOR_PAIRS: list[tuple[str, str]] = [
    ("TECH", "CONS_DISC"),      # 原始对：久期差 + 海外收入差
    ("TECH", "STAPLES"),        # 成长 vs 防御，久期差最大
    ("SEMIS", "STAPLES"),       # 同上但更极端
    ("TECH", "UTILITIES"),      # 两者久期都高，主要差在贝塔与海外收入
    ("CONS_DISC", "STAPLES"),   # 经典风险偏好对，久期差小、贝塔差大
    ("FINANCIALS", "UTILITIES"),  # 久期方向相反的一对
    ("INDUSTRIALS", "STAPLES"),   # 周期 vs 防御
    ("TECH", "HEALTHCARE"),     # 成长内部分化
]

MULTIPAIR_TICKERS = sorted(
    {BENCHMARK}
    | {SECTOR_TICKERS[s]["primary"] for s in SECTOR_TICKERS}
    | {p for s in SECTOR_TICKERS.values() for p in s["proxies"]}
)


# --------------------------------------------------------------------------
# 四点七、样本外检验用的长历史配置（2000–2007 是完全未被使用过的样本）
# --------------------------------------------------------------------------
# 【为什么需要另一套序列】
# 主模型的 5 个宏观信号里有 3 个在 2007 年前不存在：
#   DFII10   10年期TIPS实际利率 —— 2003年起
#   DTWEXBGS 广义美元指数       —— 2006年起
#   WALCL    美联储总资产       —— 2002年起
# 直接把主模型往前跑，前期会有一半信号是空的，结果没有解释力。
#
# 【替代原则：同一个经济含义，换一个有长历史的度量】
#   实际利率 → DGS10 名义10年美债（1962年起）。名义与实际在60日变化尺度上高度同向。
#   广义美元 → DTWEXM 主要货币美元指数（1973–2020）。同为贸易加权，覆盖更早。
#   联储扩表 → **直接丢弃**。M2 是月频且含义不同，强行替代不如不用。
#
# 【关键设计：新旧两段跑的是同一个 4 信号简化模型】
# 否则"样本外表现较差"会分不清是模型失效还是信号变少了。
# DTWEXM 于 2020 年停更，所以对照段取 2008–2019。

OOS_SERIES: list["MacroSeries"] = []   # 在 MacroSeries 定义之后填充（见文件末尾）

OOS_SIGNAL_ATTR_MAP: dict[str, tuple[str, float]] = {
    "DGS10_diff_60d":      ("duration", -1.0),
    "DTWEXM_pct_chg_60d":  ("foreign",  -1.0),
    "BAA10Y_diff_20d":     ("beta",     -1.0),
    "VIXCLS_diff_20d":     ("beta",     -1.0),
}

# 板块 ETF 的实际成立日：SPDR 系列 1998-12-22，QQQ 1999-03，SMH/XRT 更晚。
OOS_PRICE_START = "1998-12-22"
OOS_PERIODS = {
    "样本外 2000-2007": ("2000-01-01", "2007-12-31"),
    "对照   2008-2019": ("2008-01-01", "2019-12-31"),
}


def derive_directions_oos(sector: str) -> dict[str, float]:
    attrs = SECTOR_ATTRS[sector]
    return {k: sign * attrs[attr] for k, (attr, sign) in OOS_SIGNAL_ATTR_MAP.items()}


def derive_directions(sector: str) -> dict[str, float]:
    """由板块属性推导该板块在各宏观信号上的方向先验。"""
    attrs = SECTOR_ATTRS[sector]
    return {key: sign * attrs[attr] for key, (attr, sign) in SIGNAL_ATTR_MAP.items()}


OOS_SERIES = [
    MacroSeries("DGS10", "10年期美债名义收益率", 1, "diff_60d"),
    MacroSeries("DTWEXM", "美元指数(主要货币,长历史)", 4, "pct_chg_60d"),
    MacroSeries("BAA10Y", "穆迪Baa-10年美债信用利差", 1, "diff_20d"),
    MacroSeries("VIXCLS", "VIX波动率指数", 1, "diff_20d"),
]

SCORING = ScoringConfig()
BACKTEST = BacktestConfig()

# --------------------------------------------------------------------------
# 五、路径
# --------------------------------------------------------------------------

DATA_DIR = "data"
CACHE_PRICES = f"{DATA_DIR}/prices.csv"
CACHE_MACRO = f"{DATA_DIR}/macro.csv"
CACHE_META = f"{DATA_DIR}/source_meta.json"
OPTIONAL_PE_CSV = f"{DATA_DIR}/pe_history.csv"
OUTPUT_DIR = "output"

# --------------------------------------------------------------------------
# 六、数据源优先级
# --------------------------------------------------------------------------
# 2026年现状：Yahoo 对未认证请求限流极凶，yfinance 已不适合当主力数据源。
# 优先级理由：
#   1. Tiingo   —— 免费档 500 req/小时，**已做拆股与分红复权**，数据质量高。需免费key。
#   2. Stooq    —— 无需key、不限流，但**只有不复权收盘价**（无分红调整）。
#   3. yfinance —— 复权正确但限流严重且接口经常变，降为最后手段。
#
# 分红偏差的严重性（用 Stooq 时必须知道）：
#   QQQ 股息率约0.5%，XLY约1.2%，SPY约1.2%。做 TECH-CONSUMER 相对收益时，
#   不复权会让科技腿每年被系统性高估约 0.7个百分点。对一个年化波动仅约6%的
#   相对策略来说，这个偏差不可忽略——所以强烈建议配一个免费 Tiingo key。

PRICE_SOURCE_PRIORITY = ["tiingo", "stooq", "yfinance"]

STOOQ_CSV = "https://stooq.com/q/d/l/?s={symbol}&i=d"
TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

# 各源的股息复权情况，用于在报告与仪表盘上如实标注
SOURCE_IS_ADJUSTED = {"tiingo": True, "stooq": False, "yfinance": True, "synthetic": True}
