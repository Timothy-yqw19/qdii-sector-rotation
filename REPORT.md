# A Sector-Tilt Signal That Survived Its Own Screening

### Macro liquidity and the Tech-vs-Consumer relative trade, 2008–2026

**Tim Wang** · timothy19won@gmail.com · [github.com/Timothy-yqw19/qdii-sector-rotation](https://github.com/Timothy-yqw19/qdii-sector-rotation)

---

## Executive summary

I built a three-dimension scoring framework for the US tech-vs-consumer sector
pair, then ran it through a screening process designed to kill it. **Two of the
three dimensions failed and were removed. The third survived.**

What remains is a monthly macro-liquidity tilt with a 20-day Spearman IC of
**0.189**, a Newey-West *t* of **3.14** that stays above 3 even with a full year
of HAC lags, and an information ratio of **0.43** against a neutral 50/50
benchmark at 1.7% tracking error.

It also has a defect I could not engineer away: **strip out the best 12 months
of 236 and the cumulative excess return collapses from +15.2% to +1.9%.** The
signal is not an all-weather timing tool. It is a *stress-period* signal — it
earns +1.41% annualised during risk-off regimes versus +0.41% in calm markets.

That reframing is the actual conclusion of this project, and it only emerged
because the diagnostics were built to look for it.

![Equity curve](charts/01_equity_curve.png)

---

## 1. Why this pair, and why relative

**Universe.** Long leg: hard tech / AI (QQQ, with SMH and XLK as signal proxies).
Short leg: consumer discretionary (XLY, with XRT). Benchmark SPY.

**Relative, not absolute.** Absolute timing requires calling both direction and
magnitude; relative strength only requires ranking two assets. Common market beta
cancels, which raises signal-to-noise and matches the actual decision a sector
allocator faces. A consequence worth stating: any factor that moves both sectors
equally nets out of the spread by construction. That is correct behaviour, not a
missing feature.

**Three dimensions, chosen before looking at results:**

| Dimension | Speed | Signals |
|---|---|---|
| Momentum & positioning | Fast (daily) | Relative momentum 20d/60d, vol-adjusted momentum, money-flow proxy |
| Macro liquidity & risk appetite | Mixed | Real 10y rate, broad dollar, credit spread, Fed balance sheet, VIX |
| Relative valuation | Slow | Relative price percentile, distance from 200d MA percentile |

Each signal carries an explicit direction prior with a written economic
rationale, and the *magnitudes* differ across the two sectors — that asymmetry is
where the relative signal actually comes from. Example: a stronger dollar hurts
tech more than domestic-facing consumer names, because Nasdaq-100 foreign revenue
runs roughly 50–60% versus roughly 15% for XLY.

---

## 2. The screening: two dimensions rejected

![Dimension screening](charts/05_dimension_screening.png)

| Dimension | IC 5d | IC 20d | IC 60d | Standalone Sharpe | Turnover | Max DD | Verdict |
|---|---|---|---|---|---|---|---|
| Momentum & positioning | -0.024 | **-0.049** | -0.034 | **-0.126** | 6.20 | -34.2% | **Rejected** |
| Macro liquidity | +0.085 | **+0.187** | **+0.209** | **+0.390** | 1.81 | -6.8% | **Kept** |
| Relative valuation | -0.003 | -0.028 | -0.054 | -0.140 | 3.26 | -34.1% | **Rejected** |

Momentum was not merely uninformative — it was actively destructive: 3.4× the
turnover of the macro dimension, 5× the drawdown, negative return. Under daily
rebalancing it was worse still (Sharpe -0.45, turnover 25.0×). Equal-weighting
all three diluted the macro dimension's +0.187 down to **-0.045** and dragged the
Newey-West *t* from +3.14 to **-0.59**.

### 2.1 A citation I got wrong

I originally justified the momentum dimension with Moskowitz & Grinblatt (1999),
*Do Industries Explain Momentum?* **That citation was misapplied.** Their result
is cross-sectional momentum across roughly 20 industry portfolios — a ranking
effect across a broad cross-section. This project runs *pairwise* relative
momentum between two highly correlated sectors, where short-horizon mean
reversion is the more common empirical regularity. The negative IC is internally
consistent; it is not a data problem.

The broader lesson is about transferability: an effect documented in one
cross-sectional setting does not automatically survive translation to a two-asset
pair. Checking that translation should have happened before the signal was built,
not after.

### 2.2 The surviving dimension is not one variable in disguise

| Signal | IC 20d | 2008–14 | 2014–20 | 2020–26 |
|---|---|---|---|---|
| Fed balance sheet, 60d change | 0.120 | 0.157 | 0.286 | 0.029 |
| Credit spread (Baa–10y), 20d change | 0.118 | 0.084 | 0.187 | **0.107** |
| Broad dollar, 60d change | 0.105 | -0.046 | 0.252 | 0.076 |
| Real 10y rate, 60d change | 0.053 | 0.136 | 0.051 | -0.035 |
| VIX, 20d change | 0.038 | 0.055 | 0.051 | 0.020 |
| **Composite (equal-weighted)** | **0.187** | **0.162** | **0.336** | **0.104** |

Two observations.

**The composite beats every individual signal** (0.187 versus a best single of
0.120). Diversification is genuinely reducing idiosyncratic noise rather than
stacking correlated bets — which is the justification for keeping the whole
dimension instead of trading the Fed balance sheet alone.

**My second prior was also wrong.** I expected the real rate to dominate, on
standard equity-duration logic: tech cash flows sit further out, so they should
be more discount-rate sensitive. It is the weakest of the five and flips sign in
the most recent sub-sample. The work is being done by the Fed balance sheet and
the credit spread.

---

## 3. Does it survive stress-testing?

### 3.1 Sub-sample stability

![Sub-periods](charts/03_subperiods.png)

| Sub-period | IC 20d | Hit rate | Net Sharpe | Realised tech-vs-consumer drift |
|---|---|---|---|---|
| 2008-01 – 2014-04 | 0.156 | 56.2% | 0.358 | **-2.0%/yr** |
| 2014-04 – 2020-06 | 0.334 | 62.6% | 0.755 | +5.0%/yr |
| 2020-06 – 2026-08 | 0.114 | 55.0% | 0.264 | +7.2%/yr |

The obvious alternative explanation for any tech-vs-consumer signal over this
window is that it is one long bet on falling real rates and a secular tech bull
market. The sub-sample split addresses this directly: **the IC is positive in the
one sub-period where tech actually underperformed** (2008–14, drift -2.0%/yr).
That is not what a disguised buy-and-hold would produce.

The caveat is equally clear: the middle period is much stronger than the other
two, and the most recent one is the weakest.

### 3.2 Significance after correcting for autocorrelation

Slow-moving signals with overlapping forward windows produce heavily
autocorrelated residuals; naive *t*-statistics overstate significance
systematically. Rather than acknowledging this in prose, the report shows the
full decay curve.

![Newey-West decay](charts/04_newey_west.png)

```
20-day horizon: t_naive = 11.83 → L=22: 3.15 → L=63: 2.90 → L=126: 3.04 → L=252: 3.14
 5-day horizon: t_naive =  5.45 → L=7:  2.63 → L=63: 2.59 → L=126: 2.74 → L=252: 2.68
```

Extending the HAC lag window to a **full trading year** leaves *t* at 3.14. The
naive 11.83 is the number to discard. Signal persistence: ρ₁ = 0.997, autocorrelation
half-life ≈ 215 trading days (10.2 months), and 86 sign changes over 18.4 years —
roughly **4.7 independent bets per year**.

> A note on what *not* to report: the code also computes an AR(1) effective sample
> size, which returns N_eff ≈ 7. That number is not credible — the formula
> N(1-ρ)/(1+ρ) collapses mechanically as ρ → 1. It is flagged as a lower bound in
> the output rather than quoted as a result.

### 3.3 Where the signal actually works

![Quintiles](charts/02_quintiles.png)

| Quintile | Mean 20d forward relative return | Share positive |
|---|---|---|
| 1 (lowest) | -0.15% | 45.2% |
| 2 | -0.21% | 47.3% |
| 3 | -0.10% | 50.1% |
| 4 | +0.43% | 59.9% |
| **5 (highest)** | **+1.52%** | **70.6%** |

The payoff is **asymmetric**. Nearly all of it sits in the top quintile; the
bottom three are mildly negative and barely distinguishable from each other. This
is an overweight-confirmation signal. There is no evidence supporting the short
side, and it should not be traded that way.

### 3.4 Weighting: three schemes, no hand-picked numbers

Signals are averaged within each dimension (so that four correlated momentum
measures do not vote four times), then combined across dimensions. Rather than
asserting a weight vector, three schemes are run under identical conditions:

| Scheme | IC 20d | Net Sharpe | IR vs benchmark | Turnover |
|---|---|---|---|---|
| **Equal (adopted)** | 0.1894 | **0.425** | **0.432** | 1.61 |
| Inverse volatility | 0.1912 | 0.419 | 0.426 | 1.50 |
| Rolling IC-weighted (shrunk) | 0.1870 | 0.358 | 0.364 | 1.43 |

They are effectively indistinguishable, and equal weighting wins. IC weighting —
standard practice on the sell side — adds nothing here. A 300-draw Dirichlet
sensitivity analysis is also available.

The implementation detail that matters most: **IC weights are estimated only from
fully realised forward returns.** The rolling IC series is shifted forward by
`horizon + execution_lag − 1` days, so the most recent observation feeding the
weight at time *t* has an outcome that was already known at *t*. One day less
would be look-ahead, and two dedicated tests guard this.

---

## 4. Regime switching: proposed, tested, rejected

Fixed direction priors assume the macro-to-sector transmission is invariant. That
is questionable — easing in a recession is a *response to* a crisis, not a
tailwind.

**Method.** Risk-on/risk-off classified from rolling percentiles of the credit
spread and VIX, with hysteresis thresholds (enter 0.70, exit 0.50). Result:
risk-on 68.3% / risk-off 31.7%, ~20 episodes each, mean duration 160 / 78 trading
days. Crucially, the risk-off multipliers were **declared in advance from
economic reasoning and never fitted**, because roughly 20 risk-off episodes
cannot support estimating five coefficients.

| Variant | IC 20d | Net Sharpe | IR | Max DD | Turnover |
|---|---|---|---|---|---|
| **Baseline (kept)** | **0.189** | **0.425** | **0.432** | -6.6% | 1.61 |
| Regime-switched priors | 0.181 | 0.300 | 0.309 | -8.3% | 1.79 |
| Regime-scaled exposure (0.5×) | 0.189 | 0.408 | 0.413 | **-3.8%** | 1.45 |

**Verdict: not adopted.** The acceptance criterion was fixed before running —
both IC and net-of-cost Sharpe had to improve materially. Both got worse.

**Why it failed is more interesting than that it failed.** Three of five economic
priors were directionally right; two were backwards:

| Signal | My prior | IC risk-on | IC risk-off | Verdict |
|---|---|---|---|---|
| Credit spread | amplify 1.5× | 0.100 | **0.146** | correct |
| VIX | amplify 1.5× | 0.018 | **0.072** | correct |
| Dollar | amplify 1.3× | 0.102 | **0.133** | correct |
| Fed balance sheet | damp 0.5× | 0.083 | **0.169** | **backwards** |
| Real rate | damp 0.3× | 0.066 | 0.039 | over-damped |

Risk signals do bite harder under stress, as expected. But the intuition that
"balance-sheet expansion during stress is reactive and therefore contaminated" is
**empirically inverted** — its IC in risk-off is double its IC in risk-on. In
hindsight this is obvious: the 2009 and 2020 QE announcements *were* the turning
points for growth equities. Damping that signal discarded the most valuable part
of it.

**I did not recalibrate the multipliers to the observed regime-conditional ICs.**
That would be precisely the overfitting the pre-declaration was designed to
prevent, on ~20 episodes. The output of this experiment is a falsified hypothesis
and a corrected economic intuition — not a new parameter set.

One finding worth keeping: regime-scaled exposure **halves the maximum drawdown
(-6.6% → -3.8%) at a cost of 0.017 in Sharpe.** For a real portfolio that is often
a good trade. It is documented as a risk-management option rather than folded into
the default model.

---

## 5. What is wrong with this model

Stated up front rather than waiting to be asked.

**1. It does not beat buying and holding QQQ** (excess Sharpe -0.25). The defence
is that `static_long_leg` is an *ex-post* benchmark — nobody knew in 2008 that
tech would dominate — and against the ex-ante defensible 50/50 the IR is 0.43 at
1.7% tracking error. But the defence has a limit: **if an investor's real
alternative is simply owning QQQ, this model adds nothing for them.**

**2. Returns are extremely concentrated.**

| Excluding the best… | Remaining cumulative excess |
|---|---|
| nothing (236 months) | +15.2% |
| 3 months | +10.2% |
| 6 months | +6.6% |
| **12 months (5% of sample)** | **+1.9%** |
| 24 months (10%) | **-4.5%** |

Monthly hit rate is 53.8%. Split by regime: **+1.41% annualised excess in risk-off
(Sharpe 0.58) versus +0.41% in risk-on (Sharpe 0.35)**. This is a stress-period
signal. That is coherent with the regime diagnostics in §4 — but it means the
effective sample shrinks to a handful of crises, and future performance depends on
the next stress episode resembling the last few.

**3. Only the overweight side is supported** (§3.3).

**4. About 4.7 independent bets per year.** Enough to support *t* ≈ 3, nowhere near
enough for a strong claim, and it implies wide dispersion in any few-year
realisation.

**5. Decay.** IC by sub-period runs 0.156 → 0.334 → 0.114. Whether this is alpha
decay or the 2020s AI narrative overwhelming macro is unresolved; either way,
current-period application deserves a haircut.

**Other known limitations.** The valuation dimension uses a relative-price-extension
proxy rather than true forward P/E (free sources do not carry long forward-P/E
history; a user-supplied `data/pe_history.csv` switches it over). The flow signal is
a price-volume construct, not actual ETF creations/redemptions. The long-short
variant excludes borrow and market-impact costs.

---

## 6. Method: making the result falsifiable

**Anti-look-ahead — 13 automated tests.** The strongest is **truncation
invariance**: truncate the dataset at any historical date, re-run the whole signal
chain, and every value before the cut must match the full-sample run point for
point. Any leakage — full-sample standardisation, forward-filled future data, a
mis-signed shift — fails it. Additional tests cover rolling z-score and percentile
causality, macro publication lags, forward-return alignment, regime-classification
causality, a shuffled-signal null (IC ≈ 0), and a perfect-foresight control that
*must* be highly profitable or the backtest engine is wired backwards.

**Timing convention.** Signal computed at close T → executed at close T+1 →
returns accrue from T+2 (`execution_lag=2`). Macro series are lagged by their real
publication delay (+1 day for daily market series, +4 for the broad dollar index,
+9 for the Fed balance sheet), and the order — lag, then align to trading days,
then difference — is not interchangeable.

**Data.** Prices from Tiingo (dividend- and split-adjusted, 2007–present), with
Stooq and yfinance as fallbacks; no cross-source mixing within a run, since
inconsistent adjustment between the two legs would corrupt the relative return.
Macro from FRED.

**One data incident worth recording.** The credit-spread series was originally
`BAMLH0A0HYM2` (ICE BofA high-yield OAS), which returned only 787 rows starting
2023-08. I diagnosed silent endpoint truncation and wrote multi-path retry logic.
The actual cause was in the FRED page notes: *"Starting in April 2026, this series
will only include 3 years of observations"* — an ICE Data licensing change. No
amount of retrying could have fixed it. Substituting `BAA10Y` (Moody's Baa minus
10-year Treasury), which measures the same credit risk premium with 40 years of
unrestricted history, raised the most recent sub-period IC from 0.071 to 0.104.
The lesson now sits in the code comments: check the data source's own
documentation before writing code against an assumed failure mode.

---

## 7. How I would actually use it

Not as a timing engine. As a **monthly, asymmetric, stress-period overweight
confirmation signal**:

- Act on the top quintile; ignore the bottom (§3.3).
- Expect it to contribute during risk-off episodes and contribute little
  otherwise (§5.2).
- Size it as one input among several, not as a standalone strategy — an IR of
  0.43 at 1.7% tracking error is a modest tilt, not a return engine.
- Re-examine it if the risk-off IC advantage disappears, which is the specific
  condition under which the thesis breaks.

The strongest claim I am willing to make is narrow: *over 2008–2026, in this
sector pair, a diversified macro-liquidity composite carried information about
one-month relative returns that survived HAC correction, sub-sample splits, and a
weighting-scheme sensitivity check — concentrated in stress periods and on the
overweight side.*

Everything the screening removed is preserved in the repository so the rejections
can be reproduced: `./run.sh alldims` restores all three dimensions, `--regime`
restores regime switching.

---

## References

- Molchanov, A. & Stangl, J. (2024). The myth of business cycle sector rotation.
  *International Journal of Finance & Economics*.
- Moskowitz, T. & Grinblatt, M. (1999). Do industries explain momentum?
  *Journal of Finance*, 54, 1249–1290.
- Sarwar, G., Mateus, C. & Todorovic, N. (2020). Gauging the effectiveness of
  sector rotation strategies. *Journal of Asset Management*.
- FactSet. A practical approach to weighting signals.
- Newey, W. & West, K. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  *Econometrica*, 55, 703–708.

---

*Research exercise. Not investment advice. Backtested results are hypothetical and
do not represent actual trading.*
