# Grade A Diagnostic — 2026-05-05 09:11

Phase 1A read-only investigation. No production code or scoring weights have been changed.
Source data: `data/grade_history.csv` (165 rows; 104 with realized 5d forward returns), cross-checked against `data/grade_stats.json` and `data/grade_attribution.json`.

**TL;DR**

- The dashboard shows `Grade A: 57% hit, -0.29R avg, n=90`. A 57% hit rate with negative R means losses are bigger than wins.
- The components weighted highest in `conviction_score` (persistence, consistency, acceleration) have **near-zero Spearman correlation** with forward returns on the persisted panel.
- Two features that ARE statistically significant (`multileg_share` positive, `sweep_share` negative) are **not in the formula**.
- The 7-tier grade ladder is **not monotonic**: B (+3.8%) > A- (+0.6%) > B+ (-4.2%) on the persisted sample.
- See sections 1-6 below for the full evidence and section labelled 'Next-step recommendation' at the bottom.

---


## 1. Per-grade-tier forward-return distribution

Per-tier outcomes split A+/A/A- separately (the live backtest collapses them).

**Fine-grade tiers (signed excess return vs SPY, R uses 2% stop assumption):**

| Grade | n | Hit | Mean Excess | Median Excess | Mean R | Median R | Best R | Worst R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1 | 100.0% | +72.83% | +72.83% | 36.41 | 36.41 | 36.4 | 36.4 |
| A- | 14 | 57.1% | +0.59% | +2.66% | 0.30 | 1.33 | 29.2 | -28.0 |
| B+ | 37 | 48.6% | -4.21% | -0.19% | -2.11 | -0.09 | 23.0 | -33.6 |
| B | 47 | 57.4% | +3.76% | +2.41% | 1.88 | 1.20 | 27.2 | -25.7 |
| B- | 5 | 40.0% | +7.46% | -0.34% | 3.73 | -0.17 | 44.7 | -23.6 |

**Coarse-grade tiers (matches dashboard headline aggregation):**

| Coarse | n | Hit | Mean Excess | Mean R | Best R | Worst R |
| --- | --- | --- | --- | --- | --- | --- |
| A | 15 | 60.0% | +5.41% | 2.70 | 36.4 | -28.0 |
| B | 89 | 52.8% | +0.65% | 0.33 | 44.7 | -33.6 |

**Coarse grade x direction (LONG vs SHORT separately):**

| Coarse | Direction | n | Hit | Mean Excess | Mean R |
| --- | --- | --- | --- | --- | --- |
| A | BULLISH | 9 | 44.4% | -6.96% | -3.48 |
| A | BEARISH | 6 | 83.3% | +23.96% | 11.98 |
| B | BULLISH | 49 | 49.0% | +2.49% | 1.24 |
| B | BEARISH | 40 | 57.5% | -1.60% | -0.80 |

**Monotonicity check** (expecting A+ > A > A- > B+ > B > B- > C):
- Observed: A-(+0.59%) > B+(-4.21%) > B(+3.76%) > B-(+7.46%)
- Monotonic? **NO** (grade ladder is broken — a higher grade does not yield a higher mean return)

**Cross-check vs `data/grade_stats.json`:** live backtest reports Grade A: n=90, hit_rate=56.7%, mean_r=-0.29R, best=+14.2R, worst=-35.1R. This panel uses `grade_history.csv` (the persisted feature-attribution sample) and produces a different N because `grade_stats.json` is built by replaying `compute_multi_day_flow` over the snapshots archive (a much wider sample). For component-level analysis we use the panel; for the headline N we trust the backtest.

---

## 2. Per-component correlation with forward returns

Spearman rho of each feature vs **signed** excess return (positive rho means higher feature value → trade tends to go the right way).

| Feature | In conviction_score? | n | Spearman rho | Notes |
| --- | --- | --- | --- | --- |
| sweep_share | no | 104 | -0.2974 | *(notable)* |
| multileg_share | no | 104 | 0.2608 | *(notable)* |
| prem_mcap_bps | yes | 104 | -0.1581 | **(WRONG SIGN)** |
| window_return_pct | no | 104 | -0.0834 | — |
| perc_3_day_total_latest | no | 104 | 0.0720 | — |
| latest_oi_change | yes | 104 | 0.0641 | — |
| cumulative_premium | yes | 104 | -0.0552 | — |
| latest_iv_rank | no | 104 | 0.0400 | — |
| persistence_ratio | yes | 104 | 0.0183 | — |
| accel_ratio_today | yes | 104 | -0.0067 | — |
| accumulation_score | yes | 104 | 0.0054 | — |
| latest_put_call_ratio | no | 104 | -0.0016 | — |

**Quartile bucket means** — split each numeric feature into Q1..Q4, show mean signed-excess in each:

| Feature | Q1 (low) | Q2 | Q3 | Q4 (high) | Q4-Q1 spread |
| --- | --- | --- | --- | --- | --- |
| sweep_share | +8.85% | +0.44% | -0.78% | -3.16% | -12.01% |
| accel_ratio_today | +8.42% | -0.86% | -2.56% | +0.35% | -8.08% |
| cumulative_premium | +5.19% | -0.99% | +3.28% | -2.12% | -7.31% |
| window_return_pct | +2.99% | +3.75% | +2.59% | -3.97% | -6.96% |
| persistence_ratio | +2.12% | +6.34% | -9.56% | +6.45% | +4.33% |
| prem_mcap_bps | +6.76% | +2.64% | -6.56% | +2.51% | -4.26% |
| perc_3_day_total_latest | +3.16% | -3.71% | +6.70% | -0.81% | -3.97% |
| latest_oi_change | -3.29% | +6.08% | +2.31% | +0.26% | +3.55% |
| latest_iv_rank | +0.07% | -0.53% | +4.27% | +1.55% | +1.48% |
| accumulation_score | +0.26% | +0.15% | +5.79% | -0.85% | -1.11% |
| multileg_share | +5.62% | -3.83% | -2.44% | +6.00% | +0.39% |
| latest_put_call_ratio | +3.76% | -1.32% | -0.70% | +3.60% | -0.15% |

**Cross-check with `data/grade_attribution.json`** (the existing automated attribution): n_rows=104, status=ok. Top-ranked features match what we compute above.

**Question answered:** the components weighted **highest in `conviction_score`** (persistence_ratio 25%, accumulation/consistency 25%, accel 20%) all have Spearman correlation with forward returns near zero. Conversely, **`multileg_share`** (NOT in conviction_score) is significantly **positively** correlated, and **`sweep_share`** (NOT in conviction_score) is significantly **negatively** correlated. The current weighting is broadly orthogonal to forward returns.

---

## 3. Conditional analysis (does Grade A work in *some* regimes?)

All slices below restrict to **Grade A or A-** (the live 'Grade A' bucket per `coarse_grade`).

Grade A panel size: **n=15**. Grade B panel size: n=89 (used for relative comparison).

**3a. Grade A x Direction:**

| Direction | n | Hit | Mean Excess | Mean R |
| --- | --- | --- | --- | --- |
| BULLISH | 9 | 44.4% | -6.96% | -3.48 |
| BEARISH | 6 | 83.3% | +23.96% | 11.98 |

**3b. Grade A x DTE bucket:**

| DTE bucket | n | Hit | Mean Excess | Mean R |
| --- | --- | --- | --- | --- |
| 31-90 | 4 | 75.0% | +32.13% | 16.06 |
| unknown | 11 | 54.5% | -4.31% | -2.15 |

Compare to **all grades** by DTE (larger N):

| DTE bucket (all grades) | n | Hit | Mean Excess | Mean R |
| --- | --- | --- | --- | --- |
| 31-90 | 34 | 58.8% | +3.95% | 1.97 |
| 8-30 | 3 | 66.7% | +10.49% | 5.24 |
| 91+ | 4 | 50.0% | +4.48% | 2.24 |
| unknown | 63 | 50.8% | -0.70% | -0.35 |

**3c. Grade A x sector:**

| Sector (Grade A only, n>=2) | n | Hit | Mean Excess | Mean R |
| --- | --- | --- | --- | --- |
| Industrials | 9 | 66.7% | +15.57% | 7.79 |
| Technology | 5 | 40.0% | -12.87% | -6.43 |

**3d. Grade A x same-day market-flow regime** (proxy: aggregate bull/bear premium ratio):

| Market-flow regime | n | Hit | Mean Excess | Mean R |
| --- | --- | --- | --- | --- |
| mixed | 15 | 60.0% | +5.41% | 2.70 |

**3e. Grade A x sector-heat alignment** (does the trade's sector show concentration in the same direction that day?):

| Sector heat align | n | Hit | Mean Excess | Mean R |
| --- | --- | --- | --- | --- |
| aligned | 1 | 100.0% | +58.42% | 29.21 |
| neutral | 12 | 58.3% | +1.71% | 0.86 |
| opposite | 2 | 50.0% | +1.08% | 0.54 |

**3f. Grade A x per-day tailwind** (was the same-day basket of Grade A returns positive on average?):

| Same-day basket | n | Hit | Mean Excess | Mean R |
| --- | --- | --- | --- | --- |
| tailwind | 12 | 75.0% | +12.72% | 6.36 |
| headwind | 3 | 0.0% | -23.83% | -11.92 |

*Note: 3f is partially circular (we're conditioning on the panel's own outcome) but it isolates whether Grade A returns are *driven by a few good/bad days* — if the per-day variance dominates, recalibrating weights matters less than picking days.

---

## 4. Ticker concentration & effective sample size

- Grade A rows: n=15, unique tickers: 11
- Average rows per ticker: 1.36

**Top tickers in Grade A (by row count):**

| Ticker | Rows | Mean Excess | Mean R |
| --- | --- | --- | --- |
| CAR | 2 | +65.62% | 32.81 |
| PBI | 2 | +3.98% | 1.99 |
| GHM | 2 | -3.19% | -1.59 |
| AMSC | 2 | -0.40% | -0.20 |
| BLD | 1 | +8.11% | 4.06 |
| FRMI | 1 | +5.32% | 2.66 |
| NOW | 1 | -12.80% | -6.40 |
| WOLF | 1 | +6.63% | 3.31 |
| INFQ | 1 | +10.06% | 5.03 |
| MXL | 1 | -12.27% | -6.13 |
| POET | 1 | -55.96% | -27.98 |

**Same-ticker repeat days** (i.i.d. assumption check):

| Ticker | Rows | Avg gap between repeats |
| --- | --- | --- |
| AMSC | 2 | 1.0d |
| CAR | 2 | 1.0d |
| GHM | 2 | 1.0d |
| PBI | 2 | 1.0d |

**Block-bootstrap mean R (resample by ticker):**

- Naive mean R (i.i.d. assumption): **+2.70R**
- Block-bootstrap mean R (by ticker): **+2.61R**
- 95% CI from bootstrap: **[-5.34, +12.07]R**
- CI crosses zero — we cannot reject the hypothesis that the true mean R is 0 (or worse).

---

## 5. Worst-case attribution

Bottom-10 outcomes (by signed excess return) — what did our model think, and why was it wrong?

| Date | Ticker | Dir | Grade | Score | Pers | Prem/MC | Sweep | Multileg | DTE | Excess | R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-20 | CAR | BULL | B+ | 6.0 | 1.00 | 281.9 | 0.00 | 0.00 | unknown | -67.23% | -33.6R |
| 2026-04-24 | POET | BULL | A- | 6.8 | 1.00 | 1101.8 | 0.00 | 0.00 | unknown | -55.96% | -28.0R |
| 2026-04-23 | MXL | BEAR | B | 5.7 | 0.75 | 15.9 | 0.00 | 0.00 | unknown | -51.40% | -25.7R |
| 2026-04-27 | MXL | BEAR | B+ | 6.0 | 0.75 | 25.0 | 0.00 | 0.00 | unknown | -48.66% | -24.3R |
| 2026-04-28 | MXL | BEAR | B- | 4.9 | 0.75 | 25.0 | 0.00 | 0.00 | unknown | -47.14% | -23.6R |
| 2026-04-23 | POET | BULL | B+ | 6.6 | 1.00 | 459.0 | 0.00 | 0.00 | unknown | -31.94% | -16.0R |
| 2026-04-28 | BE | BEAR | B+ | 6.5 | 0.75 | 33.2 | 1.00 | 0.00 | 31-90 | -27.08% | -13.5R |
| 2026-04-28 | STX | BEAR | B+ | 6.7 | 1.00 | 13.0 | 0.00 | 0.00 | 31-90 | -24.28% | -12.1R |
| 2026-04-27 | WOLF | BEAR | B+ | 6.6 | 0.50 | 103.6 | 0.00 | 0.00 | unknown | -23.13% | -11.6R |
| 2026-04-27 | RMBS | BULL | B+ | 6.3 | 0.75 | 17.2 | 0.00 | 0.00 | unknown | -21.56% | -10.8R |

**Bottom-10 vs rest (mean of each feature):**

| Feature | Bottom-10 mean | Rest mean | Delta |
| --- | --- | --- | --- |
| persistence_ratio | 0.825 | 0.699 | 0.126 |
| prem_mcap_bps | 207.572 | 103.774 | 103.798 |
| accumulation_score | 52.650 | 54.746 | -2.096 |
| accel_ratio_today | 0.000 | 0.032 | -0.032 |
| cumulative_premium | 275328748.700 | 198199807.947 | 77128940.753 |
| latest_oi_change | 23.910 | 21.822 | 2.088 |
| sweep_share | 0.100 | 0.076 | 0.024 |
| multileg_share | 0.000 | 0.069 | -0.069 |
| window_return_pct | 20.194 | -1.211 | 21.405 |

**Direction breakdown of bottom-10:** BULLISH=4, BEARISH=6.

---

## 6. Candidate weight tweaks (simulated)

Each tweak below is a different weighting scheme over the same features. We score every row, redefine grades by score quartile within this sample (top-quartile = A), then measure forward returns of the new Grade A bucket. **No production code is changed.**

**Baseline (production weights replicated on this panel):**

- Weights: `persistence_ratio`=+0.25, `prem_mcap_bps`=+0.20, `accumulation_score`=+0.25, `accel_ratio_today`=+0.20, `cumulative_premium`=+0.05, `latest_oi_change`=+0.05
- New Grade A: n=26, hit=57.7%, mean_excess=+1.49%, **mean_r=0.74**

**Tweak A — Spearman re-weighting (positive-rho only):**
Use Spearman rho with forward returns as the weight; drop features with non-positive correlation.

- Weights: `persistence_ratio`=+0.02, `accumulation_score`=+0.01, `latest_oi_change`=+0.06, `multileg_share`=+0.26, `latest_iv_rank`=+0.04, `perc_3_day_total_latest`=+0.07
- New Grade A: n=26, hit=61.5%, mean_excess=+6.74%, **mean_r=3.37**

**Tweak B — feature-set swap:**
Promote the two features that *do* correlate (`multileg_share` positive, `sweep_share` negative); demote the orthogonal ones.

- Weights: `multileg_share`=+0.40, `sweep_share`=-0.30, `prem_mcap_bps`=-0.15, `persistence_ratio`=+0.05, `cumulative_premium`=+0.05, `latest_oi_change`=+0.05
- New Grade A: n=26, hit=76.9%, mean_excess=+9.89%, **mean_r=4.94**

**Tweak C — keep production weights, but only label as Grade A when DTE is 8-90 days:**
Same `conviction_score` formula; the recalibration is just a downstream gate.

| Cohort | n | Hit | Mean Excess | Mean R |
| --- | --- | --- | --- | --- |
| Grade A & DTE 8-90+ days (kept) | 4 | 75.0% | +32.13% | 16.06 |
| Grade A & DTE unknown (rejected) | 11 | 54.5% | -4.31% | -2.15 |

**Ranked tweak summary (by Grade A mean R):**

| Tweak | A n | A hit | A mean excess | A mean R |
| --- | --- | --- | --- | --- |
| Tweak B (feature swap) | 26 | 76.9% | +9.89% | 4.94 |
| Tweak A (Spearman re-weight) | 26 | 61.5% | +6.74% | 3.37 |
| Baseline (current) | 26 | 57.7% | +1.49% | 0.74 |

---

## Next-step recommendation

- Grade A panel: n=15, mean R = +2.70
- Grade ladder monotonic? **NO**
- Top-weighted components (`persistence_ratio`, `accumulation_score`, `prem_mcap_bps`) Spearman rho: persist=0.018, consist=0.005, intensity=-0.158
- Predictive components NOT in `conviction_score`: multileg_share rho=0.261, sweep_share rho=-0.297

**Recommendation: build proper backtest infra (Phase 1B-infra) before changing weights.**
The components currently weighted highest (persistence, consistency, accel) have near-zero correlation with forward returns over this sample. The components that DO correlate (`multileg_share`, `sweep_share`) are not in the formula. A weight rewrite is justified but should be done with proper out-of-sample / walk-forward validation, not by simulating on the same panel that informed the weights. Build the infra, then recalibrate.

**Interim mitigation (zero-code):** treat the dashboard's 'Grade A backtest' header as *informational only* until recalibration is done. Do NOT auto-promote Grade A names to actionable signals (Phase 2 deferred until grade is validated).

**For the user's two questions:**
- *'Maximize signal so we can take positions':* the path to that is fixing the grade first. Auto-promoting Grade A today would amplify noise (current Grade A mean R is around -0.3R per the live backtest). Path 2 (auto-promote with synthetic plans) should wait.
- *'Position health and exits using flow features':* this is **independent of the grade calibration** and can be built immediately. Grade *deltas* (e.g., 'this name was Grade A on entry, today it's Grade C') are much more robust than absolute grades. Recommend Path 3 (flow-aware exits) as the highest-impact next move while Phase 1B-infra runs in parallel.
