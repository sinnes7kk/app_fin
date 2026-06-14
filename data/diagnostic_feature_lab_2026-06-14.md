# Feature lab — Spearman ranking — 2026-06-14 23:10

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **351 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val |
| --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | 20 | +0.564 | +0.762 | 8 |
| `charm_total` | 162 | +0.200 | +0.459 | 65 |
| `far_otm_call_share` | 116 | -0.187 | -0.058 | 47 |
| `far_otm_put_share` | 116 | +0.172 | -0.027 | 47 |
| `sector_relative_pct` | 114 | +0.138 | +0.077 | 46 |
| `realized_vol_regime` | 170 | -0.113 | -0.243 | 68 |
| `max_pain_dist_pct` | 174 | +0.110 | +0.333 | 70 |
| `gex_total` | 162 | -0.103 | -0.038 | 65 |
| `iv_skew_25d` | 26 | -0.086 | -0.088 | 11 |
| `dealer_net_gamma_at_spot` | 65 | +0.070 | +0.186 | 26 |
| `dealer_net_delta_at_spot` | 65 | -0.066 | +0.360 | 26 |
| `vrp_proxy` | 170 | +0.060 | -0.003 | 68 |
| `term_slope_30_90` | 162 | +0.048 | +0.079 | 65 |
| `expiry_concentration_top1` | 162 | +0.047 | +0.075 | 65 |
| `unusual_premium_share` | 78 | -0.046 | -0.047 | 32 |
| `bullish_premium_share` | 162 | +0.038 | +0.306 | 65 |
| `atm_iv_30d` | 162 | -0.034 | -0.050 | 65 |
| `vanna_total` | 162 | -0.031 | -0.204 | 65 |
| `dollar_delta_weighted_flow` | 116 | +0.024 | +0.060 | 47 |
| `atm_iv_60d` | 162 | -0.019 | +0.014 | 65 |
| `atm_iv_90d` | 162 | -0.013 | +0.016 | 65 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | — | — | +0.00 (n=8) | +0.32 (n=7) | — |
| `charm_total` | — | -0.28 (n=7) | +0.26 (n=70) | +0.33 (n=44) | -0.14 (n=41) |
| `far_otm_call_share` | — | — | -0.21 (n=53) | -0.15 (n=36) | -0.21 (n=24) |
| `far_otm_put_share` | — | — | +0.17 (n=53) | +0.09 (n=36) | +0.45 (n=24) |
| `sector_relative_pct` | — | — | +0.21 (n=52) | +0.08 (n=31) | +0.08 (n=28) |
| `realized_vol_regime` | — | +0.14 (n=8) | -0.20 (n=69) | -0.14 (n=46) | -0.24 (n=47) |
| `max_pain_dist_pct` | — | +0.00 (n=8) | +0.13 (n=71) | -0.03 (n=47) | +0.21 (n=48) |
| `gex_total` | — | -0.32 (n=7) | -0.09 (n=70) | -0.29 (n=44) | +0.04 (n=41) |
| `iv_skew_25d` | — | — | -0.07 (n=15) | -0.02 (n=7) | — |
| `dealer_net_gamma_at_spot` | — | — | +0.19 (n=42) | +0.20 (n=17) | — |
| `dealer_net_delta_at_spot` | — | — | +0.07 (n=42) | -0.10 (n=17) | — |
| `vrp_proxy` | — | +0.49 (n=8) | +0.15 (n=69) | -0.01 (n=46) | +0.01 (n=47) |
| `term_slope_30_90` | — | -0.37 (n=7) | -0.10 (n=70) | +0.21 (n=44) | +0.16 (n=41) |
| `expiry_concentration_top1` | — | -0.73 (n=7) | +0.11 (n=70) | +0.23 (n=44) | -0.03 (n=41) |
| `unusual_premium_share` | — | — | -0.01 (n=55) | -0.23 (n=22) | — |
| `bullish_premium_share` | — | -0.02 (n=7) | +0.02 (n=70) | +0.10 (n=44) | +0.01 (n=41) |
| `atm_iv_30d` | — | +0.09 (n=7) | -0.01 (n=70) | +0.01 (n=44) | -0.08 (n=41) |
| `vanna_total` | — | +0.36 (n=7) | -0.11 (n=70) | -0.13 (n=44) | +0.19 (n=41) |
| `dollar_delta_weighted_flow` | — | — | +0.04 (n=53) | +0.12 (n=36) | -0.12 (n=24) |
| `atm_iv_60d` | — | +0.02 (n=7) | +0.01 (n=70) | +0.05 (n=44) | -0.12 (n=41) |
| `atm_iv_90d` | — | -0.04 (n=7) | +0.01 (n=70) | +0.08 (n=44) | -0.09 (n=41) |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `charm_total` | 162 | +0.200 | +0.459 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `sector_relative_pct` | 114 | +0.138 | +0.077 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `max_pain_dist_pct` | 174 | +0.110 | +0.333 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
