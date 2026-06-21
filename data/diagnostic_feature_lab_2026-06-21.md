# Feature lab — Spearman ranking — 2026-06-21 23:15

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **430 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val |
| --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | 22 | +0.607 | +0.750 | 9 |
| `charm_total` | 200 | +0.194 | +0.348 | 80 |
| `sector_relative_pct` | 134 | +0.162 | +0.195 | 54 |
| `far_otm_call_share` | 140 | -0.122 | +0.055 | 56 |
| `realized_vol_regime` | 208 | -0.093 | -0.132 | 84 |
| `iv_skew_25d` | 28 | -0.089 | -0.256 | 12 |
| `bullish_premium_share` | 200 | +0.082 | +0.198 | 80 |
| `far_otm_put_share` | 140 | +0.082 | -0.167 | 56 |
| `expiry_concentration_top1` | 200 | +0.076 | +0.166 | 80 |
| `dealer_net_gamma_at_spot` | 77 | +0.075 | -0.035 | 31 |
| `dealer_net_delta_at_spot` | 77 | -0.071 | +0.084 | 31 |
| `vrp_proxy` | 208 | +0.063 | +0.157 | 84 |
| `max_pain_dist_pct` | 212 | +0.062 | +0.150 | 85 |
| `gex_total` | 200 | -0.056 | +0.006 | 80 |
| `dollar_delta_weighted_flow` | 140 | +0.049 | +0.036 | 56 |
| `term_slope_30_90` | 200 | +0.048 | +0.026 | 80 |
| `vanna_total` | 200 | -0.033 | -0.129 | 80 |
| `unusual_premium_share` | 92 | -0.031 | -0.124 | 37 |
| `atm_iv_30d` | 200 | -0.023 | +0.010 | 80 |
| `atm_iv_60d` | 200 | -0.014 | +0.020 | 80 |
| `atm_iv_90d` | 200 | -0.008 | +0.019 | 80 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | — | — | +0.00 (n=8) | +0.32 (n=7) | +0.72 (n=5) |
| `charm_total` | — | -0.28 (n=7) | +0.23 (n=80) | +0.39 (n=55) | -0.06 (n=55) |
| `sector_relative_pct` | — | — | +0.26 (n=58) | +0.07 (n=40) | +0.09 (n=31) |
| `far_otm_call_share` | — | — | -0.17 (n=61) | -0.10 (n=43) | -0.15 (n=31) |
| `realized_vol_regime` | — | +0.14 (n=8) | -0.16 (n=79) | -0.04 (n=57) | -0.22 (n=61) |
| `iv_skew_25d` | — | — | -0.04 (n=16) | -0.02 (n=7) | — |
| `bullish_premium_share` | — | -0.02 (n=7) | -0.04 (n=80) | +0.18 (n=55) | +0.17 (n=55) |
| `far_otm_put_share` | — | — | +0.20 (n=61) | -0.13 (n=43) | +0.28 (n=31) |
| `expiry_concentration_top1` | — | -0.73 (n=7) | +0.04 (n=80) | +0.36 (n=55) | +0.01 (n=55) |
| `dealer_net_gamma_at_spot` | — | — | +0.15 (n=48) | +0.34 (n=20) | — |
| `dealer_net_delta_at_spot` | — | — | +0.04 (n=48) | -0.06 (n=20) | — |
| `vrp_proxy` | — | +0.49 (n=8) | +0.15 (n=79) | +0.02 (n=57) | -0.05 (n=61) |
| `max_pain_dist_pct` | — | +0.00 (n=8) | +0.06 (n=81) | +0.08 (n=58) | +0.09 (n=62) |
| `gex_total` | — | -0.32 (n=7) | -0.09 (n=80) | -0.07 (n=55) | +0.05 (n=55) |
| `dollar_delta_weighted_flow` | — | — | +0.05 (n=61) | +0.16 (n=43) | -0.14 (n=31) |
| `term_slope_30_90` | — | -0.37 (n=7) | -0.12 (n=80) | +0.26 (n=55) | +0.08 (n=55) |
| `vanna_total` | — | +0.36 (n=7) | -0.06 (n=80) | -0.13 (n=55) | +0.03 (n=55) |
| `unusual_premium_share` | — | — | -0.04 (n=61) | -0.16 (n=28) | — |
| `atm_iv_30d` | — | +0.09 (n=7) | +0.02 (n=80) | -0.09 (n=55) | +0.05 (n=55) |
| `atm_iv_60d` | — | +0.02 (n=7) | +0.03 (n=80) | -0.06 (n=55) | +0.03 (n=55) |
| `atm_iv_90d` | — | -0.04 (n=7) | +0.03 (n=80) | -0.04 (n=55) | +0.04 (n=55) |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `charm_total` | 200 | +0.194 | +0.348 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `sector_relative_pct` | 134 | +0.162 | +0.195 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `far_otm_call_share` | 140 | -0.122 | +0.055 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
