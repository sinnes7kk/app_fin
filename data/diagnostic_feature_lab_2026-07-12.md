# Feature lab — Spearman ranking — 2026-07-12 22:51

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **639 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val |
| --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | 27 | +0.570 | +0.826 | 11 |
| `bullish_premium_share` | 266 | +0.167 | +0.348 | 107 |
| `sector_relative_pct` | 162 | +0.160 | +0.168 | 65 |
| `iv_skew_25d` | 36 | -0.123 | -0.208 | 15 |
| `far_otm_call_share` | 184 | -0.119 | -0.039 | 74 |
| `charm_total` | 266 | +0.117 | -0.020 | 107 |
| `dollar_delta_weighted_flow` | 184 | +0.110 | +0.233 | 74 |
| `dealer_net_gamma_at_spot` | 93 | +0.104 | +0.231 | 38 |
| `realized_vol_regime` | 259 | -0.099 | -0.016 | 104 |
| `expiry_concentration_top1` | 265 | +0.072 | +0.120 | 106 |
| `vrp_proxy` | 259 | +0.067 | +0.120 | 104 |
| `unusual_premium_share` | 119 | +0.058 | +0.182 | 48 |
| `far_otm_put_share` | 184 | +0.054 | -0.186 | 74 |
| `dealer_net_delta_at_spot` | 93 | -0.048 | +0.155 | 38 |
| `gex_total` | 266 | -0.024 | +0.088 | 107 |
| `term_slope_30_90` | 266 | +0.023 | -0.025 | 107 |
| `atm_iv_60d` | 266 | -0.007 | -0.041 | 107 |
| `vanna_total` | 266 | -0.007 | +0.005 | 107 |
| `atm_iv_30d` | 266 | -0.005 | -0.021 | 107 |
| `max_pain_dist_pct` | 278 | +0.003 | -0.136 | 112 |
| `atm_iv_90d` | 266 | -0.002 | -0.046 | 107 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | — | — | +0.00 (n=8) | +0.20 (n=10) | +0.84 (n=6) |
| `bullish_premium_share` | — | +0.35 (n=11) | -0.05 (n=95) | +0.22 (n=72) | +0.27 (n=85) |
| `sector_relative_pct` | — | +0.50 (n=5) | +0.22 (n=63) | +0.12 (n=47) | +0.10 (n=45) |
| `iv_skew_25d` | — | — | -0.04 (n=18) | +0.16 (n=11) | +0.26 (n=6) |
| `far_otm_call_share` | — | -0.60 (n=6) | -0.11 (n=73) | -0.15 (n=55) | -0.17 (n=48) |
| `charm_total` | — | -0.17 (n=11) | +0.23 (n=95) | +0.18 (n=72) | -0.10 (n=85) |
| `dollar_delta_weighted_flow` | — | +0.37 (n=6) | +0.09 (n=73) | +0.21 (n=55) | +0.03 (n=48) |
| `dealer_net_gamma_at_spot` | — | — | +0.22 (n=56) | +0.17 (n=28) | — |
| `realized_vol_regime` | — | +0.35 (n=12) | -0.16 (n=85) | -0.06 (n=72) | -0.14 (n=87) |
| `expiry_concentration_top1` | — | -0.42 (n=11) | +0.09 (n=95) | +0.30 (n=72) | -0.07 (n=84) |
| `vrp_proxy` | — | +0.59 (n=12) | +0.13 (n=85) | +0.07 (n=72) | -0.03 (n=87) |
| `unusual_premium_share` | — | — | -0.00 (n=74) | +0.13 (n=41) | — |
| `far_otm_put_share` | — | +0.17 (n=6) | +0.16 (n=73) | -0.10 (n=55) | +0.20 (n=48) |
| `dealer_net_delta_at_spot` | — | — | +0.08 (n=56) | -0.00 (n=28) | — |
| `gex_total` | — | +0.08 (n=11) | -0.09 (n=95) | -0.02 (n=72) | +0.05 (n=85) |
| `term_slope_30_90` | — | +0.00 (n=11) | -0.17 (n=95) | +0.19 (n=72) | +0.07 (n=85) |
| `atm_iv_60d` | — | +0.10 (n=11) | -0.01 (n=95) | -0.19 (n=72) | +0.09 (n=85) |
| `vanna_total` | — | +0.37 (n=11) | -0.07 (n=95) | -0.06 (n=72) | +0.04 (n=85) |
| `atm_iv_30d` | — | +0.06 (n=11) | -0.00 (n=95) | -0.19 (n=72) | +0.11 (n=85) |
| `max_pain_dist_pct` | — | +0.30 (n=12) | +0.07 (n=96) | -0.01 (n=75) | -0.04 (n=92) |
| `atm_iv_90d` | — | +0.03 (n=11) | -0.01 (n=95) | -0.16 (n=72) | +0.09 (n=85) |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `bullish_premium_share` | 266 | +0.167 | +0.348 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `sector_relative_pct` | 162 | +0.160 | +0.168 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `dollar_delta_weighted_flow` | 184 | +0.110 | +0.233 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `dealer_net_gamma_at_spot` | 93 | +0.104 | +0.231 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
