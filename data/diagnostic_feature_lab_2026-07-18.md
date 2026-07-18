# Feature lab — Spearman ranking — 2026-07-18 09:00

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **720 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val |
| --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | 28 | +0.553 | +0.754 | 12 |
| `bullish_premium_share` | 283 | +0.160 | +0.309 | 114 |
| `sector_relative_pct` | 167 | +0.142 | +0.071 | 67 |
| `iv_skew_25d` | 36 | -0.123 | -0.208 | 15 |
| `far_otm_call_share` | 195 | -0.105 | -0.044 | 78 |
| `charm_total` | 283 | +0.102 | -0.099 | 114 |
| `realized_vol_regime` | 276 | -0.097 | -0.061 | 111 |
| `expiry_concentration_top1` | 282 | +0.096 | +0.158 | 113 |
| `dollar_delta_weighted_flow` | 195 | +0.086 | +0.159 | 78 |
| `dealer_net_gamma_at_spot` | 95 | +0.073 | +0.087 | 38 |
| `far_otm_put_share` | 195 | +0.071 | -0.090 | 78 |
| `term_slope_30_90` | 283 | +0.056 | +0.074 | 114 |
| `vrp_proxy` | 276 | +0.045 | +0.040 | 111 |
| `unusual_premium_share` | 127 | +0.032 | +0.133 | 51 |
| `dealer_net_delta_at_spot` | 95 | -0.030 | +0.188 | 38 |
| `gex_total` | 283 | -0.029 | +0.086 | 114 |
| `atm_iv_90d` | 283 | +0.008 | +0.037 | 114 |
| `atm_iv_60d` | 283 | +0.003 | +0.040 | 114 |
| `atm_iv_30d` | 283 | -0.003 | +0.047 | 114 |
| `max_pain_dist_pct` | 295 | -0.002 | -0.178 | 118 |
| `vanna_total` | 283 | +0.002 | +0.073 | 114 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | — | — | +0.00 (n=8) | +0.20 (n=10) | +0.84 (n=6) |
| `bullish_premium_share` | — | +0.12 (n=14) | -0.04 (n=103) | +0.22 (n=73) | +0.28 (n=90) |
| `sector_relative_pct` | — | +0.03 (n=6) | +0.19 (n=66) | +0.11 (n=48) | +0.10 (n=45) |
| `iv_skew_25d` | — | — | -0.04 (n=18) | +0.16 (n=11) | +0.26 (n=6) |
| `far_otm_call_share` | — | -0.64 (n=7) | -0.07 (n=79) | -0.15 (n=56) | -0.16 (n=51) |
| `charm_total` | — | -0.16 (n=14) | +0.17 (n=103) | +0.19 (n=73) | -0.10 (n=90) |
| `realized_vol_regime` | — | +0.40 (n=15) | -0.17 (n=93) | -0.06 (n=73) | -0.15 (n=92) |
| `expiry_concentration_top1` | — | -0.44 (n=14) | +0.15 (n=103) | +0.30 (n=73) | -0.04 (n=89) |
| `dollar_delta_weighted_flow` | — | +0.29 (n=7) | +0.03 (n=79) | +0.22 (n=56) | +0.03 (n=51) |
| `dealer_net_gamma_at_spot` | — | — | +0.16 (n=58) | +0.17 (n=28) | — |
| `far_otm_put_share` | — | +0.04 (n=7) | +0.18 (n=79) | -0.10 (n=56) | +0.22 (n=51) |
| `term_slope_30_90` | — | +0.03 (n=14) | -0.08 (n=103) | +0.19 (n=73) | +0.09 (n=90) |
| `vrp_proxy` | — | +0.65 (n=15) | +0.04 (n=93) | +0.07 (n=73) | -0.01 (n=92) |
| `unusual_premium_share` | — | — | -0.04 (n=82) | +0.13 (n=41) | — |
| `dealer_net_delta_at_spot` | — | — | +0.12 (n=58) | -0.00 (n=28) | — |
| `gex_total` | — | +0.02 (n=14) | -0.09 (n=103) | -0.02 (n=73) | +0.03 (n=90) |
| `atm_iv_90d` | — | +0.00 (n=14) | +0.04 (n=103) | -0.16 (n=73) | +0.09 (n=90) |
| `atm_iv_60d` | — | +0.12 (n=14) | +0.04 (n=103) | -0.19 (n=73) | +0.08 (n=90) |
| `atm_iv_30d` | — | +0.01 (n=14) | +0.01 (n=103) | -0.19 (n=73) | +0.10 (n=90) |
| `max_pain_dist_pct` | — | +0.12 (n=15) | +0.03 (n=104) | -0.01 (n=76) | -0.03 (n=97) |
| `vanna_total` | — | +0.09 (n=14) | -0.02 (n=103) | -0.06 (n=73) | +0.04 (n=90) |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `bullish_premium_share` | 283 | +0.160 | +0.309 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `sector_relative_pct` | 167 | +0.142 | +0.071 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
