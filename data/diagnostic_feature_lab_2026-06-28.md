# Feature lab — Spearman ranking — 2026-06-28 23:03

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **493 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val |
| --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | 23 | +0.650 | +0.818 | 10 |
| `sector_relative_pct` | 144 | +0.151 | +0.185 | 58 |
| `charm_total` | 224 | +0.149 | +0.119 | 90 |
| `iv_skew_25d` | 33 | -0.142 | -0.307 | 14 |
| `bullish_premium_share` | 224 | +0.111 | +0.230 | 90 |
| `realized_vol_regime` | 221 | -0.106 | -0.103 | 89 |
| `far_otm_call_share` | 154 | -0.097 | +0.036 | 62 |
| `dealer_net_gamma_at_spot` | 87 | +0.086 | +0.150 | 35 |
| `dollar_delta_weighted_flow` | 154 | +0.076 | +0.109 | 62 |
| `vrp_proxy` | 221 | +0.074 | +0.222 | 89 |
| `dealer_net_delta_at_spot` | 87 | -0.074 | +0.109 | 35 |
| `expiry_concentration_top1` | 224 | +0.061 | +0.107 | 90 |
| `far_otm_put_share` | 154 | +0.053 | -0.182 | 62 |
| `max_pain_dist_pct` | 236 | +0.036 | -0.044 | 95 |
| `unusual_premium_share` | 105 | +0.018 | -0.030 | 42 |
| `gex_total` | 224 | -0.015 | +0.034 | 90 |
| `atm_iv_90d` | 224 | +0.009 | +0.014 | 90 |
| `vanna_total` | 224 | -0.009 | -0.075 | 90 |
| `atm_iv_30d` | 224 | +0.007 | +0.029 | 90 |
| `term_slope_30_90` | 224 | +0.007 | -0.050 | 90 |
| `atm_iv_60d` | 224 | +0.004 | +0.024 | 90 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | — | — | +0.00 (n=8) | +0.50 (n=8) | +0.72 (n=5) |
| `sector_relative_pct` | — | — | +0.22 (n=60) | +0.09 (n=45) | +0.11 (n=34) |
| `charm_total` | — | -0.28 (n=7) | +0.24 (n=89) | +0.25 (n=62) | -0.09 (n=63) |
| `iv_skew_25d` | — | — | -0.05 (n=17) | +0.15 (n=9) | +0.26 (n=6) |
| `bullish_premium_share` | — | -0.02 (n=7) | -0.04 (n=89) | +0.21 (n=62) | +0.18 (n=63) |
| `realized_vol_regime` | — | +0.14 (n=8) | -0.18 (n=81) | -0.06 (n=63) | -0.18 (n=66) |
| `far_otm_call_share` | — | — | -0.13 (n=68) | -0.12 (n=48) | -0.11 (n=33) |
| `dealer_net_gamma_at_spot` | — | — | +0.21 (n=53) | +0.21 (n=25) | — |
| `dollar_delta_weighted_flow` | — | — | +0.08 (n=68) | +0.18 (n=48) | -0.13 (n=33) |
| `vrp_proxy` | — | +0.49 (n=8) | +0.13 (n=81) | +0.00 (n=63) | +0.04 (n=66) |
| `dealer_net_delta_at_spot` | — | — | +0.06 (n=53) | -0.05 (n=25) | — |
| `expiry_concentration_top1` | — | -0.73 (n=7) | +0.08 (n=89) | +0.34 (n=62) | -0.08 (n=63) |
| `far_otm_put_share` | — | — | +0.16 (n=68) | -0.14 (n=48) | +0.30 (n=33) |
| `max_pain_dist_pct` | — | +0.00 (n=8) | +0.08 (n=90) | -0.00 (n=65) | +0.03 (n=70) |
| `unusual_premium_share` | — | — | +0.00 (n=69) | -0.05 (n=33) | — |
| `gex_total` | — | -0.32 (n=7) | -0.09 (n=89) | -0.02 (n=62) | +0.14 (n=63) |
| `atm_iv_90d` | — | -0.04 (n=7) | +0.00 (n=89) | -0.06 (n=62) | +0.10 (n=63) |
| `vanna_total` | — | +0.36 (n=7) | -0.07 (n=89) | -0.08 (n=62) | +0.09 (n=63) |
| `atm_iv_30d` | — | +0.09 (n=7) | +0.00 (n=89) | -0.10 (n=62) | +0.14 (n=63) |
| `term_slope_30_90` | — | -0.37 (n=7) | -0.15 (n=89) | +0.18 (n=62) | +0.02 (n=63) |
| `atm_iv_60d` | — | +0.02 (n=7) | +0.00 (n=89) | -0.09 (n=62) | +0.09 (n=63) |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `sector_relative_pct` | 144 | +0.151 | +0.185 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `charm_total` | 224 | +0.149 | +0.119 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `bullish_premium_share` | 224 | +0.111 | +0.230 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
